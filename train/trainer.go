// Package train implements FP8 mixed-precision training on GoTorch-style
// models. The Trainer wraps a sequence of FP8 linear layers and drives the
// forward / backward / optimiser-update loop.
//
// Mixed-precision strategy
// ------------------------
//   • Forward  : FP8 E4M3FN  (activations + weights)
//   • Backward : FP8 E5M2    (upstream gradients)
//   • Master W : FP32        (weight storage for the optimiser)
//   • Optimiser: AdamW in FP32
package train

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/djeday123/goml_fp8_train/fp8"
)

// Config holds hyper-parameters for the FP8 trainer.
type Config struct {
	// Layer dimensions.
	LayerDims []int // e.g., [4096, 4096, 4096] gives two linear layers

	// Batch size.
	BatchSize int

	// Optimiser.
	LearningRate float32
	WeightDecay  float32
	Beta1        float32
	Beta2        float32
	Epsilon      float32

	// Logging interval (in steps).
	LogInterval int
}

// DefaultConfig returns a Config tuned for a typical large transformer layer.
func DefaultConfig() Config {
	return Config{
		LayerDims:    []int{4096, 4096, 4096},
		BatchSize:    2048,
		LearningRate: 1e-3,
		WeightDecay:  1e-2,
		Beta1:        0.9,
		Beta2:        0.95,
		Epsilon:      1e-8,
		LogInterval:  10,
	}
}

// Trainer holds state for one FP8 training run.
type Trainer struct {
	cfg    Config
	layers []*fp8.Linear

	// AdamW moment vectors (per weight, per layer).
	m1     [][]float32 // first moment
	m2     [][]float32 // second moment
	step   int
}

// New creates a Trainer from the given Config.
func New(cfg Config) (*Trainer, error) {
	if len(cfg.LayerDims) < 2 {
		return nil, fmt.Errorf("train: LayerDims must have at least 2 entries")
	}
	t := &Trainer{cfg: cfg}

	for i := 0; i < len(cfg.LayerDims)-1; i++ {
		in := cfg.LayerDims[i]
		out := cfg.LayerDims[i+1]
		t.layers = append(t.layers, fp8.NewLinear(in, out))
		t.m1 = append(t.m1, make([]float32, in*out))
		t.m2 = append(t.m2, make([]float32, in*out))
	}
	return t, nil
}

// Step runs one training step (forward + backward + optimiser update).
// x is the input batch of shape (BatchSize × InFeatures) flattened.
// Returns the scalar loss (MSE vs zero target).
func (t *Trainer) Step(x []float32) (float32, error) {
	t.step++

	// ── Forward pass ──────────────────────────────────────────────────────
	acts := [][]float32{x}
	for _, layer := range t.layers {
		y, err := layer.Forward(acts[len(acts)-1], t.cfg.BatchSize)
		if err != nil {
			return 0, fmt.Errorf("forward layer: %w", err)
		}
		acts = append(acts, y)
	}

	// Compute MSE loss against zero target.
	final := acts[len(acts)-1]
	loss := mse(final)

	// ── Backward pass ─────────────────────────────────────────────────────
	// dL/dy = 2 * y / numel  (gradient of MSE w.r.t. final activation)
	dY := make([]float32, len(final))
	scale := float32(2.0) / float32(len(final))
	for i, v := range final {
		dY[i] = v * scale
	}

	for i := len(t.layers) - 1; i >= 0; i-- {
		var err error
		dY, err = t.layers[i].Backward(dY, t.cfg.BatchSize)
		if err != nil {
			return 0, fmt.Errorf("backward layer %d: %w", i, err)
		}
	}

	// ── AdamW optimiser ───────────────────────────────────────────────────
	t.adamwUpdate()

	return loss, nil
}

// adamwUpdate applies AdamW to each layer's master weights in FP32.
func (t *Trainer) adamwUpdate() {
	cfg := t.cfg
	beta1 := float64(cfg.Beta1)
	beta2 := float64(cfg.Beta2)
	lr := float64(cfg.LearningRate)
	eps := float64(cfg.Epsilon)
	wd := float64(cfg.WeightDecay)

	bias1 := 1.0 - math.Pow(beta1, float64(t.step))
	bias2 := 1.0 - math.Pow(beta2, float64(t.step))
	lrCorr := lr * math.Sqrt(bias2) / bias1

	for i, layer := range t.layers {
		m1 := t.m1[i]
		m2 := t.m2[i]
		g := layer.GradWeight
		w := layer.MasterWeight

		for j := range w {
			gj := float64(g[j])
			wj := float64(w[j])

			// Moment updates.
			m1[j] = float32(beta1*float64(m1[j]) + (1-beta1)*gj)
			m2[j] = float32(beta2*float64(m2[j]) + (1-beta2)*gj*gj)

			// Weight update (weight decay acts on master weight).
			update := lrCorr*(float64(m1[j])/
				(math.Sqrt(float64(m2[j]))+eps)) + lr*wd*wj
			w[j] = float32(wj - update)
		}

		// Sync FP8 weight from updated master weight.
		layer.Weight.QuantizeFrom(layer.MasterWeight)
		layer.ZeroGrad()
	}
}

// BenchmarkResult holds the result of a TFLOPS benchmark.
type BenchmarkResult struct {
	ForwardTFLOPS  float64
	BackwardTFLOPS float64
	ForwardFLOPs   int64
	BackwardFLOPs  int64
	TotalSteps     int
	Duration       time.Duration
}

// String returns a human-readable summary of the benchmark.
func (r BenchmarkResult) String() string {
	fwdStr := formatFLOPS(r.ForwardTFLOPS)
	bwdStr := formatFLOPS(r.BackwardTFLOPS)
	return fmt.Sprintf(
		"FP8 Benchmark  │  Forward: %s  │  Backward: %s  │  Steps: %d  │  Time: %s",
		fwdStr, bwdStr, r.TotalSteps, r.Duration,
	)
}

// formatFLOPS returns a human-readable string for a throughput value expressed
// in TFLOPS, automatically choosing an appropriate unit (TFLOPS / GFLOPS /
// MFLOPS).
func formatFLOPS(tflops float64) string {
	switch {
	case tflops >= 0.1:
		return fmt.Sprintf("%.1f TFLOPS", tflops)
	case tflops >= 1e-4:
		return fmt.Sprintf("%.1f GFLOPS", tflops*1e3)
	default:
		return fmt.Sprintf("%.1f MFLOPS", tflops*1e6)
	}
}

// Benchmark runs numSteps training steps and computes TFLOPS for the forward
// and backward passes separately.
//
// FLOPs are counted as 2 * M * N * K per matrix multiply (the standard
// multiply-add count). For each linear layer:
//   forward  : 1 GEMM  → 2 * B * InFeatures * OutFeatures
//   backward : 2 GEMMs → 2 * (2 * B * InFeatures * OutFeatures)
func Benchmark(cfg Config, numSteps int) (BenchmarkResult, error) {
	t, err := New(cfg)
	if err != nil {
		return BenchmarkResult{}, err
	}

	// Compute theoretical FLOPs per step.
	fwdFLOPs := int64(0)
	bwdFLOPs := int64(0)
	for i := 0; i < len(cfg.LayerDims)-1; i++ {
		in := int64(cfg.LayerDims[i])
		out := int64(cfg.LayerDims[i+1])
		b := int64(cfg.BatchSize)
		fwdFLOPs += 2 * b * in * out
		bwdFLOPs += 2 * (2 * b * in * out) // dX + dW
	}

	// Warm-up.
	x0 := randInput(cfg.BatchSize, cfg.LayerDims[0])
	for i := 0; i < 2; i++ {
		if _, err := t.Step(x0); err != nil {
			return BenchmarkResult{}, err
		}
	}

	// Timed run.
	start := time.Now()
	for s := 0; s < numSteps; s++ {
		x := randInput(cfg.BatchSize, cfg.LayerDims[0])
		if _, err := t.Step(x); err != nil {
			return BenchmarkResult{}, err
		}
	}
	elapsed := time.Since(start)

	secTotal := elapsed.Seconds()
	secPerStep := secTotal / float64(numSteps)

	fwdTFLOPS := float64(fwdFLOPs) / secPerStep / 1e12
	bwdTFLOPS := float64(bwdFLOPs) / secPerStep / 1e12

	return BenchmarkResult{
		ForwardTFLOPS:  fwdTFLOPS,
		BackwardTFLOPS: bwdTFLOPS,
		ForwardFLOPs:   fwdFLOPs,
		BackwardFLOPs:  bwdFLOPs,
		TotalSteps:     numSteps,
		Duration:       elapsed,
	}, nil
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func mse(y []float32) float32 {
	sum := float64(0)
	for _, v := range y {
		sum += float64(v) * float64(v)
	}
	return float32(sum / float64(len(y)))
}

func randInput(batchSize, inFeatures int) []float32 {
	x := make([]float32, batchSize*inFeatures)
	for i := range x {
		x[i] = float32(rand.NormFloat64() * 0.02)
	}
	return x
}
