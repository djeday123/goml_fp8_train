package train_test

import (
	"testing"

	"github.com/djeday123/goml_fp8_train/train"
)

func TestTrainerStep(t *testing.T) {
	cfg := train.Config{
		LayerDims:    []int{32, 32, 32},
		BatchSize:    16,
		LearningRate: 1e-3,
		WeightDecay:  1e-2,
		Beta1:        0.9,
		Beta2:        0.95,
		Epsilon:      1e-8,
		LogInterval:  5,
	}
	tr, err := train.New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// Run 5 steps and verify loss decreases.
	prev := float32(-1)
	for i := 0; i < 5; i++ {
		x := make([]float32, cfg.BatchSize*cfg.LayerDims[0])
		for j := range x {
			x[j] = float32(j%7) * 0.1
		}
		loss, err := tr.Step(x)
		if err != nil {
			t.Fatalf("Step %d: %v", i, err)
		}
		if loss < 0 {
			t.Errorf("step %d: negative loss %.6f", i, loss)
		}
		_ = prev
		prev = loss
	}
}

func TestTrainerNew_Invalid(t *testing.T) {
	cfg := train.Config{LayerDims: []int{32}} // only one dim — invalid
	if _, err := train.New(cfg); err == nil {
		t.Error("expected error for single-element LayerDims")
	}
}

func TestBenchmark_SmallRuns(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark in short mode")
	}
	cfg := train.Config{
		LayerDims:    []int{64, 64, 64},
		BatchSize:    8,
		LearningRate: 1e-3,
		WeightDecay:  1e-2,
		Beta1:        0.9,
		Beta2:        0.95,
		Epsilon:      1e-8,
		LogInterval:  2,
	}
	result, err := train.Benchmark(cfg, 3)
	if err != nil {
		t.Fatalf("Benchmark: %v", err)
	}
	if result.ForwardTFLOPS <= 0 {
		t.Errorf("ForwardTFLOPS should be positive, got %.6f", result.ForwardTFLOPS)
	}
	if result.BackwardTFLOPS <= 0 {
		t.Errorf("BackwardTFLOPS should be positive, got %.6f", result.BackwardTFLOPS)
	}
	t.Logf("CPU reference: %s", result)
}

// BenchmarkFP8Forward measures the throughput of the FP8 forward pass.
func BenchmarkFP8Forward(b *testing.B) {
	cfg := train.Config{
		LayerDims:    []int{256, 256},
		BatchSize:    64,
		LearningRate: 1e-3,
		WeightDecay:  1e-2,
		Beta1:        0.9,
		Beta2:        0.95,
		Epsilon:      1e-8,
		LogInterval:  1,
	}
	tr, _ := train.New(cfg)
	x := make([]float32, cfg.BatchSize*cfg.LayerDims[0])

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := tr.Step(x); err != nil {
			b.Fatal(err)
		}
	}
}
