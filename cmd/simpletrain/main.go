package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/djeday123/goml/backend"
	_ "github.com/djeday123/goml/backend/cpu"
	"github.com/djeday123/goml/nn"
	"github.com/djeday123/goml/ops"
	"github.com/djeday123/goml/tensor"
	"github.com/djeday123/goml/tokenizer"
)

// SimpleLM: Embedding → LayerNorm → Linear (no attention, no transformer)
// This validates the training pipeline independent of the attention backward.
type SimpleLM struct {
	Embed  *nn.Embedding
	Norm   *nn.LayerNorm
	Hidden *nn.Linear
	Act    string // "gelu"
	Output *nn.Linear
	Config nn.ModelConfig
}

func NewSimpleLM(cfg nn.ModelConfig, dev backend.Device) (*SimpleLM, error) {
	emb, _ := nn.NewEmbedding(cfg.VocabSize, cfg.Dim, dev)
	norm, _ := nn.NewLayerNorm(cfg.Dim, 1e-5, dev)
	hidden, _ := nn.NewLinear(cfg.Dim, cfg.FFNHiddenDim, true, dev)
	output, _ := nn.NewLinear(cfg.FFNHiddenDim, cfg.VocabSize, false, dev)

	return &SimpleLM{
		Embed: emb, Norm: norm, Hidden: hidden, Output: output,
		Config: cfg, Act: "gelu",
	}, nil
}

func (m *SimpleLM) Parameters() []*tensor.Tensor {
	var p []*tensor.Tensor
	p = append(p, m.Embed.Parameters()...)
	p = append(p, m.Norm.Parameters()...)
	p = append(p, m.Hidden.Parameters()...)
	p = append(p, m.Output.Parameters()...)
	return p
}

func (m *SimpleLM) CountParams() int {
	total := 0
	for _, p := range m.Parameters() {
		total += p.NumElements()
	}
	return total
}

// ForwardAndBackward does forward + backward manually with verified gradients.
// Returns loss value.
func (m *SimpleLM) ForwardAndBackward(inputs, targets *tensor.Tensor) (float64, error) {
	seqLen := inputs.Shape()[1]
	dim := m.Config.Dim

	// ---- Forward ----

	// 1. Embedding: [1, seqLen] → [seqLen, dim]
	tokens1D, _ := tensor.FromSlice(inputs.ToInt64Slice(), tensor.Shape{seqLen})
	emb, _ := m.Embed.Forward(tokens1D)

	// Reshape to [1, seqLen, dim] for 3D ops
	embData := emb.ToFloat32Slice()
	emb3D, _ := tensor.FromSlice(embData, tensor.Shape{1, seqLen, dim})

	// 2. LayerNorm
	normed, _ := m.Norm.Forward(emb3D)

	// 3. Hidden linear: [1, seqLen, dim] → [1, seqLen, hiddenDim]
	// Flatten to [seqLen, dim] for linear
	normedFlat, _ := tensor.FromSlice(normed.ToFloat32Slice(), tensor.Shape{seqLen, dim})
	hiddenOut, _ := m.Hidden.Forward(normedFlat)

	// 4. GELU activation
	hiddenAct := geluFwd(hiddenOut)

	// 5. Output: [seqLen, hiddenDim] → [seqLen, vocabSize]
	logitsFlat, _ := m.Output.Forward(hiddenAct)

	// Reshape to [1, seqLen, vocabSize]
	logits, _ := tensor.FromSlice(logitsFlat.ToFloat32Slice(),
		tensor.Shape{1, seqLen, m.Config.VocabSize})

	// 6. Cross-entropy loss
	loss, _ := ops.CrossEntropyLoss(logits, targets)
	lossVal := float64(loss.ToFloat32Slice()[0])

	// ---- Backward ----

	// dLogits: [1, seqLen, vocabSize]
	dLogits, _ := ops.CrossEntropyBackward(logits, targets)

	// Flatten to [seqLen, vocabSize]
	dLogitsFlat, _ := tensor.FromSlice(dLogits.ToFloat32Slice(),
		tensor.Shape{seqLen, m.Config.VocabSize})

	// Backward through Output: dHiddenAct [seqLen, hiddenDim]
	dHiddenAct, _ := m.Output.Backward(hiddenAct, dLogitsFlat)

	// Backward through GELU
	dHidden := geluBwd(hiddenOut, dHiddenAct)

	// Backward through Hidden linear: dNormedFlat [seqLen, dim]
	dNormedFlat, _ := m.Hidden.Backward(normedFlat, dHidden)

	// Reshape back to [1, seqLen, dim] for LayerNorm backward
	dNormed3D, _ := tensor.FromSlice(dNormedFlat.ToFloat32Slice(),
		tensor.Shape{1, seqLen, dim})

	// Backward through LayerNorm → gives gradient w.r.t embedding output
	dEmb3D, _ := m.Norm.Backward(emb3D, dNormed3D)

	// Backward through Embedding
	dEmb2D, _ := tensor.FromSlice(dEmb3D.ToFloat32Slice(),
		tensor.Shape{seqLen, dim})
	m.Embed.Backward(tokens1D, dEmb2D)

	return lossVal, nil
}

func geluFwd(x *tensor.Tensor) *tensor.Tensor {
	data := x.ToFloat32Slice()
	c := float32(math.Sqrt(2.0 / math.Pi))
	out := make([]float32, len(data))
	for i, v := range data {
		out[i] = 0.5 * v * (1 + float32(math.Tanh(float64(c*(v+0.044715*v*v*v)))))
	}
	t, _ := tensor.FromSlice(out, x.Shape())
	return t
}

func geluBwd(x, dout *tensor.Tensor) *tensor.Tensor {
	xData := x.ToFloat32Slice()
	dData := dout.ToFloat32Slice()
	c := float64(math.Sqrt(2.0 / math.Pi))
	out := make([]float32, len(xData))
	for i, v := range xData {
		vf := float64(v)
		inner := c * (vf + 0.044715*vf*vf*vf)
		tanh := math.Tanh(inner)
		dtanh := 1 - tanh*tanh
		dinner := c * (1 + 3*0.044715*vf*vf)
		grad := 0.5*(1+tanh) + 0.5*vf*dtanh*dinner
		out[i] = dData[i] * float32(grad)
	}
	t, _ := tensor.FromSlice(out, x.Shape())
	return t
}

func main() {
	fmt.Println("=== GoML — SimpleLM Training ===")

	data, err := os.ReadFile("data/shakespeare.txt")
	if err != nil {
		data = []byte("To be, or not to be, that is the question:\nWhether 'tis nobler in the mind to suffer\nThe slings and arrows of outrageous fortune,\nOr to take arms against a sea of troubles,\n")
	}

	tok := tokenizer.NewByteTokenizer()
	allTokens := tok.Encode(string(data))
	splitIdx := int(float64(len(allTokens)) * 0.9)
	trainTokens := allTokens[:splitIdx]
	evalTokens := allTokens[splitIdx:]

	fmt.Printf("Data: %d tokens (train: %d, eval: %d)\n", len(allTokens), len(trainTokens), len(evalTokens))

	cfg := nn.ModelConfig{
		VocabSize: 256, Dim: 64, NumLayers: 0, NumHeads: 0,
		FFNHiddenDim: 256, MaxSeqLen: 64, NormEps: 1e-5,
	}

	model, _ := NewSimpleLM(cfg, backend.CPU0)
	fmt.Printf("Model: %d params\n", model.CountParams())

	// AdamW manually
	params := model.Parameters()
	lr := 1e-3
	beta1, beta2 := 0.9, 0.999
	eps := 1e-8
	wd := 0.01

	m := make([][]float32, len(params))
	v := make([][]float32, len(params))
	for i, p := range params {
		n := p.NumElements()
		m[i] = make([]float32, n)
		v[i] = make([]float32, n)
	}

	seqLen := 32
	steps := 2000

	fmt.Printf("Config: seqLen=%d, lr=%.0e, steps=%d\n\n", seqLen, lr, steps)

	// ---- Gradient check first ----
	fmt.Println("--- Gradient Check ---")
	{
		inp, _ := tensor.FromSlice([]int64{72, 101, 108, 108}, tensor.Shape{1, 4})
		tgt, _ := tensor.FromSlice([]int64{101, 108, 108, 111}, tensor.Shape{1, 4})

		// Zero grads
		for _, p := range params {
			p.SetGrad(nil)
		}

		lossBase, _ := model.ForwardAndBackward(inp, tgt)
		fmt.Printf("Base loss: %.4f (expected ~%.4f)\n", lossBase, math.Log(256))

		epsilon := 1e-4
		for pi, p := range params {
			if p.Grad() == nil {
				continue
			}
			pData := p.ToFloat32Slice()
			gData := p.Grad().ToFloat32Slice()

			// Check first element
			orig := pData[0]

			pData[0] = orig + float32(epsilon)
			// Need fresh forward without backward
			inp2, _ := tensor.FromSlice([]int64{72, 101, 108, 108}, tensor.Shape{1, 4})
			tgt2, _ := tensor.FromSlice([]int64{101, 108, 108, 111}, tensor.Shape{1, 4})
			for _, pp := range params {
				pp.SetGrad(nil)
			}
			lPlus, _ := model.ForwardAndBackward(inp2, tgt2)

			pData[0] = orig - float32(epsilon)
			inp3, _ := tensor.FromSlice([]int64{72, 101, 108, 108}, tensor.Shape{1, 4})
			tgt3, _ := tensor.FromSlice([]int64{101, 108, 108, 111}, tensor.Shape{1, 4})
			for _, pp := range params {
				pp.SetGrad(nil)
			}
			lMinus, _ := model.ForwardAndBackward(inp3, tgt3)

			pData[0] = orig

			numGrad := (lPlus - lMinus) / (2 * epsilon)

			// Restore original grad
			for _, pp := range params {
				pp.SetGrad(nil)
			}
			model.ForwardAndBackward(inp, tgt)

			anaGrad := float64(gData[0])
			relErr := math.Abs(numGrad-anaGrad) / (math.Abs(numGrad) + math.Abs(anaGrad) + 1e-8)

			status := "✓"
			if relErr > 0.01 {
				status = "✗"
			}
			fmt.Printf("  param %d: ana=%.6f num=%.6f rel_err=%.6f %s\n", pi, anaGrad, numGrad, relErr, status)
			if pi >= 5 {
				break
			} // check first few
		}
	}

	// ---- Training ----
	fmt.Println("\n--- Training ---")
	totalStart := time.Now()
	smoothLoss := float64(0)

	for step := 1; step <= steps; step++ {
		// Zero grads
		for _, p := range params {
			p.SetGrad(nil)
		}

		// Get random sequence
		maxStart := len(trainTokens) - seqLen - 1
		start := rand.Intn(maxStart)
		inputData := trainTokens[start : start+seqLen]
		targetData := trainTokens[start+1 : start+seqLen+1]
		inputs, _ := tensor.FromSlice(inputData, tensor.Shape{1, seqLen})
		targets, _ := tensor.FromSlice(targetData, tensor.Shape{1, seqLen})

		// Forward + backward
		lossVal, err := model.ForwardAndBackward(inputs, targets)
		if err != nil || math.IsNaN(lossVal) {
			fmt.Printf("step %d: error or NaN\n", step)
			continue
		}

		if smoothLoss == 0 {
			smoothLoss = lossVal
		} else {
			smoothLoss = 0.99*smoothLoss + 0.01*lossVal
		}

		// LR schedule
		warmup := 100
		currentLR := lr
		if step < warmup {
			currentLR = lr * float64(step) / float64(warmup)
		} else {
			progress := float64(step-warmup) / float64(steps-warmup)
			currentLR = lr*0.1 + 0.5*(lr-lr*0.1)*(1+math.Cos(math.Pi*progress))
		}

		// AdamW step
		t := float64(step)
		bc1 := 1.0 - math.Pow(beta1, t)
		bc2 := 1.0 - math.Pow(beta2, t)

		for i, p := range params {
			grad := p.Grad()
			if grad == nil {
				continue
			}
			pData := p.ToFloat32Slice()
			gData := grad.ToFloat32Slice()

			for j := range pData {
				g := gData[j]
				m[i][j] = float32(beta1)*m[i][j] + float32(1-beta1)*g
				v[i][j] = float32(beta2)*v[i][j] + float32(1-beta2)*g*g
				mHat := float64(m[i][j]) / bc1
				vHat := float64(v[i][j]) / bc2
				update := mHat / (math.Sqrt(vHat) + eps)
				pData[j] -= float32(currentLR) * (float32(update) + float32(wd)*pData[j])
			}
		}

		if step%100 == 0 || step == 1 {
			tokSec := float64(seqLen) / time.Since(totalStart).Seconds() * float64(step)
			_ = tokSec
			fmt.Printf("step %4d | loss %.4f (smooth %.4f) | lr %.1e\n",
				step, lossVal, smoothLoss, currentLR)
		}
	}

	totalTime := time.Since(totalStart)
	fmt.Printf("\nDone in %v\n", totalTime)
	fmt.Printf("Final smooth loss: %.4f (random: %.4f)\n", smoothLoss, math.Log(256))

	// Generate
	fmt.Println("\n--- Generation ---")
	for _, prompt := range []string{"The ", "To be", "KING "} {
		tokens := tok.Encode(prompt)
		for i := 0; i < 100; i++ {
			window := tokens
			if len(window) > cfg.MaxSeqLen {
				window = window[len(window)-cfg.MaxSeqLen:]
			}
			input, _ := tensor.FromSlice(window, tensor.Shape{1, len(window)})

			// Just forward, no backward
			seqL := len(window)
			tokens1D, _ := tensor.FromSlice(input.ToInt64Slice(), tensor.Shape{seqL})
			emb, _ := model.Embed.Forward(tokens1D)
			embData := emb.ToFloat32Slice()
			emb3D, _ := tensor.FromSlice(embData, tensor.Shape{1, seqL, cfg.Dim})
			normed, _ := model.Norm.Forward(emb3D)
			normedFlat, _ := tensor.FromSlice(normed.ToFloat32Slice(), tensor.Shape{seqL, cfg.Dim})
			hiddenOut, _ := model.Hidden.Forward(normedFlat)
			hiddenAct := geluFwd(hiddenOut)
			logitsFlat, _ := model.Output.Forward(hiddenAct)

			logitsData := logitsFlat.ToFloat32Slice()
			lastOff := (seqL - 1) * cfg.VocabSize
			lastLogits := logitsData[lastOff : lastOff+cfg.VocabSize]

			// Temperature sampling
			temp := float32(0.8)
			for k := range lastLogits {
				lastLogits[k] /= temp
			}

			// Softmax
			maxV := float32(-1e9)
			for _, v := range lastLogits {
				if v > maxV {
					maxV = v
				}
			}
			sumExp := float32(0)
			for k := range lastLogits {
				lastLogits[k] = float32(math.Exp(float64(lastLogits[k] - maxV)))
				sumExp += lastLogits[k]
			}
			for k := range lastLogits {
				lastLogits[k] /= sumExp
			}

			// Sample
			r := rand.Float32()
			cum := float32(0)
			next := int64(0)
			for k, p := range lastLogits {
				cum += p
				if r < cum {
					next = int64(k)
					break
				}
			}
			tokens = append(tokens, next)
		}
		fmt.Printf("%q → %q\n\n", prompt, tok.Decode(tokens))
	}
}
