package main

import (
	"fmt"
	"math"

	"github.com/djeday123/goml/backend"
	_ "github.com/djeday123/goml/backend/cpu"
	"github.com/djeday123/goml/nn"
	"github.com/djeday123/goml/ops"
	"github.com/djeday123/goml/tensor"
)

func main() {
	fmt.Println("=== Gradient Check ===\n")

	// Tiny model: 1 layer, dim=8, 2 heads
	cfg := nn.ModelConfig{
		VocabSize:    16,
		Dim:          8,
		NumLayers:    1,
		NumHeads:     2,
		FFNHiddenDim: 16,
		UseSwiGLU:    true,
		MaxSeqLen:    8,
		NormEps:      1e-5,
	}

	model, _ := nn.NewLLM(cfg, backend.CPU0)
	fmt.Printf("Model: %d params\n", model.CountParameters())

	// Fixed input
	inputs, _ := tensor.FromSlice([]int64{1, 3, 5, 7}, tensor.Shape{1, 4})
	targets, _ := tensor.FromSlice([]int64{3, 5, 7, 9}, tensor.Shape{1, 4})

	// Test 1: Can we compute loss?
	logits, cache, err := model.ForwardWithCache(inputs)
	if err != nil {
		panic(err)
	}
	loss, _ := ops.CrossEntropyLoss(logits, targets)
	lossVal := loss.ToFloat32Slice()[0]
	fmt.Printf("Initial loss: %.6f (expected ~%.4f = ln(%d))\n", lossVal, math.Log(float64(cfg.VocabSize)), cfg.VocabSize)

	// Test 2: Compute analytical gradients
	dLogits, _ := ops.CrossEntropyBackward(logits, targets)

	// Zero all grads
	for _, p := range model.Parameters() {
		p.SetGrad(nil)
	}

	err = model.Backward(cache, dLogits)
	if err != nil {
		panic(fmt.Sprintf("Backward error: %v", err))
	}

	// Test 3: Numerical gradient check on a few parameters
	fmt.Println("\n--- Numerical Gradient Check ---")
	eps := float64(1e-4)

	params := model.Parameters()
	paramsToCheck := []struct {
		name string
		idx  int
	}{
		{"output.weight", len(params) - 1},
		{"final_norm.gamma", len(params) - 3},
		{"layer0.ffn.w1.weight", 6},
		{"layer0.attn.wq.weight", 2},
		{"embedding", 0},
	}

	for _, pc := range paramsToCheck {
		if pc.idx >= len(params) {
			continue
		}
		p := params[pc.idx]
		grad := p.Grad()
		if grad == nil {
			fmt.Printf("%-25s: NO GRADIENT!\n", pc.name)
			continue
		}

		pData := p.ToFloat32Slice()
		gData := grad.ToFloat32Slice()

		// Check first 3 elements
		maxErr := float64(0)
		for j := 0; j < 3 && j < len(pData); j++ {
			original := pData[j]

			// f(x + eps)
			pData[j] = original + float32(eps)
			logitsPlus, _ := model.Forward(inputs)
			lossPlus, _ := ops.CrossEntropyLoss(logitsPlus, targets)

			// f(x - eps)
			pData[j] = original - float32(eps)
			logitsMinus, _ := model.Forward(inputs)
			lossMinus, _ := ops.CrossEntropyLoss(logitsMinus, targets)

			// Restore
			pData[j] = original

			numGrad := (float64(lossPlus.ToFloat32Slice()[0]) - float64(lossMinus.ToFloat32Slice()[0])) / (2 * eps)
			anaGrad := float64(gData[j])

			relErr := math.Abs(numGrad-anaGrad) / (math.Abs(numGrad) + math.Abs(anaGrad) + 1e-8)
			if relErr > maxErr {
				maxErr = relErr
			}

			if j == 0 {
				fmt.Printf("%-25s: ana=%.6f num=%.6f rel_err=%.6f", pc.name, anaGrad, numGrad, relErr)
			}
		}

		status := "✓"
		if maxErr > 0.01 {
			status = "✗ BAD"
		} else if maxErr > 0.001 {
			status = "~ OK"
		}
		fmt.Printf(" max_err=%.6f %s\n", maxErr, status)
	}

	// Test 4: Try a single gradient step and see if loss decreases
	fmt.Println("\n--- Single Step Test ---")
	fmt.Printf("Loss before: %.6f\n", lossVal)

	lr := float32(0.01)
	for _, p := range model.Parameters() {
		if p.Grad() == nil {
			continue
		}
		pData := p.ToFloat32Slice()
		gData := p.Grad().ToFloat32Slice()
		for i := range pData {
			pData[i] -= lr * gData[i]
		}
	}

	logits2, _ := model.Forward(inputs)
	loss2, _ := ops.CrossEntropyLoss(logits2, targets)
	lossVal2 := loss2.ToFloat32Slice()[0]
	fmt.Printf("Loss after:  %.6f\n", lossVal2)

	if lossVal2 < lossVal {
		fmt.Println("✓ Loss decreased! Gradients are correct direction.")
	} else {
		fmt.Println("✗ Loss INCREASED! Gradient direction is wrong!")
	}
}
