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
	fmt.Println("=== Minimal Gradient Test ===\n")

	vocabSize := 16
	dim := 8

	// Just one linear layer: input [seqLen, dim] → logits [seqLen, vocab]
	linear, _ := nn.NewLinear(dim, vocabSize, true, backend.CPU0)

	// Random input
	seqLen := 4
	xData := make([]float32, seqLen*dim)
	for i := range xData {
		xData[i] = float32(i)*0.1 - 1.0
	}
	x, _ := tensor.FromSlice(xData, tensor.Shape{seqLen, dim})

	targets, _ := tensor.FromSlice([]int64{3, 5, 7, 9}, tensor.Shape{1, seqLen})

	// Forward
	logitsFlat, _ := linear.Forward(x)
	logits, _ := tensor.FromSlice(logitsFlat.ToFloat32Slice(), tensor.Shape{1, seqLen, vocabSize})

	loss, _ := ops.CrossEntropyLoss(logits, targets)
	lossVal := loss.ToFloat32Slice()[0]
	fmt.Printf("Loss: %.6f (expected ~%.4f)\n", lossVal, math.Log(float64(vocabSize)))

	// Backward
	dLogits, _ := ops.CrossEntropyBackward(logits, targets)
	dLogitsFlat, _ := tensor.FromSlice(dLogits.ToFloat32Slice(), tensor.Shape{seqLen, vocabSize})

	linear.Weight.SetGrad(nil)
	if linear.Bias != nil {
		linear.Bias.SetGrad(nil)
	}
	dx, _ := linear.Backward(x, dLogitsFlat)
	_ = dx

	// Numerical gradient check
	fmt.Println("\n--- Weight Gradient Check ---")
	eps := 1e-4
	wData := linear.Weight.ToFloat32Slice()
	wGrad := linear.Weight.Grad().ToFloat32Slice()

	maxErr := float64(0)
	for j := 0; j < 10 && j < len(wData); j++ {
		orig := wData[j]

		wData[j] = orig + float32(eps)
		logP, _ := linear.Forward(x)
		logP3, _ := tensor.FromSlice(logP.ToFloat32Slice(), tensor.Shape{1, seqLen, vocabSize})
		lP, _ := ops.CrossEntropyLoss(logP3, targets)

		wData[j] = orig - float32(eps)
		logM, _ := linear.Forward(x)
		logM3, _ := tensor.FromSlice(logM.ToFloat32Slice(), tensor.Shape{1, seqLen, vocabSize})
		lM, _ := ops.CrossEntropyLoss(logM3, targets)

		wData[j] = orig

		numGrad := (float64(lP.ToFloat32Slice()[0]) - float64(lM.ToFloat32Slice()[0])) / (2 * eps)
		anaGrad := float64(wGrad[j])
		relErr := math.Abs(numGrad-anaGrad) / (math.Abs(numGrad) + math.Abs(anaGrad) + 1e-8)
		if relErr > maxErr {
			maxErr = relErr
		}

		status := "✓"
		if relErr > 0.01 {
			status = "✗"
		}
		fmt.Printf("  w[%d]: ana=%.6f num=%.6f rel=%.6f %s\n", j, anaGrad, numGrad, relErr, status)
	}
	fmt.Printf("Max relative error: %.6f\n", maxErr)

	// Bias check
	if linear.Bias != nil && linear.Bias.Grad() != nil {
		fmt.Println("\n--- Bias Gradient Check ---")
		bData := linear.Bias.ToFloat32Slice()
		bGrad := linear.Bias.Grad().ToFloat32Slice()
		for j := 0; j < 5 && j < len(bData); j++ {
			orig := bData[j]

			bData[j] = orig + float32(eps)
			logP, _ := linear.Forward(x)
			logP3, _ := tensor.FromSlice(logP.ToFloat32Slice(), tensor.Shape{1, seqLen, vocabSize})
			lP, _ := ops.CrossEntropyLoss(logP3, targets)

			bData[j] = orig - float32(eps)
			logM, _ := linear.Forward(x)
			logM3, _ := tensor.FromSlice(logM.ToFloat32Slice(), tensor.Shape{1, seqLen, vocabSize})
			lM, _ := ops.CrossEntropyLoss(logM3, targets)

			bData[j] = orig

			numGrad := (float64(lP.ToFloat32Slice()[0]) - float64(lM.ToFloat32Slice()[0])) / (2 * eps)
			anaGrad := float64(bGrad[j])
			relErr := math.Abs(numGrad-anaGrad) / (math.Abs(numGrad) + math.Abs(anaGrad) + 1e-8)

			status := "✓"
			if relErr > 0.01 {
				status = "✗"
			}
			fmt.Printf("  b[%d]: ana=%.6f num=%.6f rel=%.6f %s\n", j, anaGrad, numGrad, relErr, status)
		}
	}

	// dx check
	fmt.Println("\n--- dx Gradient Check ---")
	dxData := dx.ToFloat32Slice()
	for j := 0; j < 5; j++ {
		orig := xData[j]

		xData[j] = orig + float32(eps)
		xP, _ := tensor.FromSlice(xData, tensor.Shape{seqLen, dim})
		logP, _ := linear.Forward(xP)
		logP3, _ := tensor.FromSlice(logP.ToFloat32Slice(), tensor.Shape{1, seqLen, vocabSize})
		lP, _ := ops.CrossEntropyLoss(logP3, targets)

		xData[j] = orig - float32(eps)
		xM, _ := tensor.FromSlice(xData, tensor.Shape{seqLen, dim})
		logM, _ := linear.Forward(xM)
		logM3, _ := tensor.FromSlice(logM.ToFloat32Slice(), tensor.Shape{1, seqLen, vocabSize})
		lM, _ := ops.CrossEntropyLoss(logM3, targets)

		xData[j] = orig

		numGrad := (float64(lP.ToFloat32Slice()[0]) - float64(lM.ToFloat32Slice()[0])) / (2 * eps)
		anaGrad := float64(dxData[j])
		relErr := math.Abs(numGrad-anaGrad) / (math.Abs(numGrad) + math.Abs(anaGrad) + 1e-8)

		status := "✓"
		if relErr > 0.01 {
			status = "✗"
		}
		fmt.Printf("  dx[%d]: ana=%.6f num=%.6f rel=%.6f %s\n", j, anaGrad, numGrad, relErr, status)
	}

	// Single gradient step test
	fmt.Println("\n--- Single Step Test ---")
	fmt.Printf("Loss before: %.6f\n", lossVal)
	lr := float32(0.1)
	for i := range wData {
		wData[i] -= lr * wGrad[i]
	}
	logits2Flat, _ := linear.Forward(x)
	logits2, _ := tensor.FromSlice(logits2Flat.ToFloat32Slice(), tensor.Shape{1, seqLen, vocabSize})
	loss2, _ := ops.CrossEntropyLoss(logits2, targets)
	lossVal2 := loss2.ToFloat32Slice()[0]
	fmt.Printf("Loss after:  %.6f\n", lossVal2)
	if lossVal2 < lossVal {
		fmt.Println("✓ Loss decreased!")
	} else {
		fmt.Println("✗ Loss INCREASED!")
	}
}
