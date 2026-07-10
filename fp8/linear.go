package fp8

import (
	"fmt"
	"math"
	"math/rand"
)

// Linear is an FP8 linear layer (affine transform without bias).
// It stores the weight matrix in FP8 E4M3FN format and applies delayed
// scaling to activations (input) and gradients (backward) to maintain
// numerical stability.
//
// Forward:  Y  = X * W^T          (X in E4M3FN, W in E4M3FN)
// Backward: dX = dY * W           (dY in E5M2)
//           dW = X^T * dY         (X in E4M3FN, dY in E5M2)
type Linear struct {
	// Weight: (OutFeatures × InFeatures) row-major, stored in FP8 E4M3FN.
	Weight *Tensor

	// Master weight (FP32) used by the optimiser.
	MasterWeight []float32

	// Gradient of the weight (FP32 accumulation).
	GradWeight []float32

	InFeatures  int
	OutFeatures int

	// Delayed scalers for the three FP8 GEMM operands.
	scalerX  *DelayedScaler // input activations
	scalerW  *DelayedScaler // weight
	scalerDY *DelayedScaler // upstream gradient

	// Saved from the forward pass for backward.
	savedX *Tensor
}

// NewLinear creates a new FP8 linear layer.
// Weights are Xavier-initialised in FP32 then quantised to FP8.
func NewLinear(inFeatures, outFeatures int) *Linear {
	l := &Linear{
		Weight:      NewTensor([]int{outFeatures, inFeatures}, E4M3FN),
		MasterWeight: make([]float32, outFeatures*inFeatures),
		GradWeight:   make([]float32, outFeatures*inFeatures),
		InFeatures:   inFeatures,
		OutFeatures:  outFeatures,
		scalerX:      NewDelayedScaler(E4M3FN, 16),
		scalerW:      NewDelayedScaler(E4M3FN, 16),
		scalerDY:     NewDelayedScaler(E5M2, 16),
	}

	// Xavier uniform initialisation.
	limit := float32(math.Sqrt(6.0 / float64(inFeatures+outFeatures)))
	for i := range l.MasterWeight {
		l.MasterWeight[i] = (rand.Float32()*2 - 1) * limit
	}
	l.Weight.QuantizeFrom(l.MasterWeight)

	return l
}

// Forward performs the FP8 forward pass Y = X * W^T.
//
// x must be a flat []float32 of length (batchSize * InFeatures).
// Returns flat []float32 of length (batchSize * OutFeatures).
//
// Both X and W are quantised to FP8 E4M3FN before the GEMM; the output is
// accumulated in float32.
func (l *Linear) Forward(x []float32, batchSize int) ([]float32, error) {
	expected := batchSize * l.InFeatures
	if len(x) != expected {
		return nil, fmt.Errorf("fp8 linear: forward input size %d != %d", len(x), expected)
	}

	// Quantise input with delayed scale.
	xFP8 := l.scalerX.Quantize(x)

	// Re-quantise weight with its own delayed scale.
	wFP8 := l.scalerW.Quantize(l.MasterWeight)
	l.Weight = wFP8

	// Save for backward.
	l.savedX = xFP8

	// GEMM: Y (batchSize × OutFeatures) = X (batchSize × InFeatures) * W^T
	// We pass W as (OutFeatures × InFeatures) and compute X * W^T by
	// calling GEMM with the transposed B interpretation. We implement this
	// by swapping A and B so that:
	//   C (OutFeatures × batchSize) = W (OutFeatures × InFeatures) * X^T
	// then transpose the result back.
	//
	// For simplicity in the reference implementation we inline the transpose.
	M := batchSize
	N := l.OutFeatures
	K := l.InFeatures

	// Scale here is the dequantization scale (multiply to go FP8→float32).
	y := GEMM(
		xFP8.Data, xFP8.Scale, E4M3FN,
		wFP8.Data, wFP8.Scale, E4M3FN,
		M, N, K,
	)
	return y, nil
}

// Backward computes dX and accumulates dW from the upstream gradient dY.
//
// dY is a flat []float32 of length (batchSize * OutFeatures).
// Returns dX of length (batchSize * InFeatures).
func (l *Linear) Backward(dY []float32, batchSize int) ([]float32, error) {
	expected := batchSize * l.OutFeatures
	if len(dY) != expected {
		return nil, fmt.Errorf("fp8 linear: backward dY size %d != %d", len(dY), expected)
	}

	// Quantise dY in E5M2 (wider exponent for gradient values).
	dyFP8 := l.scalerDY.Quantize(dY)

	M := batchSize
	N := l.OutFeatures
	K := l.InFeatures

	// dX (M×K) = dY (M×N) * W (N×K) using backward-dX GEMM.
	// We use the CPU GEMM with transposed W (re-use gemmCPU by swapping dims).
	// On GPU this maps to fp8_gemm_backward_dX.
	dX := backwardDX(dyFP8, l.Weight, M, N, K)

	// dW accumulation: dW (N×K) += dY^T (N×M) * X (M×K)
	// i.e., dW[n,k] += sum_m dY[m,n] * X[m,k]
	dW := backwardDW(l.savedX, dyFP8, M, N, K)

	// Accumulate into float32 gradient.
	for i, v := range dW {
		l.GradWeight[i] += v
	}

	// Update delayed scales for next iteration.
	l.scalerX.UpdateScale()
	l.scalerW.UpdateScale()
	l.scalerDY.UpdateScale()

	return dX, nil
}

// ZeroGrad resets the accumulated weight gradient to zero.
func (l *Linear) ZeroGrad() {
	for i := range l.GradWeight {
		l.GradWeight[i] = 0
	}
}

// ---------------------------------------------------------------------------
// Backward GEMM helpers (CPU reference; replaced by CUDA on GPU)
// ---------------------------------------------------------------------------

// backwardDX: dX (M×K) = dY (M×N) * W (N×K)
func backwardDX(dyFP8 *Tensor, wFP8 *Tensor, M, N, K int) []float32 {
	dy := dequantSlice(dyFP8.Data, E5M2, dyFP8.Scale)
	w := dequantSlice(wFP8.Data, E4M3FN, wFP8.Scale)
	dx := make([]float32, M*K)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ {
			sum := float32(0)
			for n := 0; n < N; n++ {
				sum += dy[m*N+n] * w[n*K+k]
			}
			dx[m*K+k] = sum
		}
	}
	return dx
}

// backwardDW: dW (N×K) = dY^T (N×M) * X (M×K)
func backwardDW(xFP8 *Tensor, dyFP8 *Tensor, M, N, K int) []float32 {
	x := dequantSlice(xFP8.Data, E4M3FN, xFP8.Scale)
	dy := dequantSlice(dyFP8.Data, E5M2, dyFP8.Scale)
	dw := make([]float32, N*K)
	for n := 0; n < N; n++ {
		for k := 0; k < K; k++ {
			sum := float32(0)
			for m := 0; m < M; m++ {
				sum += dy[m*N+n] * x[m*K+k]
			}
			dw[n*K+k] = sum
		}
	}
	return dw
}
