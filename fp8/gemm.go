// Package fp8gemm provides GEMM (general matrix multiplication) operations
// for FP8 tensors. On systems with CUDA and an NVIDIA Hopper GPU the
// operations use cublasLtMatmul with FP8 data types to achieve near-peak
// tensor-core throughput. On CPU a reference float32 implementation is used.
package fp8

import "unsafe"

// GEMMBackend selects the compute path for FP8 GEMM.
type GEMMBackend uint8

const (
	// BackendCPU uses a plain float32 reference implementation (for testing
	// and development without a GPU).
	BackendCPU GEMMBackend = iota
	// BackendCUDA uses CUDA cublasLtMatmul with FP8 data types.
	BackendCUDA
)

// activeBackend is the backend selected at init time.
var activeBackend = BackendCPU

// Backend returns the current GEMM backend.
func Backend() GEMMBackend { return activeBackend }

// GEMM computes C = alpha * op(A) * op(B) * scaleAB + beta * C
// where A is (M×K) and B is (K×N) in row-major layout.
//
// aData / bData are raw FP8 bytes (E4M3FN); scaleA and scaleB are the
// per-tensor dequantization scale factors (float32). The output C is
// accumulated in float32.
//
// On a CUDA-capable Hopper GPU this calls into the native FP8 path
// (see cuda/fp8_gemm.cu); otherwise it falls back to the CPU reference.
func GEMM(
	aData []uint8, aScale float32, aDtype DType,
	bData []uint8, bScale float32, bDtype DType,
	M, N, K int,
) []float32 {
	_ = unsafe.Sizeof(aData) // keep import
	switch activeBackend {
	case BackendCUDA:
		return gemmCUDA(aData, aScale, aDtype, bData, bScale, bDtype, M, N, K)
	default:
		return gemmCPU(aData, aScale, aDtype, bData, bScale, bDtype, M, N, K)
	}
}

// gemmCPU is the reference float32 implementation. It dequantises both
// operands and performs a naive triple-loop GEMM for correctness testing.
func gemmCPU(
	aData []uint8, aScale float32, aDtype DType,
	bData []uint8, bScale float32, bDtype DType,
	M, N, K int,
) []float32 {
	// Dequantise A (M×K).
	a := dequantSlice(aData, aDtype, aScale)
	// Dequantise B (K×N).
	b := dequantSlice(bData, bDtype, bScale)

	c := make([]float32, M*N)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			sum := float32(0)
			for k := 0; k < K; k++ {
				sum += a[m*K+k] * b[k*N+n]
			}
			c[m*N+n] = sum
		}
	}
	return c
}

func dequantSlice(data []uint8, dtype DType, scale float32) []float32 {
	out := make([]float32, len(data))
	switch dtype {
	case E4M3FN:
		for i, b := range data {
			out[i] = DequantizeE4M3(b) * scale
		}
	case E5M2:
		for i, b := range data {
			out[i] = DequantizeE5M2(b) * scale
		}
	}
	return out
}
