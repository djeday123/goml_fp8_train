//go:build !cuda

package fp8

// gemmCUDA is a no-op stub compiled when the cuda build tag is absent.
// On GPU systems, replace this file with cuda/fp8_gemm_cuda.go (build tag:
// cuda) which uses cgo to call into the cuBLASLt FP8 GEMM path.
func gemmCUDA(
	aData []uint8, aScale float32, aDtype DType,
	bData []uint8, bScale float32, bDtype DType,
	M, N, K int,
) []float32 {
	// Fall through to CPU reference when CUDA is not available.
	return gemmCPU(aData, aScale, aDtype, bData, bScale, bDtype, M, N, K)
}
