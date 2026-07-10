package cuda

import (
	"fmt"
	"sync"

	"github.com/ebitengine/purego"
)

// FP8 GEMM kernel modes
const (
	ModeOriginal   = 0 // v10b baseline
	ModeSinglesync = 1 // paired K-steps, +2-3% on medium matrices
)

// fp8gemm holds the loaded library and function symbols
var fp8gemm struct {
	once     sync.Once
	err      error
	lib      uintptr
	gemm     func(M, N, K int32, A, B, C uintptr, mode int32, stream uintptr) int32
	original func(M, N, K int32, A, B, C uintptr, stream uintptr) int32
	ss       func(M, N, K int32, A, B, C uintptr, stream uintptr) int32
}

// initFP8GEMM loads libfp8gemm.so and resolves symbols
func initFP8GEMM() error {
	fp8gemm.once.Do(func() {
		lib, err := purego.Dlopen(resolveLib("libfp8gemm.so"), purego.RTLD_LAZY|purego.RTLD_GLOBAL)
		if err != nil {
			fp8gemm.err = fmt.Errorf("fp8gemm: dlopen: %w", err)
			return
		}
		fp8gemm.lib = lib

		purego.RegisterLibFunc(&fp8gemm.gemm, lib, "fp8_gemm")
		purego.RegisterLibFunc(&fp8gemm.original, lib, "fp8_gemm_original")
		purego.RegisterLibFunc(&fp8gemm.ss, lib, "fp8_gemm_singlesync")
	})
	return fp8gemm.err
}

// FP8GEMM computes C[M,N] = A[M,K] × B[N,K]^T using FP8 tensor cores.
//
// A: FP8 E4M3 (uint8), device ptr, row-major [M,K]
// B: FP8 E4M3 (uint8), device ptr, row-major [N,K]
// C: FP16     (uint16), device ptr, row-major [M,N]
// mode: ModeOriginal (0) or ModeSinglesync (1)
// stream: CUDA stream device ptr (0 for default)
//
// Returns CUDA error code (0 = success).
func FP8GEMM(M, N, K int, A, B, C uintptr, mode int, stream uintptr) error {
	if err := initFP8GEMM(); err != nil {
		return err
	}

	rc := fp8gemm.gemm(int32(M), int32(N), int32(K), A, B, C, int32(mode), stream)
	if rc != 0 {
		return fmt.Errorf("fp8_gemm: CUDA error %d", rc)
	}
	return nil
}

// FP8GEMMOriginal calls the original v10b kernel directly.
func FP8GEMMOriginal(M, N, K int, A, B, C uintptr, stream uintptr) error {
	if err := initFP8GEMM(); err != nil {
		return err
	}

	rc := fp8gemm.original(int32(M), int32(N), int32(K), A, B, C, stream)
	if rc != 0 {
		return fmt.Errorf("fp8_gemm_original: CUDA error %d", rc)
	}
	return nil
}

// FP8GEMMSinglesync calls the singlesync kernel directly.
// Preferred for inference (medium matrix sizes).
func FP8GEMMSinglesync(M, N, K int, A, B, C uintptr, stream uintptr) error {
	if err := initFP8GEMM(); err != nil {
		return err
	}

	rc := fp8gemm.ss(int32(M), int32(N), int32(K), A, B, C, stream)
	if rc != 0 {
		return fmt.Errorf("fp8_gemm_singlesync: CUDA error %d", rc)
	}
	return nil
}
