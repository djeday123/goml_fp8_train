//go:build cuda

// Package fp8 - CUDA FP8 GEMM backend via cgo.
// Build with: CGO_LDFLAGS="-lcublas -lcublasLt -lcudart" go build -tags cuda .
package fp8

/*
#cgo CFLAGS: -I${SRCDIR}/../cuda
#cgo LDFLAGS: -L${SRCDIR}/../cuda -lfp8gemm -lcublas -lcublasLt -lcudart -lstdc++

#include "fp8_gemm.h"
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"
)

func init() {
	// Switch to CUDA backend when this file is compiled in.
	activeBackend = BackendCUDA
}

// gemmCUDA calls the native CUDA FP8 GEMM implementation via cgo.
// A (M×K) and B (K×N) are stored as row-major FP8 bytes; the output C (M×N)
// is returned in float32.
func gemmCUDA(
	aData []uint8, aScale float32, aDtype DType,
	bData []uint8, bScale float32, bDtype DType,
	M, N, K int,
) []float32 {
	c := make([]float32, M*N)

	dtypeA := C.int(int(aDtype))
	dtypeB := C.int(int(bDtype))

	C.fp8_gemm(
		(*C.uint8_t)(unsafe.Pointer(&aData[0])),
		C.float(aScale),
		dtypeA,
		(*C.uint8_t)(unsafe.Pointer(&bData[0])),
		C.float(bScale),
		dtypeB,
		(*C.float)(unsafe.Pointer(&c[0])),
		C.int(M), C.int(N), C.int(K),
	)
	return c
}
