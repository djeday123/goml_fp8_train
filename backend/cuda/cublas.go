package cuda

// cuBLAS bindings via purego.
//
// Two modes:
//   1. cublasSgemm_v2 (14 args, direct purego) -- FP32 IO + TF32 compute. Always available.
//   2. cublasGemmEx via libcublas_wrapper.so (1 struct arg) -- FP16 IO + TF32 compute. ~2x faster.
//
// The wrapper .so packs 19 args into a struct, exposing a single-pointer entry point
// that purego can call. Built once:
//   gcc -shared -fPIC -o libcublas_wrapper.so cublas_wrapper.c -lcublas -L/usr/local/cuda/lib64 -I/usr/local/cuda/include

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// cuBLAS types
type cublasHandle uintptr
type cublasStatus int32

const (
	CUBLAS_STATUS_SUCCESS          cublasStatus = 0
	CUBLAS_STATUS_NOT_INITIALIZED  cublasStatus = 1
	CUBLAS_STATUS_ALLOC_FAILED     cublasStatus = 3
	CUBLAS_STATUS_INVALID_VALUE    cublasStatus = 7
	CUBLAS_STATUS_ARCH_MISMATCH    cublasStatus = 8
	CUBLAS_STATUS_EXECUTION_FAILED cublasStatus = 13
	CUBLAS_STATUS_INTERNAL_ERROR   cublasStatus = 14
	CUBLAS_STATUS_NOT_SUPPORTED    cublasStatus = 15
)

func (s cublasStatus) Error() string {
	names := map[cublasStatus]string{
		0: "SUCCESS", 1: "NOT_INITIALIZED", 3: "ALLOC_FAILED",
		7: "INVALID_VALUE", 8: "ARCH_MISMATCH", 13: "EXECUTION_FAILED",
		14: "INTERNAL_ERROR", 15: "NOT_SUPPORTED",
	}
	if name, ok := names[s]; ok {
		return fmt.Sprintf("CUBLAS_STATUS_%s", name)
	}
	return fmt.Sprintf("CUBLAS_ERROR(%d)", s)
}

// cuBLAS enums
type cublasOperation int32

const (
	CUBLAS_OP_N cublasOperation = 0
	CUBLAS_OP_T cublasOperation = 1
)

// cublasMath_t for SetMathMode
const (
	CUBLAS_DEFAULT_MATH        = 0
	CUBLAS_TF32_TENSOR_OP_MATH = 3
)

// cudaDataType
const (
	CUDA_R_16F     int32 = 2  // FP16 (half)
	CUDA_R_32F     int32 = 0  // FP32 (float)
	CUDA_R_16BF    int32 = 14 // BF16
	CUDA_R_8F_E4M3 int32 = 28 // FP8 E4M3 (forward pass, higher precision)
	CUDA_R_8F_E5M2 int32 = 29 // FP8 E5M2 (backward pass, wider range)
)

// cublasComputeType
const (
	CUBLAS_COMPUTE_16F           int32 = 64
	CUBLAS_COMPUTE_32F           int32 = 68
	CUBLAS_COMPUTE_32F_FAST_16F  int32 = 74
	CUBLAS_COMPUTE_32F_FAST_TF32 int32 = 77
)

// cublasGemmAlgo
const (
	CUBLAS_GEMM_DEFAULT           int32 = -1
	CUBLAS_GEMM_DEFAULT_TENSOR_OP int32 = 99
)

// -- Function pointers --

var (
	cublasOnce sync.Once
	cublasErr  error

	cublasCreate_v2    func(handle *cublasHandle) cublasStatus
	cublasDestroy_v2   func(handle cublasHandle) cublasStatus
	cublasSetStream_v2 func(handle cublasHandle, stream uintptr) cublasStatus
	cublasSetMathMode  func(handle cublasHandle, mode int32) cublasStatus

	// 14 args -- always available via purego
	cublasSgemm_v2 func(
		handle cublasHandle,
		transa, transb cublasOperation,
		m, n, k int32,
		alpha unsafe.Pointer,
		A uintptr, lda int32,
		B uintptr, ldb int32,
		beta unsafe.Pointer,
		C uintptr, ldc int32,
	) cublasStatus

	// GemmEx wrapper: loaded from libcublas_wrapper.so (optional)
	gemmExWrapper               func(args unsafe.Pointer) cublasStatus
	gemmStridedBatchedExWrapper func(args unsafe.Pointer) cublasStatus
	hasGemmEx                   bool
)

// GemmExArgs matches the C struct layout.
// Field order and padding must match cublas_wrapper.c exactly.
type GemmExArgs struct {
	Handle      cublasHandle
	TransA      cublasOperation
	TransB      cublasOperation
	M           int32
	N           int32
	K           int32
	_pad0       int32
	Alpha       unsafe.Pointer
	A           uintptr
	Atype       int32
	Lda         int32
	B           uintptr
	Btype       int32
	Ldb         int32
	Beta        unsafe.Pointer
	C           uintptr
	Ctype       int32
	Ldc         int32
	ComputeType int32
	Algo        int32
}

// GemmStridedBatchedExArgs matches the C struct.
type GemmStridedBatchedExArgs struct {
	Handle      cublasHandle    // offset 0
	TransA      cublasOperation // offset 8
	TransB      cublasOperation // offset 12
	M           int32           // offset 16
	N           int32           // offset 20
	K           int32           // offset 24
	_pad0       int32           // offset 28 (align Alpha to 8)
	Alpha       unsafe.Pointer  // offset 32
	A           uintptr         // offset 40
	Atype       int32           // offset 48
	Lda         int32           // offset 52
	StrideA     int64           // offset 56 (naturally aligned)
	B           uintptr         // offset 64
	Btype       int32           // offset 72
	Ldb         int32           // offset 76
	StrideB     int64           // offset 80
	Beta        unsafe.Pointer  // offset 88
	C           uintptr         // offset 96
	Ctype       int32           // offset 104
	Ldc         int32           // offset 108
	StrideC     int64           // offset 112
	BatchCount  int32           // offset 120
	ComputeType int32           // offset 124
	Algo        int32           // offset 128
}

func initCuBLAS() error {
	cublasOnce.Do(func() {
		var lib uintptr
		lib, cublasErr = purego.Dlopen("libcublas.so.12", purego.RTLD_LAZY|purego.RTLD_GLOBAL)
		if cublasErr != nil {
			lib, cublasErr = purego.Dlopen("libcublas.so.11", purego.RTLD_LAZY|purego.RTLD_GLOBAL)
			if cublasErr != nil {
				lib, cublasErr = purego.Dlopen("libcublas.so", purego.RTLD_LAZY|purego.RTLD_GLOBAL)
				if cublasErr != nil {
					cublasErr = fmt.Errorf("cannot load libcublas.so: %w", cublasErr)
					return
				}
			}
		}

		purego.RegisterLibFunc(&cublasCreate_v2, lib, "cublasCreate_v2")
		purego.RegisterLibFunc(&cublasDestroy_v2, lib, "cublasDestroy_v2")
		purego.RegisterLibFunc(&cublasSetStream_v2, lib, "cublasSetStream_v2")
		purego.RegisterLibFunc(&cublasSetMathMode, lib, "cublasSetMathMode")
		purego.RegisterLibFunc(&cublasSgemm_v2, lib, "cublasSgemm_v2")

		// Try loading GemmEx wrapper (optional)
		wrapperLib, wErr := purego.Dlopen(resolveLib("libcublas_wrapper.so"), purego.RTLD_LAZY)
		if wErr == nil {
			purego.RegisterLibFunc(&gemmExWrapper, wrapperLib, "gemmex_wrapper")
			purego.RegisterLibFunc(&gemmStridedBatchedExWrapper, wrapperLib, "gemm_strided_batched_ex_wrapper")
			hasGemmEx = true
			fmt.Println("[GoML] cublasGemmEx wrapper loaded -- FP16 mixed precision available")
		}
		initCuBLASLt()
	})
	return cublasErr
}

// -- Handle --

type CuBLASHandle struct {
	handle cublasHandle
}

func NewCuBLASHandle() (*CuBLASHandle, error) {
	if err := initCuBLAS(); err != nil {
		return nil, err
	}
	h := &CuBLASHandle{}
	status := cublasCreate_v2(&h.handle)
	if status != CUBLAS_STATUS_SUCCESS {
		return nil, fmt.Errorf("cublasCreate: %s", status.Error())
	}
	cublasSetMathMode(h.handle, CUBLAS_TF32_TENSOR_OP_MATH)
	return h, nil
}

func (h *CuBLASHandle) SetStream(stream uintptr) error {
	status := cublasSetStream_v2(h.handle, stream)
	if status != CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasSetStream: %s", status.Error())
	}
	return nil
}

func (h *CuBLASHandle) Destroy() {
	if h.handle != 0 {
		cublasDestroy_v2(h.handle)
		h.handle = 0
	}
}

func (h *CuBLASHandle) HasGemmEx() bool {
	return hasGemmEx
}

// ================================================================
// FP32 MatMul (always available)
// ================================================================

// MatMulF32: C = A @ B, all FP32, row-major.
func (h *CuBLASHandle) MatMulF32(dstPtr, aPtr, bPtr uintptr, M, K, N int) error {
	alpha := float32(1.0)
	beta := float32(0.0)
	status := cublasSgemm_v2(
		h.handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		int32(N), int32(M), int32(K),
		unsafe.Pointer(&alpha),
		bPtr, int32(N),
		aPtr, int32(K),
		unsafe.Pointer(&beta),
		dstPtr, int32(N),
	)
	if status != CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasSgemm: %s", status.Error())
	}
	return nil
}

// BatchedMatMulF32: batched C = A @ B via loop.
func (h *CuBLASHandle) BatchedMatMulF32(dstPtr, aPtr, bPtr uintptr, batch, M, K, N int) error {
	alpha := float32(1.0)
	beta := float32(0.0)
	strideA := uintptr(M * K * 4)
	strideB := uintptr(K * N * 4)
	strideC := uintptr(M * N * 4)

	for i := 0; i < batch; i++ {
		status := cublasSgemm_v2(
			h.handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			int32(N), int32(M), int32(K),
			unsafe.Pointer(&alpha),
			bPtr+strideB*uintptr(i), int32(N),
			aPtr+strideA*uintptr(i), int32(K),
			unsafe.Pointer(&beta),
			dstPtr+strideC*uintptr(i), int32(N),
		)
		if status != CUBLAS_STATUS_SUCCESS {
			return fmt.Errorf("cublasSgemm batch %d: %s", i, status.Error())
		}
	}
	return nil
}

// ================================================================
// Mixed Precision MatMul (requires libcublas_wrapper.so)
// ================================================================

// MatMulMixed: C = A @ B with configurable types.
//
// Common configs:
//
//	FP16 in + FP32 out + TF32 compute (standard mixed precision training)
//	BF16 in + FP32 out + TF32 compute
//	FP16 in + FP16 out + FP16 compute (inference)
func (h *CuBLASHandle) MatMulMixed(
	dstPtr, aPtr, bPtr uintptr,
	M, K, N int,
	aType, bType, cType int32,
	computeType int32,
) error {
	if !hasGemmEx {
		return fmt.Errorf("cublasGemmEx not available: build libcublas_wrapper.so first")
	}

	alpha := float32(1.0)
	beta := float32(0.0)

	args := GemmExArgs{
		Handle:      h.handle,
		TransA:      CUBLAS_OP_N,
		TransB:      CUBLAS_OP_N,
		M:           int32(N),
		N:           int32(M),
		K:           int32(K),
		Alpha:       unsafe.Pointer(&alpha),
		A:           bPtr,
		Atype:       bType,
		Lda:         int32(N),
		B:           aPtr,
		Btype:       aType,
		Ldb:         int32(K),
		Beta:        unsafe.Pointer(&beta),
		C:           dstPtr,
		Ctype:       cType,
		Ldc:         int32(N),
		ComputeType: computeType,
		Algo:        CUBLAS_GEMM_DEFAULT_TENSOR_OP,
	}

	status := gemmExWrapper(unsafe.Pointer(&args))
	if status != CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasGemmEx: %s", status.Error())
	}
	return nil
}

// MatMulF16: FP16 in, FP32 out, TF32 compute. Standard mixed precision.
func (h *CuBLASHandle) MatMulF16(dstPtr, aPtr, bPtr uintptr, M, K, N int) error {
	return h.MatMulMixed(dstPtr, aPtr, bPtr, M, K, N,
		CUDA_R_16F, CUDA_R_16F, CUDA_R_32F,
		CUBLAS_COMPUTE_32F_FAST_TF32)
}

// BatchedMatMulMixed: batched with mixed types via single kernel call.
func (h *CuBLASHandle) BatchedMatMulMixed(
	dstPtr, aPtr, bPtr uintptr,
	batch, M, K, N int,
	aType, bType, cType int32,
	computeType int32,
	elemSizeA, elemSizeB, elemSizeC int,
) error {
	if !hasGemmEx {
		return fmt.Errorf("cublasGemmStridedBatchedEx not available")
	}

	alpha := float32(1.0)
	beta := float32(0.0)

	args := GemmStridedBatchedExArgs{
		Handle:      h.handle,
		TransA:      CUBLAS_OP_N,
		TransB:      CUBLAS_OP_N,
		M:           int32(N),
		N:           int32(M),
		K:           int32(K),
		Alpha:       unsafe.Pointer(&alpha),
		A:           bPtr,
		Atype:       bType,
		Lda:         int32(N),
		StrideA:     int64(K * N * elemSizeB),
		B:           aPtr,
		Btype:       aType,
		Ldb:         int32(K),
		StrideB:     int64(M * K * elemSizeA),
		Beta:        unsafe.Pointer(&beta),
		C:           dstPtr,
		Ctype:       cType,
		Ldc:         int32(N),
		StrideC:     int64(M * N * elemSizeC),
		BatchCount:  int32(batch),
		ComputeType: computeType,
		Algo:        CUBLAS_GEMM_DEFAULT_TENSOR_OP,
	}

	status := gemmStridedBatchedExWrapper(unsafe.Pointer(&args))
	if status != CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("cublasGemmStridedBatchedEx: %s", status.Error())
	}
	return nil
}

// BatchedMatMulF16: batched FP16 convenience.
func (h *CuBLASHandle) BatchedMatMulF16(dstPtr, aPtr, bPtr uintptr, batch, M, K, N int) error {
	return h.BatchedMatMulMixed(dstPtr, aPtr, bPtr, batch, M, K, N,
		CUDA_R_16F, CUDA_R_16F, CUDA_R_32F,
		CUBLAS_COMPUTE_32F_FAST_TF32,
		2, 2, 4)
}
