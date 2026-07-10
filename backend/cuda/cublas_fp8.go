package cuda

import (
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego"
)

// ================================================================
// cuBLASLt FP8 (SM 8.9+ Ada/Hopper, CUDA 12+)
// ================================================================

var (
	fp8MatmulWrapper  func(args unsafe.Pointer) int32
	cublasLtCreateFn  func(handle *uintptr) int32
	cublasLtDestroyFn func(handle uintptr) int32
	cudaDeviceSync    func() int32
	hasCublasLt       bool
	ltHandle          uintptr
)

type Fp8MatmulArgs struct {
	Handle uintptr
	M      int32
	N      int32
	K      int32
	_pad   int32
	A      uintptr
	B      uintptr
	C      uintptr
	Alpha  unsafe.Pointer
	Beta   unsafe.Pointer
}

func initCuBLASLt() {
	wrapperLib, err := purego.Dlopen(resolveLib("libcublaslt_wrapper.so"), purego.RTLD_LAZY)
	if err != nil {
		fmt.Println("[GoML] cuBLASLt wrapper not found:", err)
		return
	}

	ltLib, err := purego.Dlopen("libcublasLt.so.12", purego.RTLD_LAZY|purego.RTLD_GLOBAL)
	if err != nil {
		fmt.Println("[GoML] libcublasLt.so.12 not found:", err)
		return
	}

	purego.RegisterLibFunc(&cublasLtCreateFn, ltLib, "cublasLtCreate")
	purego.RegisterLibFunc(&cublasLtDestroyFn, ltLib, "cublasLtDestroy")
	purego.RegisterLibFunc(&fp8MatmulWrapper, wrapperLib, "fp8_matmul_wrapper")
	purego.RegisterLibFunc(&cudaDeviceSync, wrapperLib, "cuda_device_sync")

	status := cublasLtCreateFn(&ltHandle)
	if status != 0 {
		fmt.Printf("[GoML] cublasLtCreate failed: %d\n", status)
		return
	}

	hasCublasLt = true
	fmt.Println("[GoML] cuBLASLt FP8 wrapper loaded -- FP8 E4M3 available")
}

func (h *CuBLASHandle) MatMulF8E4M3(dstPtr, aPtr, bPtr uintptr, M, K, N int) error {
	if !hasCublasLt {
		return fmt.Errorf("cuBLASLt FP8 not available")
	}
	alpha := float32(1.0)
	beta := float32(0.0)
	args := Fp8MatmulArgs{
		Handle: ltHandle,
		M:      int32(M),
		N:      int32(N),
		K:      int32(K),
		A:      aPtr,
		B:      bPtr,
		C:      dstPtr,
		Alpha:  unsafe.Pointer(&alpha),
		Beta:   unsafe.Pointer(&beta),
	}
	status := fp8MatmulWrapper(unsafe.Pointer(&args))
	if status != 0 {
		return fmt.Errorf("cublasLtMatmul FP8: status %d", status)
	}
	return nil
}

func (h *CuBLASHandle) MatMulF8E5M2(dstPtr, aPtr, bPtr uintptr, M, K, N int) error {
	return fmt.Errorf("FP8 E5M2 not yet implemented via cuBLASLt")
}

// DeviceSync calls cudaDeviceSynchronize (syncs ALL streams).
func (b *Backend) DeviceSync() error {
	if cudaDeviceSync != nil {
		st := cudaDeviceSync()
		if st != 0 {
			return fmt.Errorf("cudaDeviceSynchronize: %d", st)
		}
	}
	return nil
}
