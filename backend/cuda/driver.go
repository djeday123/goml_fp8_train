package cuda

// CUDA Driver API bindings via purego.
// No cgo required — loads libcuda.so at runtime via dlopen.
//
// We bind only the ~20 functions needed for our training pipeline:
//   - Device/context management: cuInit, cuDeviceGet, cuCtxCreate, cuCtxSetCurrent
//   - Memory: cuMemAlloc, cuMemFree, cuMemcpyHtoD, cuMemcpyDtoH, cuMemcpyDtoD, cuMemsetD8
//   - Module/kernel: cuModuleLoadData, cuModuleGetFunction, cuLaunchKernel
//   - Streams: cuStreamCreate, cuStreamSynchronize, cuStreamDestroy
//   - Device info: cuDeviceGetName, cuDeviceGetAttribute, cuDeviceTotalMem

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// CUresult error codes (subset we care about).
type CUresult int32

const (
	CUDA_SUCCESS               CUresult = 0
	CUDA_ERROR_INVALID_VALUE   CUresult = 1
	CUDA_ERROR_OUT_OF_MEMORY   CUresult = 2
	CUDA_ERROR_NOT_INITIALIZED CUresult = 3
	CUDA_ERROR_INVALID_CONTEXT CUresult = 201
	CUDA_ERROR_INVALID_HANDLE  CUresult = 400
	CUDA_ERROR_NOT_FOUND       CUresult = 500
	CUDA_ERROR_NOT_READY       CUresult = 600
	CUDA_ERROR_NO_DEVICE       CUresult = 100
	CUDA_ERROR_LAUNCH_FAILED   CUresult = 719
)

func (r CUresult) Error() string {
	if r == CUDA_SUCCESS {
		return "CUDA_SUCCESS"
	}
	names := map[CUresult]string{
		1: "INVALID_VALUE", 2: "OUT_OF_MEMORY", 3: "NOT_INITIALIZED",
		100: "NO_DEVICE", 201: "INVALID_CONTEXT", 400: "INVALID_HANDLE",
		500: "NOT_FOUND", 600: "NOT_READY", 719: "LAUNCH_FAILED",
	}
	if name, ok := names[r]; ok {
		return fmt.Sprintf("CUDA_ERROR_%s (%d)", name, r)
	}
	return fmt.Sprintf("CUDA_ERROR(%d)", r)
}

// CUdevice_attribute codes we need.
const (
	CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK       = 1
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X             = 2
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X              = 5
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
	CU_DEVICE_ATTRIBUTE_WARP_SIZE                   = 10
	CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT        = 16
	CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR    = 75
	CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR    = 76
)

// Stream flag.
const CU_STREAM_NON_BLOCKING = 1

// ──────────────────────────────────────────────────────────
// Driver function pointers — populated by initDriver()
// ──────────────────────────────────────────────────────────

var (
	driverOnce sync.Once
	driverErr  error

	// Init
	cuInit func(flags uint32) CUresult

	// Device
	cuDeviceGet          func(device *int32, ordinal int32) CUresult
	cuDeviceGetName      func(name *byte, len int32, dev int32) CUresult
	cuDeviceGetAttribute func(pi *int32, attrib int32, dev int32) CUresult
	cuDeviceTotalMem     func(bytes *uint64, dev int32) CUresult

	// Context
	cuCtxCreate     func(pctx *uintptr, flags uint32, dev int32) CUresult
	cuCtxSetCurrent func(ctx uintptr) CUresult
	cuCtxDestroy    func(ctx uintptr) CUresult

	// Memory
	cuMemAlloc   func(dptr *uintptr, bytesize uint64) CUresult
	cuMemFree    func(dptr uintptr) CUresult
	cuMemcpyHtoD func(dstDevice uintptr, srcHost unsafe.Pointer, byteCount uint64) CUresult
	cuMemcpyDtoH func(dstHost unsafe.Pointer, srcDevice uintptr, byteCount uint64) CUresult
	cuMemcpyDtoD func(dstDevice uintptr, srcDevice uintptr, byteCount uint64) CUresult
	cuMemsetD8   func(dstDevice uintptr, uc byte, n uint64) CUresult

	// Module / Kernel
	cuModuleLoadData    func(module *uintptr, image unsafe.Pointer) CUresult
	cuModuleGetFunction func(hfunc *uintptr, hmod uintptr, name *byte) CUresult
	cuModuleUnload      func(hmod uintptr) CUresult
	cuLaunchKernel      func(
		f uintptr,
		gridDimX, gridDimY, gridDimZ uint32,
		blockDimX, blockDimY, blockDimZ uint32,
		sharedMemBytes uint32,
		hStream uintptr,
		kernelParams unsafe.Pointer,
		extra unsafe.Pointer,
	) CUresult

	// Stream
	cuStreamCreate      func(phStream *uintptr, flags uint32) CUresult
	cuStreamSynchronize func(hStream uintptr) CUresult
	cuStreamDestroy     func(hStream uintptr) CUresult
)

// initDriver loads libcuda.so and registers all function pointers.
func initDriver() error {
	driverOnce.Do(func() {
		var lib uintptr
		lib, driverErr = purego.Dlopen("libcuda.so.1", purego.RTLD_LAZY|purego.RTLD_GLOBAL)
		if driverErr != nil {
			// Try alternative name
			lib, driverErr = purego.Dlopen("libcuda.so", purego.RTLD_LAZY|purego.RTLD_GLOBAL)
			if driverErr != nil {
				driverErr = fmt.Errorf("cannot load libcuda.so: %w (is NVIDIA driver installed?)", driverErr)
				return
			}
		}

		// Register all functions
		purego.RegisterLibFunc(&cuInit, lib, "cuInit")
		purego.RegisterLibFunc(&cuDeviceGet, lib, "cuDeviceGet")
		purego.RegisterLibFunc(&cuDeviceGetName, lib, "cuDeviceGetName")
		purego.RegisterLibFunc(&cuDeviceGetAttribute, lib, "cuDeviceGetAttribute")
		purego.RegisterLibFunc(&cuDeviceTotalMem, lib, "cuDeviceTotalMem_v2")
		purego.RegisterLibFunc(&cuCtxCreate, lib, "cuCtxCreate_v2")
		purego.RegisterLibFunc(&cuCtxSetCurrent, lib, "cuCtxSetCurrent")
		purego.RegisterLibFunc(&cuCtxDestroy, lib, "cuCtxDestroy_v2")
		purego.RegisterLibFunc(&cuMemAlloc, lib, "cuMemAlloc_v2")
		purego.RegisterLibFunc(&cuMemFree, lib, "cuMemFree_v2")
		purego.RegisterLibFunc(&cuMemcpyHtoD, lib, "cuMemcpyHtoD_v2")
		purego.RegisterLibFunc(&cuMemcpyDtoH, lib, "cuMemcpyDtoH_v2")
		purego.RegisterLibFunc(&cuMemcpyDtoD, lib, "cuMemcpyDtoD_v2")
		purego.RegisterLibFunc(&cuMemsetD8, lib, "cuMemsetD8_v2")
		purego.RegisterLibFunc(&cuModuleLoadData, lib, "cuModuleLoadData")
		purego.RegisterLibFunc(&cuModuleGetFunction, lib, "cuModuleGetFunction")
		purego.RegisterLibFunc(&cuModuleUnload, lib, "cuModuleUnload")
		purego.RegisterLibFunc(&cuLaunchKernel, lib, "cuLaunchKernel")
		purego.RegisterLibFunc(&cuStreamCreate, lib, "cuStreamCreate")
		purego.RegisterLibFunc(&cuStreamSynchronize, lib, "cuStreamSynchronize")
		purego.RegisterLibFunc(&cuStreamDestroy, lib, "cuStreamDestroy_v2")
	})
	return driverErr
}

// ──────────────────────────────────────────────────────────
// High-level wrappers with error checking
// ──────────────────────────────────────────────────────────

func check(r CUresult, op string) error {
	if r != CUDA_SUCCESS {
		return fmt.Errorf("%s: %s", op, r.Error())
	}
	return nil
}

// DeviceInfo holds information about a CUDA device.
type DeviceInfo struct {
	Index      int
	Name       string
	TotalMemMB int
	SMCount    int
	ComputeMaj int
	ComputeMin int
	MaxThreads int
	WarpSize   int
}

// QueryDevice returns information about a CUDA device.
func QueryDevice(index int) (*DeviceInfo, error) {
	if err := initDriver(); err != nil {
		return nil, err
	}
	if r := cuInit(0); r != CUDA_SUCCESS {
		return nil, fmt.Errorf("cuInit: %s", r.Error())
	}

	var dev int32
	if err := check(cuDeviceGet(&dev, int32(index)), "cuDeviceGet"); err != nil {
		return nil, err
	}

	info := &DeviceInfo{Index: index}

	// Name
	nameBuf := make([]byte, 256)
	if err := check(cuDeviceGetName(&nameBuf[0], 256, dev), "cuDeviceGetName"); err != nil {
		return nil, err
	}
	for i, b := range nameBuf {
		if b == 0 {
			info.Name = string(nameBuf[:i])
			break
		}
	}

	// Total memory
	var totalMem uint64
	if err := check(cuDeviceTotalMem(&totalMem, dev), "cuDeviceTotalMem"); err != nil {
		return nil, err
	}
	info.TotalMemMB = int(totalMem / (1024 * 1024))

	// Attributes
	getAttr := func(attr int32) int {
		var val int32
		cuDeviceGetAttribute(&val, attr, dev)
		return int(val)
	}
	info.SMCount = getAttr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
	info.ComputeMaj = getAttr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
	info.ComputeMin = getAttr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
	info.MaxThreads = getAttr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
	info.WarpSize = getAttr(CU_DEVICE_ATTRIBUTE_WARP_SIZE)

	return info, nil
}

func (d *DeviceInfo) String() string {
	return fmt.Sprintf("%s (SM %d.%d, %d SMs, %d MB, %d max threads/block)",
		d.Name, d.ComputeMaj, d.ComputeMin, d.SMCount, d.TotalMemMB, d.MaxThreads)
}
