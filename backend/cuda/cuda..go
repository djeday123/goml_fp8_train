package cuda

// CUDA Backend for GoML -- implements backend.Backend interface.
//
// Architecture:
//   - MatMul -> cuBLAS GemmEx (Tensor Cores, TF32/FP16)
//   - Elementwise/Softmax/RoPE/AdamW -> custom PTX kernels
//   - Memory -> CUDA Driver API via purego (zero cgo)
//
// Registration: import _ "github.com/djeday123/goml/backend/cuda"
// This triggers init() which calls backend.Register(&Backend{}).
// The backend is initialized lazily on first use.

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/djeday123/goml/backend"
)

// Backend implements backend.Backend for NVIDIA GPUs.
type Backend struct {
	mu          sync.Mutex
	initialized bool

	deviceIdx int
	device    int32
	ctx       uintptr
	stream    uintptr
	info      *DeviceInfo

	cublas *CuBLASHandle
	pool   *Pool

	// PTX module + cached kernel function pointers
	ptxModule  uintptr
	ptxModuleB uintptr // Phase B kernels
	kernels    map[string]uintptr
}

func init() {
	// Only register if CUDA driver is available.
	// This allows the binary to run on machines without NVIDIA GPUs.
	if err := initDriver(); err != nil {
		return // silently skip -- CPU backend will be used
	}
	if r := cuInit(0); r != CUDA_SUCCESS {
		return // no CUDA devices
	}
	backend.Register(&Backend{})
}

func (b *Backend) Name() string                   { return "cuda" }
func (b *Backend) DeviceType() backend.DeviceType { return backend.CUDA }

// ensureInit performs lazy initialization on first use.
func (b *Backend) ensureInit() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.initialized {
		cuCtxSetCurrent(b.ctx)
		return nil
	}

	// Get device
	if r := cuDeviceGet(&b.device, int32(b.deviceIdx)); r != CUDA_SUCCESS {
		return fmt.Errorf("cuDeviceGet(%d): %s", b.deviceIdx, r.Error())
	}

	// Create context
	if r := cuCtxCreate(&b.ctx, 0, b.device); r != CUDA_SUCCESS {
		return fmt.Errorf("cuCtxCreate: %s", r.Error())
	}

	// Create default stream
	if r := cuStreamCreate(&b.stream, CU_STREAM_NON_BLOCKING); r != CUDA_SUCCESS {
		return fmt.Errorf("cuStreamCreate: %s", r.Error())
	}

	// Query device info
	var err error
	b.info, err = QueryDevice(b.deviceIdx)
	if err != nil {
		return fmt.Errorf("QueryDevice: %w", err)
	}

	// Init cuBLAS
	b.cublas, err = NewCuBLASHandle()
	if err != nil {
		return fmt.Errorf("cuBLAS init: %w", err)
	}
	if err := b.cublas.SetStream(b.stream); err != nil {
		return fmt.Errorf("cuBLAS set stream: %w", err)
	}

	// Load PTX kernels (Phase A)
	b.kernels = make(map[string]uintptr)
	ptxBytes := append([]byte(kernelPTX), 0) // null-terminate
	if r := cuModuleLoadData(&b.ptxModule, unsafe.Pointer(&ptxBytes[0])); r != CUDA_SUCCESS {
		return fmt.Errorf("cuModuleLoadData (PTX): %s", r.Error())
	}
	for _, name := range kernelNames {
		nameBytes := append([]byte(name), 0)
		var fn uintptr
		if r := cuModuleGetFunction(&fn, b.ptxModule, &nameBytes[0]); r != CUDA_SUCCESS {
			return fmt.Errorf("cuModuleGetFunction(%s): %s", name, r.Error())
		}
		b.kernels[name] = fn
	}

	// Load PTX kernels (Phase B)
	ptxBytesB := append([]byte(kernelPTX_B), 0)
	if r := cuModuleLoadData(&b.ptxModuleB, unsafe.Pointer(&ptxBytesB[0])); r != CUDA_SUCCESS {
		return fmt.Errorf("cuModuleLoadData (PTX Phase B): %s", r.Error())
	}
	for _, name := range kernelNames_B {
		nameBytes := append([]byte(name), 0)
		var fn uintptr
		if r := cuModuleGetFunction(&fn, b.ptxModuleB, &nameBytes[0]); r != CUDA_SUCCESS {
			return fmt.Errorf("cuModuleGetFunction(%s): %s", name, r.Error())
		}
		b.kernels[name] = fn
	}

	// Init memory pool
	b.pool = NewPool(backend.CUDADevice(b.deviceIdx))

	b.initialized = true
	fmt.Printf("[GoML] CUDA backend initialized: %s (%s, sm_%d%d)\n",
		b.info.Name,
		archName(b.info.ComputeMaj, b.info.ComputeMin),
		b.info.ComputeMaj, b.info.ComputeMin)
	if d := LibsDir(); d != "" {
		fmt.Printf("[GoML] CUDA libs dir: %s\n", d)
	}
	return nil
}

// devPtr extracts the raw device pointer (uintptr) from a Storage.
func devPtr(s backend.Storage) uintptr {
	if cs, ok := s.(*Storage); ok {
		return cs.DevicePtr()
	}
	return uintptr(s.Ptr())
}

// ----------------------------------------------------------------
// Memory management
// ----------------------------------------------------------------

func (b *Backend) Alloc(byteLen int) (backend.Storage, error) {
	if err := b.ensureInit(); err != nil {
		return nil, err
	}
	return b.pool.Get(byteLen)
}

func (b *Backend) Free(s backend.Storage) {
	if cs, ok := s.(*Storage); ok {
		b.pool.Put(cs)
	}
}

func (b *Backend) Copy(dst, src backend.Storage, byteLen int) error {
	if err := b.ensureInit(); err != nil {
		return err
	}
	r := cuMemcpyDtoD(devPtr(dst), devPtr(src), uint64(byteLen))
	if r != CUDA_SUCCESS {
		return fmt.Errorf("cuMemcpyDtoD: %s", r.Error())
	}
	return nil
}

func (b *Backend) ToDevice(dst backend.Device, src backend.Storage) (backend.Storage, error) {
	if err := b.ensureInit(); err != nil {
		return nil, err
	}

	if dst.Type == backend.CUDA {
		// CPU -> GPU
		newStore, err := b.pool.Get(src.ByteLen())
		if err != nil {
			return nil, err
		}
		hostBytes := src.Bytes()
		if hostBytes != nil {
			if err := CopyHtoD(newStore, hostBytes); err != nil {
				return nil, err
			}
		}
		return newStore, nil
	}

	if dst.Type == backend.CPU {
		// GPU -> CPU
		hostBytes := make([]byte, src.ByteLen())
		gpuStore := src.(*Storage)
		if err := CopyDtoH(hostBytes, gpuStore); err != nil {
			return nil, err
		}
		return &cpuBridge{data: hostBytes}, nil
	}

	return nil, fmt.Errorf("ToDevice: unsupported transfer %s -> %s", src.Device(), dst)
}

// cpuBridge is a minimal CPU storage for GPU->CPU transfers.
type cpuBridge struct {
	data []byte
}

func (s *cpuBridge) Device() backend.Device { return backend.CPU0 }
func (s *cpuBridge) Ptr() unsafe.Pointer {
	if len(s.data) == 0 {
		return nil
	}
	return unsafe.Pointer(&s.data[0])
}
func (s *cpuBridge) Bytes() []byte { return s.data }
func (s *cpuBridge) ByteLen() int  { return len(s.data) }
func (s *cpuBridge) Free()         { s.data = nil }

// ----------------------------------------------------------------
// Kernel launch helpers
// ----------------------------------------------------------------

// launch launches a PTX kernel with given grid/block dims and params.
func (b *Backend) launch(name string, gridX, gridY, gridZ, blockX, blockY, blockZ uint32, params []unsafe.Pointer) error {
	fn, ok := b.kernels[name]
	if !ok {
		return fmt.Errorf("kernel not found: %s", name)
	}

	var paramsPtr unsafe.Pointer
	if len(params) > 0 {
		paramsPtr = unsafe.Pointer(&params[0])
	}

	r := cuLaunchKernel(
		fn,
		gridX, gridY, gridZ,
		blockX, blockY, blockZ,
		0,        // shared memory
		b.stream, // stream
		paramsPtr,
		nil, // extra
	)
	if r != CUDA_SUCCESS {
		return fmt.Errorf("cuLaunchKernel(%s): %s", name, r.Error())
	}
	return nil
}

// gridSize1D computes grid dimension for N elements with given block size.
func gridSize1D(n, blockSize int) uint32 {
	return uint32((n + blockSize - 1) / blockSize)
}

// Sync waits for all operations on the default stream to complete.
func (b *Backend) Sync() error {
	r := cuStreamSynchronize(b.stream)
	if r != CUDA_SUCCESS {
		return fmt.Errorf("cuStreamSynchronize: %s", r.Error())
	}
	return nil
}

// Launch exposes kernel launching for custom kernels (e.g. AdamW from training loop).
func (b *Backend) Launch(name string, gridX, gridY, gridZ, blockX, blockY, blockZ uint32, params []unsafe.Pointer) error {
	if err := b.ensureInit(); err != nil {
		return err
	}
	return b.launch(name, gridX, gridY, gridZ, blockX, blockY, blockZ, params)
}

// ----------------------------------------------------------------
// Mixed precision
// ----------------------------------------------------------------

// MatMulF16 performs C = A @ B with FP16 inputs and FP32 output.
func (b *Backend) MatMulF16(dstPtr, aPtr, bPtr uintptr, M, K, N int) error {
	if err := b.ensureInit(); err != nil {
		return err
	}
	return b.cublas.MatMulF16(dstPtr, aPtr, bPtr, M, K, N)
}

// MatMulF8E4M3 performs C = A @ B with FP8 E4M3 inputs and FP32 output.
// Requires SM 8.9+ (Ada/Hopper) and libcublas_wrapper.so.
func (b *Backend) MatMulF8E4M3(dstPtr, aPtr, bPtr uintptr, M, K, N int) error {
	if err := b.ensureInit(); err != nil {
		return err
	}
	return b.cublas.MatMulF8E4M3(dstPtr, aPtr, bPtr, M, K, N)
}

// MatMulF8E5M2 performs C = A @ B with FP8 E5M2 inputs and FP32 output.
func (b *Backend) MatMulF8E5M2(dstPtr, aPtr, bPtr uintptr, M, K, N int) error {
	if err := b.ensureInit(); err != nil {
		return err
	}
	return b.cublas.MatMulF8E5M2(dstPtr, aPtr, bPtr, M, K, N)
}

// HasGemmEx reports whether cublasGemmEx mixed precision is available.
func (b *Backend) HasGemmEx() bool {
	return b.cublas.HasGemmEx()
}

// ----------------------------------------------------------------
// Shutdown
// ----------------------------------------------------------------

// Close releases all CUDA resources.
func (b *Backend) Close() error {
	if !b.initialized {
		return nil
	}
	b.pool.FreeAll()
	b.cublas.Destroy()
	if b.ptxModule != 0 {
		cuModuleUnload(b.ptxModule)
	}
	if b.ptxModuleB != 0 {
		cuModuleUnload(b.ptxModuleB)
	}
	if b.stream != 0 {
		cuStreamDestroy(b.stream)
	}
	if b.ctx != 0 {
		cuCtxDestroy(b.ctx)
	}
	b.initialized = false
	return nil
}

// Info returns the device information (after init).
func (b *Backend) Info() *DeviceInfo {
	return b.info
}
