package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"
	"unsafe"

	"github.com/djeday123/goml/backend"
	_ "github.com/djeday123/goml/backend/cpu"
	_ "github.com/djeday123/goml/backend/cuda"
	"github.com/djeday123/goml/core"
)

func main() {
	runtime.LockOSThread()
	fmt.Println("=== FP32 vs FP16 vs FP8 MatMul Benchmark ===")
	fmt.Println()

	gpu, err := backend.Get(backend.CUDA)
	if err != nil {
		fmt.Printf("CUDA not available: %v\n", err)
		return
	}

	// Force init
	s, _ := gpu.Alloc(4)
	gpu.Free(s)
	fmt.Println()

	sizes := [][3]int{
		{512, 512, 512},
		{1024, 1024, 1024},
		{2048, 2048, 2048},
		{4096, 4096, 4096},
		{8192, 8192, 8192},
	}

	fmt.Printf("%-20s | %-15s | %-15s | %-15s | FP16/32 | FP8/32\n",
		"Size", "FP32 (TFLOPS)", "FP16 (TFLOPS)", "FP8 (TFLOPS)")
	fmt.Println("---------------------+-----------------+-----------------+-----------------+---------+-------")

	for _, sz := range sizes {
		M, K, N := sz[0], sz[1], sz[2]
		flops := float64(2) * float64(M) * float64(K) * float64(N)

		iters := 50
		if M >= 4096 {
			iters = 20
		}

		// --- FP32 benchmark ---
		dataA := randomF32(M * K)
		dataB := randomF32(K * N)

		gpuA32 := hostToGPU(gpu, f32ToBytes(dataA))
		gpuB32 := hostToGPU(gpu, f32ToBytes(dataB))
		gpuC32, _ := gpu.Alloc(M * N * 4)

		gpu.MatMul(gpuC32, gpuA32, gpuB32, core.Shape{M, K}, core.Shape{K, N}, core.Float32)
		deviceSync(gpu)

		start := time.Now()
		for i := 0; i < iters; i++ {
			gpu.MatMul(gpuC32, gpuA32, gpuB32, core.Shape{M, K}, core.Shape{K, N}, core.Float32)
		}
		deviceSync(gpu)
		fp32Time := time.Since(start).Seconds() / float64(iters)
		fp32Tflops := flops / fp32Time / 1e12

		c32Host := gpuToHostF32(gpu, gpuC32, M*N)

		gpu.Free(gpuA32)
		gpu.Free(gpuB32)
		gpu.Free(gpuC32)

		// --- FP16 benchmark ---
		dataAf16 := f32SliceToF16Bytes(dataA)
		dataBf16 := f32SliceToF16Bytes(dataB)

		gpuA16 := hostToGPU(gpu, dataAf16)
		gpuB16 := hostToGPU(gpu, dataBf16)
		gpuC16, _ := gpu.Alloc(M * N * 4)

		aPtr := devPtr(gpuA16)
		bPtr := devPtr(gpuB16)
		cPtr := devPtr(gpuC16)

		err := matMulF16(gpu, cPtr, aPtr, bPtr, M, K, N)
		if err != nil {
			fmt.Printf("  [%4dx%4dx%4d] FP16 not available: %v\n", M, K, N, err)
			gpu.Free(gpuA16)
			gpu.Free(gpuB16)
			gpu.Free(gpuC16)
			continue
		}
		deviceSync(gpu)

		c16Host := gpuToHostF32(gpu, gpuC16, M*N)
		_ = c16Host
		_ = c32Host

		// Warmup
		matMulF16(gpu, cPtr, aPtr, bPtr, M, K, N)
		deviceSync(gpu)

		start = time.Now()
		for i := 0; i < iters; i++ {
			matMulF16(gpu, cPtr, aPtr, bPtr, M, K, N)
		}
		deviceSync(gpu)
		fp16Time := time.Since(start).Seconds() / float64(iters)
		fp16Tflops := flops / fp16Time / 1e12

		speedup16 := fp16Tflops / fp32Tflops

		// --- FP8 E4M3 benchmark ---
		dataAf8 := f32SliceToF8E4M3Bytes(dataA)
		dataBf8 := f32SliceToF8E4M3Bytes(dataB)

		gpuA8 := hostToGPU(gpu, dataAf8)
		gpuB8 := hostToGPU(gpu, dataBf8)
		gpuC8, _ := gpu.Alloc(M * N * 2) // output FP16 = 2 bytes

		a8Ptr := devPtr(gpuA8)
		b8Ptr := devPtr(gpuB8)
		c8Ptr := devPtr(gpuC8)

		var fp8Tflops float64
		var speedup8 float64
		var fp8Err string

		err8 := matMulF8E4M3(gpu, c8Ptr, a8Ptr, b8Ptr, M, K, N)
		if err8 != nil {
			fp8Err = fmt.Sprintf("(%v)", err8)
		} else {
			deviceSync(gpu)

			// Warmup
			matMulF8E4M3(gpu, c8Ptr, a8Ptr, b8Ptr, M, K, N)
			deviceSync(gpu)

			start = time.Now()
			for i := 0; i < iters; i++ {
				matMulF8E4M3(gpu, c8Ptr, a8Ptr, b8Ptr, M, K, N)
			}
			deviceSync(gpu)
			fp8Time := time.Since(start).Seconds() / float64(iters)
			fp8Tflops = flops / fp8Time / 1e12
			speedup8 = fp8Tflops / fp32Tflops
		}

		if fp8Err != "" {
			fmt.Printf("[%4dx%4dx%4d] | %7.1f TFLOPS | %7.1f TFLOPS |   N/A %-10s| %.2fx  | N/A\n",
				M, K, N, fp32Tflops, fp16Tflops, fp8Err, speedup16)
		} else {
			fmt.Printf("[%4dx%4dx%4d] | %7.1f TFLOPS | %7.1f TFLOPS | %7.1f TFLOPS | %.2fx  | %.2fx\n",
				M, K, N, fp32Tflops, fp16Tflops, fp8Tflops, speedup16, speedup8)
		}

		gpu.Free(gpuA16)
		gpu.Free(gpuB16)
		gpu.Free(gpuC16)
		gpu.Free(gpuA8)
		gpu.Free(gpuB8)
		gpu.Free(gpuC8)
	}

	fmt.Println()
	fmt.Println("=== Benchmark Complete ===")
}

// deviceSync calls cudaDeviceSynchronize -- syncs ALL GPU streams.
func deviceSync(b backend.Backend) {
	type devSyncer interface{ DeviceSync() error }
	if ds, ok := b.(devSyncer); ok {
		must(ds.DeviceSync(), "DeviceSync")
	}
}

func matMulF16(b backend.Backend, dstPtr, aPtr, bPtr uintptr, M, K, N int) error {
	type f16Matmuler interface {
		MatMulF16(dstPtr, aPtr, bPtr uintptr, M, K, N int) error
	}
	type cublasAccessor interface {
		CuBLASHandle() interface{}
	}
	if mm, ok := b.(f16Matmuler); ok {
		return mm.MatMulF16(dstPtr, aPtr, bPtr, M, K, N)
	}
	if ca, ok := b.(cublasAccessor); ok {
		h := ca.CuBLASHandle()
		if mm, ok := h.(f16Matmuler); ok {
			return mm.MatMulF16(dstPtr, aPtr, bPtr, M, K, N)
		}
	}
	return fmt.Errorf("FP16 MatMul not available on this backend")
}

func matMulF8E4M3(b backend.Backend, dstPtr, aPtr, bPtr uintptr, M, K, N int) error {
	type f8Matmuler interface {
		MatMulF8E4M3(dstPtr, aPtr, bPtr uintptr, M, K, N int) error
	}
	if mm, ok := b.(f8Matmuler); ok {
		return mm.MatMulF8E4M3(dstPtr, aPtr, bPtr, M, K, N)
	}
	return fmt.Errorf("FP8 MatMul not available")
}

// === FP16 conversion ===

func f32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := (b >> 31) & 1
	exp := int((b>>23)&0xFF) - 127
	frac := b & 0x7FFFFF

	if exp > 15 {
		return uint16(sign<<15 | 0x7C00)
	}
	if exp < -14 {
		return uint16(sign << 15)
	}
	hexp := uint16(exp+15) & 0x1F
	hfrac := uint16(frac >> 13)
	return uint16(sign<<15) | (hexp << 10) | hfrac
}

func f32SliceToF16Bytes(data []float32) []byte {
	out := make([]byte, len(data)*2)
	for i, v := range data {
		h := f32ToF16(v)
		binary.LittleEndian.PutUint16(out[i*2:], h)
	}
	return out
}

// === FP8 E4M3 conversion ===

func f32ToF8E4M3(f float32) byte {
	b := math.Float32bits(f)
	sign := (b >> 31) & 1
	exp := int((b>>23)&0xFF) - 127
	frac := b & 0x7FFFFF

	if exp > 8 {
		return byte(sign<<7 | 0x7E)
	}
	if exp < -6 {
		return byte(sign << 7)
	}
	if (b & 0x7FFFFFFF) == 0 {
		return byte(sign << 7)
	}

	hexp := byte(exp+7) & 0x0F
	hfrac := byte((frac + (1 << 19)) >> 20)
	if hfrac > 7 {
		hfrac = 0
		hexp++
		if hexp > 15 {
			return byte(sign<<7 | 0x7E)
		}
	}
	return byte(sign<<7) | (hexp << 3) | hfrac
}

func f32SliceToF8E4M3Bytes(data []float32) []byte {
	out := make([]byte, len(data))
	for i, v := range data {
		out[i] = f32ToF8E4M3(v)
	}
	return out
}

// === Helpers ===

func must(err error, msg string) {
	if err != nil {
		panic(fmt.Sprintf("%s: %v", msg, err))
	}
}

func randomF32(n int) []float32 {
	data := make([]float32, n)
	for i := range data {
		data[i] = rand.Float32()*2 - 1
	}
	return data
}

func f32ToBytes(data []float32) []byte {
	b := make([]byte, len(data)*4)
	for i, v := range data {
		bits := math.Float32bits(v)
		b[i*4] = byte(bits)
		b[i*4+1] = byte(bits >> 8)
		b[i*4+2] = byte(bits >> 16)
		b[i*4+3] = byte(bits >> 24)
	}
	return b
}

func bytesToF32(b []byte) []float32 {
	n := len(b) / 4
	data := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint32(b[i*4]) | uint32(b[i*4+1])<<8 | uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24
		data[i] = math.Float32frombits(bits)
	}
	return data
}

func hostToGPU(b backend.Backend, data []byte) backend.Storage {
	s, err := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: data})
	must(err, "hostToGPU")
	return s
}

func gpuToHostF32(b backend.Backend, s backend.Storage, n int) []float32 {
	cpuS, err := b.ToDevice(backend.CPU0, s)
	must(err, "gpuToHost")
	return bytesToF32(cpuS.Bytes()[:n*4])
}

func devPtr(s backend.Storage) uintptr {
	type devPtrer interface{ DevicePtr() uintptr }
	if dp, ok := s.(devPtrer); ok {
		return dp.DevicePtr()
	}
	return uintptr(s.Ptr())
}

type cpuStorage struct{ data []byte }

func (s *cpuStorage) Device() backend.Device { return backend.CPU0 }
func (s *cpuStorage) Ptr() unsafe.Pointer {
	if len(s.data) == 0 {
		return nil
	}
	return unsafe.Pointer(&s.data[0])
}
func (s *cpuStorage) Bytes() []byte { return s.data }
func (s *cpuStorage) ByteLen() int  { return len(s.data) }
func (s *cpuStorage) Free()         { s.data = nil }
