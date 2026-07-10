package main

// CUDA backend smoke test.
// Run on a machine with NVIDIA GPU + driver.
//
// Tests:
//   1. Device detection and info
//   2. Memory alloc/transfer (HtoD, DtoH)
//   3. MatMul via cuBLAS vs CPU (correctness + timing)
//   4. PTX kernel launch (element-wise ops)
//
// Usage: go run cmd/cudatest/main.go

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"
	"unsafe"

	"github.com/djeday123/goml/backend"
	// Import CUDA backend — triggers init() registration
	_ "github.com/djeday123/goml/backend/cuda"
	// Import CPU backend for comparison
	_ "github.com/djeday123/goml/backend/cpu"
	"github.com/djeday123/goml/core"
)

func main() {
	runtime.LockOSThread()
	fmt.Println("=== GoML CUDA Backend Test ===")
	fmt.Println()

	// ── Test 1: Backend registration ──
	fmt.Print("1. Checking backend registration... ")
	cudaBackend, err := backend.Get(backend.CUDA)
	if err != nil {
		fmt.Printf("CUDA not available: %v\n", err)
		fmt.Println("   (This is expected on machines without NVIDIA GPU)")
		fmt.Println("   Running CPU-only tests...")
		testCPUOnly()
		return
	}
	fmt.Printf("OK — %s\n", cudaBackend.Name())

	// ── Test 2: Memory operations ──
	fmt.Print("2. Memory alloc + HtoD + DtoH... ")
	testMemory(cudaBackend)

	// ── Test 3: MatMul correctness ──
	fmt.Print("3. MatMul correctness (cuBLAS vs CPU)... ")
	testMatMulCorrectness(cudaBackend)

	// ── Test 4: MatMul performance ──
	fmt.Println("4. MatMul performance benchmark:")
	testMatMulPerformance(cudaBackend)

	// ── Test 5: PTX kernel (fill) ──
	fmt.Print("5. PTX kernel (Fill)... ")
	testFill(cudaBackend)

	// ── Test 6: Element-wise ops ──
	fmt.Print("6. Element-wise (Add, Mul)... ")
	testElementwise(cudaBackend)

	fmt.Println()
	fmt.Println("=== All tests passed ===")
}

func testMemory(b backend.Backend) {
	// Allocate 1MB on GPU
	size := 1024 * 1024
	gpuMem, err := b.Alloc(size)
	must(err, "Alloc")

	// Create host data
	hostData := make([]byte, size)
	for i := range hostData {
		hostData[i] = byte(i % 256)
	}

	// Transfer to GPU
	gpuDst, err := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: hostData})
	must(err, "ToDevice(CPU→GPU)")

	// Transfer back
	gpuBack, err := b.ToDevice(backend.CPU0, gpuDst)
	must(err, "ToDevice(GPU→CPU)")

	// Verify
	result := gpuBack.Bytes()
	for i := 0; i < 100; i++ {
		if result[i] != hostData[i] {
			panic(fmt.Sprintf("mismatch at %d: got %d, want %d", i, result[i], hostData[i]))
		}
	}

	b.Free(gpuMem)
	b.Free(gpuDst)
	fmt.Println("OK")
}

func testMatMulCorrectness(cudaB backend.Backend) {
	cpuB, _ := backend.Get(backend.CPU)

	M, K, N := 64, 128, 32

	// Random matrices
	a := randomF32(M * K)
	bMat := randomF32(K * N)

	// CPU MatMul
	cpuA := allocF32(cpuB, a)
	cpuBm := allocF32(cpuB, bMat)
	cpuC, _ := cpuB.Alloc(M * N * 4)
	must(cpuB.MatMul(cpuC, cpuA, cpuBm, core.Shape{M, K}, core.Shape{K, N}, core.Float32), "CPU MatMul")

	// GPU MatMul
	gpuA := hostToGPU(cudaB, f32ToBytes(a))
	gpuB := hostToGPU(cudaB, f32ToBytes(bMat))
	gpuC, _ := cudaB.Alloc(M * N * 4)
	must(cudaB.MatMul(gpuC, gpuA, gpuB, core.Shape{M, K}, core.Shape{K, N}, core.Float32), "CUDA MatMul")

	// Sync and compare
	syncCUDA(cudaB)
	gpuResult := gpuToHost(cudaB, gpuC, M*N*4)

	cpuData := readF32(cpuC, M*N)
	gpuData := bytesToF32(gpuResult)

	maxDiff := float32(0)
	for i := 0; i < M*N; i++ {
		diff := float32(math.Abs(float64(cpuData[i] - gpuData[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	if maxDiff > 0.01 {
		panic(fmt.Sprintf("MatMul mismatch: max diff = %f", maxDiff))
	}
	fmt.Printf("OK (max diff: %.6f)\n", maxDiff)
}

func testMatMulPerformance(cudaB backend.Backend) {
	cpuB, _ := backend.Get(backend.CPU)

	sizes := []struct{ M, K, N int }{
		{128, 128, 128},
		{512, 512, 512},
		{1024, 1024, 1024},
		{2048, 2048, 2048},
	}

	for _, sz := range sizes {
		a := randomF32(sz.M * sz.K)
		b := randomF32(sz.K * sz.N)

		// CPU timing
		cpuA := allocF32(cpuB, a)
		cpuBm := allocF32(cpuB, b)
		cpuC, _ := cpuB.Alloc(sz.M * sz.N * 4)
		start := time.Now()
		for i := 0; i < 5; i++ {
			cpuB.MatMul(cpuC, cpuA, cpuBm, core.Shape{sz.M, sz.K}, core.Shape{sz.K, sz.N}, core.Float32)
		}
		cpuTime := time.Since(start) / 5

		// GPU timing
		gpuA := hostToGPU(cudaB, f32ToBytes(a))
		gpuBm := hostToGPU(cudaB, f32ToBytes(b))
		gpuC, _ := cudaB.Alloc(sz.M * sz.N * 4)

		// Warmup
		cudaB.MatMul(gpuC, gpuA, gpuBm, core.Shape{sz.M, sz.K}, core.Shape{sz.K, sz.N}, core.Float32)
		syncCUDA(cudaB)

		start = time.Now()
		for i := 0; i < 100; i++ {
			cudaB.MatMul(gpuC, gpuA, gpuBm, core.Shape{sz.M, sz.K}, core.Shape{sz.K, sz.N}, core.Float32)
		}
		syncCUDA(cudaB)
		gpuTime := time.Since(start) / 100

		// GFLOPS
		flops := 2.0 * float64(sz.M) * float64(sz.K) * float64(sz.N)
		cpuGFLOPS := flops / float64(cpuTime.Nanoseconds())
		gpuGFLOPS := flops / float64(gpuTime.Nanoseconds())
		speedup := float64(cpuTime) / float64(gpuTime)

		fmt.Printf("   [%4dx%4dx%4d] CPU: %8.1f ms (%6.1f GFLOPS) | GPU: %8.3f ms (%8.1f GFLOPS) | %.0fx speedup\n",
			sz.M, sz.K, sz.N,
			float64(cpuTime.Microseconds())/1000, cpuGFLOPS,
			float64(gpuTime.Microseconds())/1000, gpuGFLOPS,
			speedup)
	}
}

func testFill(b backend.Backend) {
	n := 1024
	gpuBuf, _ := b.Alloc(n * 4)
	must(b.Fill(gpuBuf, core.Shape{n}, 3.14, core.Float32), "Fill")
	syncCUDA(b)

	result := bytesToF32(gpuToHost(b, gpuBuf, n*4))
	for i := 0; i < n; i++ {
		diff := math.Abs(float64(result[i] - 3.14))
		if diff > 0.001 {
			panic(fmt.Sprintf("Fill mismatch at %d: got %f", i, result[i]))
		}
	}
	fmt.Println("OK")
}

func testElementwise(b backend.Backend) {
	n := 1024
	a := randomF32(n)
	bVec := randomF32(n)

	gpuA := hostToGPU(b, f32ToBytes(a))
	gpuB := hostToGPU(b, f32ToBytes(bVec))
	gpuC, _ := b.Alloc(n * 4)

	shape := core.Shape{n}
	must(b.Add(gpuC, gpuA, gpuB, shape, shape, shape, core.Float32), "Add")
	syncCUDA(b)

	result := bytesToF32(gpuToHost(b, gpuC, n*4))
	for i := 0; i < n; i++ {
		expected := a[i] + bVec[i]
		diff := math.Abs(float64(result[i] - expected))
		if diff > 0.001 {
			panic(fmt.Sprintf("Add mismatch at %d: got %f, want %f", i, result[i], expected))
		}
	}
	fmt.Println("OK (Add verified)")
}

func testCPUOnly() {
	fmt.Println("CPU backend test — verifying the interface works")
	cpuB, _ := backend.Get(backend.CPU)
	a := randomF32(64 * 64)
	b := randomF32(64 * 64)
	cpuA := allocF32(cpuB, a)
	cpuBm := allocF32(cpuB, b)
	cpuC, _ := cpuB.Alloc(64 * 64 * 4)
	must(cpuB.MatMul(cpuC, cpuA, cpuBm, core.Shape{64, 64}, core.Shape{64, 64}, core.Float32), "CPU MatMul")
	fmt.Println("CPU MatMul OK")
}

// ── Helpers ──

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

func allocF32(b backend.Backend, data []float32) backend.Storage {
	s, _ := b.Alloc(len(data) * 4)
	bytes := s.Bytes()
	for i, v := range data {
		bits := math.Float32bits(v)
		bytes[i*4] = byte(bits)
		bytes[i*4+1] = byte(bits >> 8)
		bytes[i*4+2] = byte(bits >> 16)
		bytes[i*4+3] = byte(bits >> 24)
	}
	return s
}

func readF32(s backend.Storage, n int) []float32 {
	return bytesToF32(s.Bytes()[:n*4])
}

func hostToGPU(b backend.Backend, data []byte) backend.Storage {
	s, err := b.ToDevice(backend.CUDADevice(0), &cpuStorage{data: data})
	must(err, "hostToGPU")
	return s
}

func gpuToHost(b backend.Backend, s backend.Storage, byteLen int) []byte {
	cpuS, err := b.ToDevice(backend.CPU0, s)
	must(err, "gpuToHost")
	return cpuS.Bytes()[:byteLen]
}

func syncCUDA(b backend.Backend) {
	type syncer interface{ Sync() error }
	if s, ok := b.(syncer); ok {
		must(s.Sync(), "Sync")
	}
}

// cpuStorage wraps a byte slice as backend.Storage.
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
