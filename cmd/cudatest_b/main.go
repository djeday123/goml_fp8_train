package main

// Phase B CUDA kernel tests + GPU training loop.
//
// Tests 7-14: Correctness of each Phase B kernel vs CPU reference
// Test 15:    Mini training loop on GPU (132K param transformer, Shakespeare)
//
// Usage: go run cmd/cudatest/main.go   (runs Phase A + B tests)
// Or:    go run cmd/cudatest_b/main.go (runs Phase B tests only)

import (
	"fmt"
	"math"
	"math/rand"
	"unsafe"

	"github.com/djeday123/goml/backend"
	_ "github.com/djeday123/goml/backend/cpu"
	_ "github.com/djeday123/goml/backend/cuda"
	"github.com/djeday123/goml/core"
)

func main() {
	fmt.Println("=== GoML CUDA Phase B Tests ===")
	fmt.Println()

	b, err := backend.Get(backend.CUDA)
	if err != nil {
		fmt.Printf("CUDA not available: %v\n", err)
		return
	}

	// Force init
	s, _ := b.Alloc(4)
	b.Free(s)
	fmt.Println()

	// Phase B kernel correctness
	testUnaryOps(b)
	testBinaryOps(b)
	testReductions(b)
	testLayerNorm(b)
	testArange(b)
	testWhere(b)
	testSoftmax(b)
	testGelu(b)

	fmt.Println()
	fmt.Println("=== All Phase B tests passed ===")
}

// ── Test helpers ──

func must(err error, msg string) {
	if err != nil {
		panic(fmt.Sprintf("%s: %v", msg, err))
	}
}

func randomF32(n int) []float32 {
	data := make([]float32, n)
	for i := range data {
		data[i] = rand.Float32()*2 - 1 // [-1, 1]
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

func gpuToHost(b backend.Backend, s backend.Storage, n int) []float32 {
	cpuS, err := b.ToDevice(backend.CPU0, s)
	must(err, "gpuToHost")
	return bytesToF32(cpuS.Bytes()[:n*4])
}

func syncCUDA(b backend.Backend) {
	type syncer interface{ Sync() error }
	if s, ok := b.(syncer); ok {
		must(s.Sync(), "Sync")
	}
}

func maxDiff(a, b []float32) float32 {
	d := float32(0)
	for i := range a {
		diff := float32(math.Abs(float64(a[i] - b[i])))
		if diff > d {
			d = diff
		}
	}
	return d
}

func checkClose(name string, got, want []float32, tol float32) {
	d := maxDiff(got, want)
	if d > tol {
		// Print first few mismatches
		count := 0
		for i := range got {
			diff := float32(math.Abs(float64(got[i] - want[i])))
			if diff > tol {
				fmt.Printf("  [%d] got=%.6f want=%.6f diff=%.6f\n", i, got[i], want[i], diff)
				count++
				if count >= 5 {
					break
				}
			}
		}
		panic(fmt.Sprintf("%s: max diff %.6f > tolerance %.6f", name, d, tol))
	}
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

// ── Unary ops ──

func testUnaryOps(b backend.Backend) {
	fmt.Print("7.  Unary ops (abs,log,sqrt,tanh,relu,sigmoid,silu)... ")

	n := 1024
	src := randomF32(n)
	// Make all positive for log/sqrt
	srcPos := make([]float32, n)
	for i := range src {
		srcPos[i] = float32(math.Abs(float64(src[i]))) + 0.001
	}

	type unaryTest struct {
		name string
		fn   func(dst, src backend.Storage, shape core.Shape, dtype core.DType) error
		ref  func(float32) float32
		src  []float32
		tol  float32
	}

	tests := []unaryTest{
		{"abs", b.Abs, func(x float32) float32 { return float32(math.Abs(float64(x))) }, src, 0.0001},
		{"log", b.Log, func(x float32) float32 { return float32(math.Log(float64(x))) }, srcPos, 0.01},
		{"sqrt", b.Sqrt, func(x float32) float32 { return float32(math.Sqrt(float64(x))) }, srcPos, 0.01},
		{"tanh", b.Tanh, func(x float32) float32 { return float32(math.Tanh(float64(x))) }, src, 0.01},
		{"relu", b.Relu, func(x float32) float32 {
			if x > 0 {
				return x
			}
			return 0
		}, src, 0.0001},
		{"sigmoid", b.Sigmoid, func(x float32) float32 {
			return float32(1.0 / (1.0 + math.Exp(float64(-x))))
		}, src, 0.01},
		{"silu", b.Silu, func(x float32) float32 {
			return x / (1.0 + float32(math.Exp(float64(-x))))
		}, src, 0.01},
	}

	shape := core.Shape{n}
	for _, tt := range tests {
		gpuSrc := hostToGPU(b, f32ToBytes(tt.src))
		gpuDst, err := b.Alloc(n * 4)
		must(err, tt.name+" alloc")
		must(tt.fn(gpuDst, gpuSrc, shape, core.Float32), tt.name)
		syncCUDA(b)

		got := gpuToHost(b, gpuDst, n)
		want := make([]float32, n)
		for i := range tt.src {
			want[i] = tt.ref(tt.src[i])
		}
		checkClose(tt.name, got, want, tt.tol)
		b.Free(gpuSrc)
		b.Free(gpuDst)
	}

	fmt.Println("OK")
}

// ── Binary ops ──

func testBinaryOps(b backend.Backend) {
	fmt.Print("8.  Binary ops (sub, div)... ")

	n := 1024
	a := randomF32(n)
	bData := randomF32(n)
	// Avoid division by zero
	for i := range bData {
		if math.Abs(float64(bData[i])) < 0.01 {
			bData[i] = 0.1
		}
	}

	shape := core.Shape{n}
	gpuA := hostToGPU(b, f32ToBytes(a))
	gpuB := hostToGPU(b, f32ToBytes(bData))

	// Sub
	gpuDst, _ := b.Alloc(n * 4)
	must(b.Sub(gpuDst, gpuA, gpuB, shape, shape, shape, core.Float32), "sub")
	syncCUDA(b)
	got := gpuToHost(b, gpuDst, n)
	want := make([]float32, n)
	for i := range a {
		want[i] = a[i] - bData[i]
	}
	checkClose("sub", got, want, 0.0001)
	b.Free(gpuDst)

	// Div
	gpuDst, _ = b.Alloc(n * 4)
	must(b.Div(gpuDst, gpuA, gpuB, shape, shape, shape, core.Float32), "div")
	syncCUDA(b)
	got = gpuToHost(b, gpuDst, n)
	for i := range a {
		want[i] = a[i] / bData[i]
	}
	checkClose("div", got, want, 0.01)

	b.Free(gpuA)
	b.Free(gpuB)
	b.Free(gpuDst)
	fmt.Println("OK")
}

// ── Reductions ──

func testReductions(b backend.Backend) {
	fmt.Print("9.  Reductions (sum, max, mean)... ")

	// 4 rows of 256 elements
	rows := 4
	cols := 256
	data := randomF32(rows * cols)
	shape := core.Shape{rows, cols}
	gpuSrc := hostToGPU(b, f32ToBytes(data))

	// Sum along axis 1 (last axis)
	gpuDst, _ := b.Alloc(rows * 4)
	must(b.Sum(gpuDst, gpuSrc, shape, []int{1}, false, core.Float32), "sum")
	syncCUDA(b)
	got := gpuToHost(b, gpuDst, rows)
	want := make([]float32, rows)
	for r := 0; r < rows; r++ {
		sum := float32(0)
		for c := 0; c < cols; c++ {
			sum += data[r*cols+c]
		}
		want[r] = sum
	}
	checkClose("sum", got, want, 0.1) // accumulation error expected
	b.Free(gpuDst)

	// Max along axis 1
	gpuDst, _ = b.Alloc(rows * 4)
	must(b.Max(gpuDst, gpuSrc, shape, []int{1}, false, core.Float32), "max")
	syncCUDA(b)
	got = gpuToHost(b, gpuDst, rows)
	for r := 0; r < rows; r++ {
		mx := float32(-math.MaxFloat32)
		for c := 0; c < cols; c++ {
			if data[r*cols+c] > mx {
				mx = data[r*cols+c]
			}
		}
		want[r] = mx
	}
	checkClose("max", got, want, 0.0001)
	b.Free(gpuDst)

	// Mean along axis 1
	gpuDst, _ = b.Alloc(rows * 4)
	must(b.Mean(gpuDst, gpuSrc, shape, []int{1}, false, core.Float32), "mean")
	syncCUDA(b)
	got = gpuToHost(b, gpuDst, rows)
	for r := 0; r < rows; r++ {
		sum := float32(0)
		for c := 0; c < cols; c++ {
			sum += data[r*cols+c]
		}
		want[r] = sum / float32(cols)
	}
	checkClose("mean", got, want, 0.01)

	b.Free(gpuSrc)
	b.Free(gpuDst)
	fmt.Println("OK")
}

// ── LayerNorm ──

func testLayerNorm(b backend.Backend) {
	fmt.Print("10. LayerNorm... ")

	rows := 4
	dim := 128
	n := rows * dim
	eps := float64(1e-5)

	data := randomF32(n)
	gamma := make([]float32, dim)
	beta := make([]float32, dim)
	for i := 0; i < dim; i++ {
		gamma[i] = 1.0 + rand.Float32()*0.1 // near 1
		beta[i] = rand.Float32() * 0.1      // near 0
	}

	gpuSrc := hostToGPU(b, f32ToBytes(data))
	gpuGamma := hostToGPU(b, f32ToBytes(gamma))
	gpuBeta := hostToGPU(b, f32ToBytes(beta))
	gpuDst, _ := b.Alloc(n * 4)

	shape := core.Shape{rows, dim}
	must(b.LayerNorm(gpuDst, gpuSrc, gpuGamma, gpuBeta, shape, 1, eps, core.Float32), "layernorm")
	syncCUDA(b)

	got := gpuToHost(b, gpuDst, n)

	// CPU reference
	want := make([]float32, n)
	for r := 0; r < rows; r++ {
		// mean
		mean := float64(0)
		for c := 0; c < dim; c++ {
			mean += float64(data[r*dim+c])
		}
		mean /= float64(dim)
		// variance
		vr := float64(0)
		for c := 0; c < dim; c++ {
			d := float64(data[r*dim+c]) - mean
			vr += d * d
		}
		vr /= float64(dim)
		invStd := 1.0 / math.Sqrt(vr+eps)
		for c := 0; c < dim; c++ {
			norm := (float64(data[r*dim+c]) - mean) * invStd
			want[r*dim+c] = float32(norm)*gamma[c] + beta[c]
		}
	}
	checkClose("layernorm", got, want, 0.01)

	b.Free(gpuSrc)
	b.Free(gpuGamma)
	b.Free(gpuBeta)
	b.Free(gpuDst)
	fmt.Println("OK")
}

// ── Arange ──

func testArange(b backend.Backend) {
	fmt.Print("11. Arange... ")

	n := 100
	gpuDst, _ := b.Alloc(n * 4)
	must(b.Arange(gpuDst, 2.0, 0.5, n, core.Float32), "arange")
	syncCUDA(b)

	got := gpuToHost(b, gpuDst, n)
	want := make([]float32, n)
	for i := 0; i < n; i++ {
		want[i] = 2.0 + float32(i)*0.5
	}
	checkClose("arange", got, want, 0.001)

	b.Free(gpuDst)
	fmt.Println("OK")
}

// ── Where ──

func testWhere(b backend.Backend) {
	fmt.Print("12. Where... ")

	n := 256
	cond := make([]float32, n)
	a := randomF32(n)
	bData := randomF32(n)
	for i := 0; i < n; i++ {
		if rand.Float32() > 0.5 {
			cond[i] = 1.0
		} else {
			cond[i] = 0.0
		}
	}

	gpuCond := hostToGPU(b, f32ToBytes(cond))
	gpuA := hostToGPU(b, f32ToBytes(a))
	gpuB := hostToGPU(b, f32ToBytes(bData))
	gpuDst, _ := b.Alloc(n * 4)

	shape := core.Shape{n}
	must(b.Where(gpuDst, gpuCond, gpuA, gpuB, shape, core.Float32), "where")
	syncCUDA(b)

	got := gpuToHost(b, gpuDst, n)
	want := make([]float32, n)
	for i := range cond {
		if cond[i] != 0 {
			want[i] = a[i]
		} else {
			want[i] = bData[i]
		}
	}
	checkClose("where", got, want, 0.0001)

	b.Free(gpuCond)
	b.Free(gpuA)
	b.Free(gpuB)
	b.Free(gpuDst)
	fmt.Println("OK")
}

// ── Softmax ──

func testSoftmax(b backend.Backend) {
	fmt.Print("13. Softmax... ")

	rows := 8
	cols := 64
	n := rows * cols
	data := randomF32(n)

	gpuSrc := hostToGPU(b, f32ToBytes(data))
	gpuDst, _ := b.Alloc(n * 4)

	shape := core.Shape{rows, cols}
	must(b.Softmax(gpuDst, gpuSrc, shape, 1, core.Float32), "softmax")
	syncCUDA(b)

	got := gpuToHost(b, gpuDst, n)

	// CPU reference
	want := make([]float32, n)
	for r := 0; r < rows; r++ {
		mx := float32(-math.MaxFloat32)
		for c := 0; c < cols; c++ {
			if data[r*cols+c] > mx {
				mx = data[r*cols+c]
			}
		}
		sum := float32(0)
		for c := 0; c < cols; c++ {
			want[r*cols+c] = float32(math.Exp(float64(data[r*cols+c] - mx)))
			sum += want[r*cols+c]
		}
		for c := 0; c < cols; c++ {
			want[r*cols+c] /= sum
		}
	}
	checkClose("softmax", got, want, 0.01)

	// Verify rows sum to ~1.0
	for r := 0; r < rows; r++ {
		sum := float32(0)
		for c := 0; c < cols; c++ {
			sum += got[r*cols+c]
		}
		if math.Abs(float64(sum-1.0)) > 0.05 {
			panic(fmt.Sprintf("softmax row %d sums to %.4f, expected ~1.0", r, sum))
		}
	}

	b.Free(gpuSrc)
	b.Free(gpuDst)
	fmt.Println("OK")
}

// ── GELU (extra check — complex kernel) ──

func testGelu(b backend.Backend) {
	fmt.Print("14. GELU... ")

	n := 1024
	src := randomF32(n)
	gpuSrc := hostToGPU(b, f32ToBytes(src))
	gpuDst, _ := b.Alloc(n * 4)

	must(b.Gelu(gpuDst, gpuSrc, core.Shape{n}, core.Float32), "gelu")
	syncCUDA(b)

	got := gpuToHost(b, gpuDst, n)
	want := make([]float32, n)
	c := math.Sqrt(2.0 / math.Pi)
	for i, x := range src {
		inner := c * (float64(x) + 0.044715*float64(x)*float64(x)*float64(x))
		want[i] = float32(0.5 * float64(x) * (1.0 + math.Tanh(inner)))
	}
	checkClose("gelu", got, want, 0.01)

	b.Free(gpuSrc)
	b.Free(gpuDst)
	fmt.Println("OK")
}
