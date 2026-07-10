package backend

import (
	"fmt"
	"unsafe"

	"github.com/djeday123/goml/core"
)

// DeviceType represents the compute device.
type DeviceType uint8

const (
	CPU DeviceType = iota
	CUDA
	ROCm
	Metal
	Vulkan
)

func (d DeviceType) String() string {
	names := [...]string{"cpu", "cuda", "rocm", "metal", "vulkan"}
	if int(d) < len(names) {
		return names[d]
	}
	return fmt.Sprintf("device(%d)", d)
}

// Device identifies a specific device (type + index).
type Device struct {
	Type  DeviceType
	Index int // GPU index, 0 for CPU
}

var CPU0 = Device{Type: CPU, Index: 0}

func CUDADevice(index int) Device  { return Device{Type: CUDA, Index: index} }
func ROCmDevice(index int) Device  { return Device{Type: ROCm, Index: index} }
func MetalDevice(index int) Device { return Device{Type: Metal, Index: index} }

func (d Device) String() string {
	if d.Type == CPU {
		return "cpu"
	}
	return fmt.Sprintf("%s:%d", d.Type, d.Index)
}

// Storage represents a raw memory buffer on a device.
type Storage interface {
	// Device returns which device this storage lives on.
	Device() Device

	// Ptr returns the raw pointer to the data.
	// For CPU this is a Go pointer, for GPU it's a device pointer.
	Ptr() unsafe.Pointer

	// Bytes returns the underlying byte slice (CPU only, nil for GPU).
	Bytes() []byte

	// ByteLen returns the total size in bytes.
	ByteLen() int

	// Free releases the memory.
	Free()
}

// Backend defines the compute interface that all hardware backends must implement.
// Each operation takes raw storage pointers and shape metadata.
type Backend interface {
	// Device info
	Name() string
	DeviceType() DeviceType

	// Memory management
	Alloc(byteLen int) (Storage, error)
	Free(s Storage)
	Copy(dst, src Storage, byteLen int) error
	ToDevice(dst Device, src Storage) (Storage, error) // cross-device transfer

	// Unary ops
	Neg(dst, src Storage, shape core.Shape, dtype core.DType) error
	Abs(dst, src Storage, shape core.Shape, dtype core.DType) error
	Exp(dst, src Storage, shape core.Shape, dtype core.DType) error
	Log(dst, src Storage, shape core.Shape, dtype core.DType) error
	Sqrt(dst, src Storage, shape core.Shape, dtype core.DType) error
	Tanh(dst, src Storage, shape core.Shape, dtype core.DType) error
	Relu(dst, src Storage, shape core.Shape, dtype core.DType) error
	Gelu(dst, src Storage, shape core.Shape, dtype core.DType) error
	Sigmoid(dst, src Storage, shape core.Shape, dtype core.DType) error
	Silu(dst, src Storage, shape core.Shape, dtype core.DType) error

	// Binary ops (with broadcasting)
	Add(dst, a, b Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error
	Sub(dst, a, b Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error
	Mul(dst, a, b Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error
	Div(dst, a, b Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error

	// Reduction ops
	Sum(dst, src Storage, shape core.Shape, axes []int, keepDim bool, dtype core.DType) error
	Max(dst, src Storage, shape core.Shape, axes []int, keepDim bool, dtype core.DType) error
	Mean(dst, src Storage, shape core.Shape, axes []int, keepDim bool, dtype core.DType) error

	// MatMul: C = A @ B
	// A: [M, K], B: [K, N], C: [M, N]
	// Batched: [..., M, K] @ [..., K, N] = [..., M, N]
	MatMul(dst, a, b Storage, shapeA, shapeB core.Shape, dtype core.DType) error

	// Softmax along given axis
	Softmax(dst, src Storage, shape core.Shape, axis int, dtype core.DType) error

	// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
	LayerNorm(dst, src, gamma, beta Storage, shape core.Shape, normAxis int, eps float64, dtype core.DType) error

	// Embedding lookup
	Embedding(dst, weight, indices Storage, vocabSize, embedDim, seqLen int, dtype core.DType) error

	// RoPE (Rotary Positional Embedding)
	RoPE(dst, src Storage, shape core.Shape, headDim int, base float64, dtype core.DType) error

	// Attention: scaled dot-product attention with causal mask
	ScaledDotProductAttention(
		dst, q, k, v Storage,
		batchSize, numHeads, seqLen, headDim int,
		causal bool, dtype core.DType,
	) error

	// Fill ops
	Fill(dst Storage, shape core.Shape, value float64, dtype core.DType) error
	Arange(dst Storage, start, step float64, n int, dtype core.DType) error

	// Comparison
	Where(dst, cond, a, b Storage, shape core.Shape, dtype core.DType) error
}

// Registry holds all available backends.
var registry = map[DeviceType]Backend{}

// Register adds a backend to the global registry.
func Register(b Backend) {
	registry[b.DeviceType()] = b
}

// Get returns the backend for a device type.
func Get(dt DeviceType) (Backend, error) {
	b, ok := registry[dt]
	if !ok {
		return nil, fmt.Errorf("backend %s not registered", dt)
	}
	return b, nil
}

// GetForDevice returns the backend for a specific device.
func GetForDevice(d Device) (Backend, error) {
	return Get(d.Type)
}
