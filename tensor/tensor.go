package tensor

import (
	"fmt"

	"github.com/djeday123/goml/backend"
)

// Tensor is the core n-dimensional array.
// It can live on any device and supports autograd.
type Tensor struct {
	storage backend.Storage
	shape   Shape
	strides Strides
	dtype   DType
	offset  int // byte offset into storage (for views)

	// Autograd fields
	requiresGrad bool
	grad         *Tensor
	gradFn       GradFn // function that produced this tensor
	isLeaf       bool   // true if created by user (not by an op)
}

// GradFn represents the backward function for autograd.
type GradFn interface {
	Backward(gradOutput *Tensor) []*Tensor // returns gradients for each input
	Inputs() []*Tensor
	Name() string
}

// ---- Constructors ----

// NewTensor creates a tensor with given storage and metadata.
func NewTensor(storage backend.Storage, shape Shape, dtype DType) *Tensor {
	strides := ContiguousStrides(shape, dtype.Size())
	return &Tensor{
		storage: storage,
		shape:   shape.Clone(),
		strides: strides,
		dtype:   dtype,
		isLeaf:  true,
	}
}

// FromSlice creates a CPU tensor from a Go slice.
func FromSlice[T float32 | float64 | int32 | int64](data []T, shape Shape) (*Tensor, error) {
	n := shape.NumElements()
	if len(data) != n {
		return nil, fmt.Errorf("data length %d != shape elements %d", len(data), n)
	}

	var dtype DType
	switch any(data[0]).(type) {
	case float32:
		dtype = Float32
	case float64:
		dtype = Float64
	case int32:
		dtype = Int32
	case int64:
		dtype = Int64
	}

	b, err := backend.Get(backend.CPU)
	if err != nil {
		return nil, err
	}

	byteLen := n * int(dtype.Size())
	store, err := b.Alloc(byteLen)
	if err != nil {
		return nil, err
	}

	// Copy data into storage
	copySliceToStorage(data, store.Bytes())

	return NewTensor(store, shape, dtype), nil
}

// Zeros creates a zero-filled tensor.
func Zeros(shape Shape, dtype DType, device backend.Device) (*Tensor, error) {
	b, err := backend.GetForDevice(device)
	if err != nil {
		return nil, err
	}

	n := shape.NumElements()
	byteLen := n * int(dtype.Size())
	store, err := b.Alloc(byteLen)
	if err != nil {
		return nil, err
	}

	if err := b.Fill(store, shape, 0, dtype); err != nil {
		store.Free()
		return nil, err
	}

	return NewTensor(store, shape, dtype), nil
}

// Ones creates a tensor filled with ones.
func Ones(shape Shape, dtype DType, device backend.Device) (*Tensor, error) {
	b, err := backend.GetForDevice(device)
	if err != nil {
		return nil, err
	}

	n := shape.NumElements()
	byteLen := n * int(dtype.Size())
	store, err := b.Alloc(byteLen)
	if err != nil {
		return nil, err
	}

	if err := b.Fill(store, shape, 1, dtype); err != nil {
		store.Free()
		return nil, err
	}

	return NewTensor(store, shape, dtype), nil
}

// Arange creates a 1D tensor with values [start, start+step, start+2*step, ...].
func Arange(start, step float64, n int, dtype DType, device backend.Device) (*Tensor, error) {
	b, err := backend.GetForDevice(device)
	if err != nil {
		return nil, err
	}

	byteLen := n * int(dtype.Size())
	store, err := b.Alloc(byteLen)
	if err != nil {
		return nil, err
	}

	if err := b.Arange(store, start, step, n, dtype); err != nil {
		store.Free()
		return nil, err
	}

	return NewTensor(store, Shape{n}, dtype), nil
}

// ---- Accessors ----

func (t *Tensor) Shape() Shape             { return t.shape }
func (t *Tensor) Strides() Strides         { return t.strides }
func (t *Tensor) DType() DType             { return t.dtype }
func (t *Tensor) NDim() int                { return len(t.shape) }
func (t *Tensor) NumElements() int         { return t.shape.NumElements() }
func (t *Tensor) Device() backend.Device   { return t.storage.Device() }
func (t *Tensor) Storage() backend.Storage { return t.storage }
func (t *Tensor) IsLeaf() bool             { return t.isLeaf }

func (t *Tensor) IsContiguous() bool {
	return IsContiguous(t.shape, t.strides, t.dtype.Size())
}

// Contiguous returns a contiguous copy of the tensor.
// If already contiguous, returns the same tensor (no copy).
func (t *Tensor) Contiguous() (*Tensor, error) {
	if t.IsContiguous() {
		return t, nil
	}

	n := t.NumElements()
	elemSize := int(t.dtype.Size())
	byteLen := n * elemSize

	// Allocate new contiguous storage
	newData := make([]float32, n)

	shape := t.shape
	strides := t.strides
	ndim := len(shape)
	indices := make([]int, ndim)
	srcSlice := SliceFromPtr[float32](t.storage.Ptr(), n*2) // oversized for safety

	for i := 0; i < n; i++ {
		// Compute source offset from strides (in bytes → elements)
		srcOffset := 0
		for d := 0; d < ndim; d++ {
			srcOffset += indices[d] * (strides[d] / elemSize)
		}
		newData[i] = srcSlice[srcOffset]

		// Increment indices
		for d := ndim - 1; d >= 0; d-- {
			indices[d]++
			if indices[d] < shape[d] {
				break
			}
			indices[d] = 0
		}
	}

	result, err := FromSlice(newData, shape)
	if err != nil {
		// fallback: alloc manually
		_ = byteLen
		return nil, err
	}
	return result, nil
}

func (t *Tensor) RequiresGrad() bool { return t.requiresGrad }

func (t *Tensor) SetRequiresGrad(v bool) *Tensor {
	t.requiresGrad = v
	return t
}

func (t *Tensor) Grad() *Tensor { return t.grad }

func (t *Tensor) SetGradFn(fn GradFn) {
	t.gradFn = fn
	t.isLeaf = false
}

func (t *Tensor) GradFn() GradFn { return t.gradFn }

func (t *Tensor) SetGrad(grad *Tensor) { t.grad = grad }

// ---- Views ----

// View returns a tensor with a new shape but shared storage.
func (t *Tensor) View(newShape Shape) (*Tensor, error) {
	if !t.IsContiguous() {
		return nil, fmt.Errorf("view requires contiguous tensor")
	}
	if newShape.NumElements() != t.NumElements() {
		return nil, fmt.Errorf("view shape %v has %d elements, need %d",
			newShape, newShape.NumElements(), t.NumElements())
	}
	return &Tensor{
		storage:      t.storage,
		shape:        newShape.Clone(),
		strides:      ContiguousStrides(newShape, t.dtype.Size()),
		dtype:        t.dtype,
		offset:       t.offset,
		requiresGrad: t.requiresGrad,
		isLeaf:       false,
	}, nil
}

// Transpose returns a view with permuted axes.
func (t *Tensor) Transpose(axes []int) (*Tensor, error) {
	newShape, newStrides, err := Permute(t.shape, t.strides, axes)
	if err != nil {
		return nil, err
	}
	return &Tensor{
		storage:      t.storage,
		shape:        newShape,
		strides:      newStrides,
		dtype:        t.dtype,
		offset:       t.offset,
		requiresGrad: t.requiresGrad,
		isLeaf:       false,
	}, nil
}

// T transposes a 2D tensor (shorthand for Transpose([]int{1, 0})).
func (t *Tensor) T() (*Tensor, error) {
	if t.NDim() != 2 {
		return nil, fmt.Errorf("T() requires 2D tensor, got %dD", t.NDim())
	}
	return t.Transpose([]int{1, 0})
}

// Free releases the underlying storage.
func (t *Tensor) Free() {
	if t.storage != nil {
		t.storage.Free()
		t.storage = nil
	}
	if t.grad != nil {
		t.grad.Free()
		t.grad = nil
	}
}

func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor(shape=%v, dtype=%s, device=%s, grad=%v)",
		t.shape, t.dtype, t.Device(), t.requiresGrad)
}
