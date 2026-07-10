package core

import "fmt"

// Shape represents the dimensions of a tensor.
type Shape []int

// Strides represents byte offsets between consecutive elements along each dimension.
type Strides []int

// NumElements returns the total number of elements in the shape.
func (s Shape) NumElements() int {
	if len(s) == 0 {
		return 1 // scalar
	}
	n := 1
	for _, d := range s {
		n *= d
	}
	return n
}

// NDim returns the number of dimensions.
func (s Shape) NDim() int {
	return len(s)
}

// Equal checks if two shapes are identical.
func (s Shape) Equal(other Shape) bool {
	if len(s) != len(other) {
		return false
	}
	for i := range s {
		if s[i] != other[i] {
			return false
		}
	}
	return true
}

// Clone returns a copy of the shape.
func (s Shape) Clone() Shape {
	c := make(Shape, len(s))
	copy(c, s)
	return c
}

func (s Shape) String() string {
	return fmt.Sprintf("%v", []int(s))
}

// ContiguousStrides computes row-major (C-order) strides for a given shape and element size.
func ContiguousStrides(shape Shape, elemSize uintptr) Strides {
	ndim := len(shape)
	if ndim == 0 {
		return Strides{}
	}
	strides := make(Strides, ndim)
	strides[ndim-1] = int(elemSize)
	for i := ndim - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}
	return strides
}

// IsContiguous checks if strides represent a contiguous row-major layout.
func IsContiguous(shape Shape, strides Strides, elemSize uintptr) bool {
	if len(shape) != len(strides) {
		return false
	}
	expected := ContiguousStrides(shape, elemSize)
	for i := range strides {
		if strides[i] != expected[i] {
			return false
		}
	}
	return true
}

// BroadcastShapes returns the broadcast-compatible shape of two shapes.
// Follows NumPy broadcasting rules.
func BroadcastShapes(a, b Shape) (Shape, error) {
	maxDim := len(a)
	if len(b) > maxDim {
		maxDim = len(b)
	}

	result := make(Shape, maxDim)
	for i := 0; i < maxDim; i++ {
		da := 1
		db := 1
		if i < len(a) {
			da = a[len(a)-1-i]
		}
		if i < len(b) {
			db = b[len(b)-1-i]
		}

		switch {
		case da == db:
			result[maxDim-1-i] = da
		case da == 1:
			result[maxDim-1-i] = db
		case db == 1:
			result[maxDim-1-i] = da
		default:
			return nil, fmt.Errorf("shapes %v and %v are not broadcast-compatible", a, b)
		}
	}
	return result, nil
}

// FlatIndex converts a multi-dimensional index to a flat byte offset.
func FlatIndex(indices []int, strides Strides) int {
	offset := 0
	for i, idx := range indices {
		offset += idx * strides[i]
	}
	return offset
}

// Permute returns new shape and strides for a transposed view.
func Permute(shape Shape, strides Strides, axes []int) (Shape, Strides, error) {
	if len(axes) != len(shape) {
		return nil, nil, fmt.Errorf("axes length %d != ndim %d", len(axes), len(shape))
	}

	seen := make([]bool, len(axes))
	for _, a := range axes {
		if a < 0 || a >= len(shape) {
			return nil, nil, fmt.Errorf("axis %d out of range for %d dimensions", a, len(shape))
		}
		if seen[a] {
			return nil, nil, fmt.Errorf("duplicate axis %d", a)
		}
		seen[a] = true
	}

	newShape := make(Shape, len(shape))
	newStrides := make(Strides, len(strides))
	for i, a := range axes {
		newShape[i] = shape[a]
		newStrides[i] = strides[a]
	}
	return newShape, newStrides, nil
}
