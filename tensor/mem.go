package tensor

import "unsafe"

// copySliceToStorage copies a Go slice into a storage buffer safely.
func copySliceToStorage[T any](data []T, dst []byte) {
	if len(data) == 0 || len(dst) == 0 {
		return
	}
	var zero T
	elemSize := int(unsafe.Sizeof(zero))
	srcLen := len(data) * elemSize
	if srcLen > len(dst) {
		srcLen = len(dst)
	}
	srcBytes := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), srcLen)
	copy(dst, srcBytes)
}

// ptrSlice interprets a storage's memory as a typed slice.
// Uses Bytes() for safe access on CPU.
func ptrSlice[T any](b []byte, n int) []T {
	if n == 0 || len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*T)(unsafe.Pointer(&b[0])), n)
}

// SliceFromPtr interprets raw memory as a Go slice (for GPU backends).
func SliceFromPtr[T any](ptr unsafe.Pointer, n int) []T {
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*T)(ptr), n)
}

// ToFloat32Slice returns the tensor data as []float32.
func (t *Tensor) ToFloat32Slice() []float32 {
	if b := t.storage.Bytes(); b != nil {
		return ptrSlice[float32](b, t.NumElements())
	}
	return SliceFromPtr[float32](t.storage.Ptr(), t.NumElements())
}

// ToFloat64Slice returns the tensor data as []float64.
func (t *Tensor) ToFloat64Slice() []float64 {
	if b := t.storage.Bytes(); b != nil {
		return ptrSlice[float64](b, t.NumElements())
	}
	return SliceFromPtr[float64](t.storage.Ptr(), t.NumElements())
}

// ToInt32Slice returns the tensor data as []int32.
func (t *Tensor) ToInt32Slice() []int32 {
	if b := t.storage.Bytes(); b != nil {
		return ptrSlice[int32](b, t.NumElements())
	}
	return SliceFromPtr[int32](t.storage.Ptr(), t.NumElements())
}

// ToInt64Slice returns the tensor data as []int64.
func (t *Tensor) ToInt64Slice() []int64 {
	if b := t.storage.Bytes(); b != nil {
		return ptrSlice[int64](b, t.NumElements())
	}
	return SliceFromPtr[int64](t.storage.Ptr(), t.NumElements())
}
