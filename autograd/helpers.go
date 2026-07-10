package autograd

import (
	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/tensor"
)

// AddTensors adds two tensors element-wise for gradient accumulation.
func AddTensors(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	bk, err := backend.GetForDevice(a.Device())
	if err != nil {
		return nil, err
	}

	outShape, err := tensor.BroadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}

	n := outShape.NumElements()
	store, err := bk.Alloc(n * int(a.DType().Size()))
	if err != nil {
		return nil, err
	}

	err = bk.Add(store, a.Storage(), b.Storage(), a.Shape(), b.Shape(), outShape, a.DType())
	if err != nil {
		store.Free()
		return nil, err
	}

	return tensor.NewTensor(store, outShape, a.DType()), nil
}
