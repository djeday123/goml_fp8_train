package nn

import (
	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/ops"
	"github.com/djeday123/goml/tensor"
)

// LayerNorm implements layer normalization.
type LayerNorm struct {
	Gamma *tensor.Tensor // [normSize] scale
	Beta  *tensor.Tensor // [normSize] shift
	Eps   float64
	Axis  int
}

// NewLayerNorm creates a layer norm with gamma=1, beta=0.
func NewLayerNorm(normSize int, eps float64, device backend.Device) (*LayerNorm, error) {
	gamma, err := tensor.Ones(tensor.Shape{normSize}, tensor.Float32, device)
	if err != nil {
		return nil, err
	}
	gamma.SetRequiresGrad(true)

	beta, err := tensor.Zeros(tensor.Shape{normSize}, tensor.Float32, device)
	if err != nil {
		return nil, err
	}
	beta.SetRequiresGrad(true)

	return &LayerNorm{Gamma: gamma, Beta: beta, Eps: eps, Axis: -1}, nil
}

// Forward applies layer normalization.
// x shape: [..., normSize] → same shape
func (ln *LayerNorm) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	axis := ln.Axis
	if axis < 0 {
		axis = x.NDim() + axis // -1 → last axis
	}
	return ops.LayerNorm(x, ln.Gamma, ln.Beta, axis, ln.Eps)
}

// Parameters returns trainable parameters.
func (ln *LayerNorm) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{ln.Gamma, ln.Beta}
}
