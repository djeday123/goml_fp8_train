package nn

import (
	"math"
	"math/rand"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/ops"
	"github.com/djeday123/goml/tensor"
)

// Linear implements y = x @ W^T + bias
type Linear struct {
	Weight *tensor.Tensor // [outFeatures, inFeatures]
	Bias   *tensor.Tensor // [outFeatures] or nil
	InF    int
	OutF   int
}

// NewLinear creates a linear layer with Kaiming initialization.
func NewLinear(inFeatures, outFeatures int, bias bool, device backend.Device) (*Linear, error) {
	// Kaiming He init: scale = sqrt(2 / fan_in)
	scale := math.Sqrt(2.0 / float64(inFeatures))

	wData := make([]float32, outFeatures*inFeatures)
	for i := range wData {
		wData[i] = float32(rand.NormFloat64() * scale)
	}

	w, err := tensor.FromSlice(wData, tensor.Shape{outFeatures, inFeatures})
	if err != nil {
		return nil, err
	}
	w.SetRequiresGrad(true)

	l := &Linear{Weight: w, InF: inFeatures, OutF: outFeatures}

	if bias {
		bData := make([]float32, outFeatures)
		b, err := tensor.FromSlice(bData, tensor.Shape{outFeatures})
		if err != nil {
			return nil, err
		}
		b.SetRequiresGrad(true)
		l.Bias = b
	}

	return l, nil
}

// Forward computes y = x @ W^T + bias.
// x shape: [..., inFeatures] → output: [..., outFeatures]
func (l *Linear) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	// W^T: [inFeatures, outFeatures]
	wT, err := l.Weight.T()
	if err != nil {
		return nil, err
	}

	// x @ W^T: [..., inFeatures] @ [inFeatures, outFeatures] = [..., outFeatures]
	out, err := ops.MatMul(x, wT)
	if err != nil {
		return nil, err
	}

	if l.Bias != nil {
		out, err = ops.Add(out, l.Bias)
		if err != nil {
			return nil, err
		}
	}

	return out, nil
}

// Parameters returns all trainable parameters.
func (l *Linear) Parameters() []*tensor.Tensor {
	if l.Bias != nil {
		return []*tensor.Tensor{l.Weight, l.Bias}
	}
	return []*tensor.Tensor{l.Weight}
}
