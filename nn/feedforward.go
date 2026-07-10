package nn

import (
	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/ops"
	"github.com/djeday123/goml/tensor"
)

// FeedForward implements the FFN block.
// SwiGLU variant (LLaMA style): out = W2(SiLU(W1(x)) * W3(x))
// Standard variant: out = W2(GELU(W1(x)))
type FeedForward struct {
	W1        *Linear // gate projection   [dim, hiddenDim]
	W2        *Linear // down projection   [hiddenDim, dim]
	W3        *Linear // up projection     [dim, hiddenDim] (SwiGLU only)
	UseSwiGLU bool
}

// NewFeedForward creates a feed-forward block.
// hiddenDim is typically 4*dim for standard, or (2/3)*4*dim for SwiGLU.
func NewFeedForward(dim, hiddenDim int, useSwiGLU bool, device backend.Device) (*FeedForward, error) {
	w1, err := NewLinear(dim, hiddenDim, false, device)
	if err != nil {
		return nil, err
	}
	w2, err := NewLinear(hiddenDim, dim, false, device)
	if err != nil {
		return nil, err
	}

	ff := &FeedForward{W1: w1, W2: w2, UseSwiGLU: useSwiGLU}

	if useSwiGLU {
		w3, err := NewLinear(dim, hiddenDim, false, device)
		if err != nil {
			return nil, err
		}
		ff.W3 = w3
	}

	return ff, nil
}

// Forward runs the FFN.
// x: [batch, seqLen, dim] → [batch, seqLen, dim]
func (ff *FeedForward) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	if ff.UseSwiGLU {
		return ff.forwardSwiGLU(x)
	}
	return ff.forwardStandard(x)
}

// forwardSwiGLU: out = W2(SiLU(W1(x)) * W3(x))
func (ff *FeedForward) forwardSwiGLU(x *tensor.Tensor) (*tensor.Tensor, error) {
	// Gate: SiLU(W1(x))
	gate, err := ff.W1.Forward(x)
	if err != nil {
		return nil, err
	}
	gate, err = ops.Silu(gate)
	if err != nil {
		return nil, err
	}

	// Up: W3(x)
	up, err := ff.W3.Forward(x)
	if err != nil {
		return nil, err
	}

	// Element-wise multiply: gate * up
	hidden, err := ops.Mul(gate, up)
	if err != nil {
		return nil, err
	}

	// Down: W2(hidden)
	return ff.W2.Forward(hidden)
}

// forwardStandard: out = W2(GELU(W1(x)))
func (ff *FeedForward) forwardStandard(x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := ff.W1.Forward(x)
	if err != nil {
		return nil, err
	}
	h, err = ops.Gelu(h)
	if err != nil {
		return nil, err
	}
	return ff.W2.Forward(h)
}

// Parameters returns all trainable parameters.
func (ff *FeedForward) Parameters() []*tensor.Tensor {
	var params []*tensor.Tensor
	params = append(params, ff.W1.Parameters()...)
	params = append(params, ff.W2.Parameters()...)
	if ff.W3 != nil {
		params = append(params, ff.W3.Parameters()...)
	}
	return params
}
