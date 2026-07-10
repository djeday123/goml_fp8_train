package nn

import (
	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/ops"
	"github.com/djeday123/goml/tensor"
)

// TransformerBlock is one layer of the transformer.
// Pre-norm architecture (LLaMA style):
//
//	x = x + Attention(LayerNorm(x))
//	x = x + FFN(LayerNorm(x))
type TransformerBlock struct {
	AttnNorm *LayerNorm
	Attn     *MultiHeadAttention
	FFNNorm  *LayerNorm
	FFN      *FeedForward
}

// NewTransformerBlock creates one transformer layer.
func NewTransformerBlock(dim, numHeads, ffnHiddenDim int, useSwiGLU bool, device backend.Device) (*TransformerBlock, error) {
	attnNorm, err := NewLayerNorm(dim, 1e-5, device)
	if err != nil {
		return nil, err
	}

	attn, err := NewMultiHeadAttention(dim, numHeads, device)
	if err != nil {
		return nil, err
	}

	ffnNorm, err := NewLayerNorm(dim, 1e-5, device)
	if err != nil {
		return nil, err
	}

	ffn, err := NewFeedForward(dim, ffnHiddenDim, useSwiGLU, device)
	if err != nil {
		return nil, err
	}

	return &TransformerBlock{
		AttnNorm: attnNorm,
		Attn:     attn,
		FFNNorm:  ffnNorm,
		FFN:      ffn,
	}, nil
}

// Forward runs one transformer layer.
// x: [batch, seqLen, dim] → [batch, seqLen, dim]
func (tb *TransformerBlock) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	// Pre-norm attention with residual
	normed, err := tb.AttnNorm.Forward(x)
	if err != nil {
		return nil, err
	}

	attnOut, err := tb.Attn.Forward(normed, true) // causal=true
	if err != nil {
		return nil, err
	}

	// Residual connection: x = x + attn(norm(x))
	x, err = ops.Add(x, attnOut)
	if err != nil {
		return nil, err
	}

	// Pre-norm FFN with residual
	normed, err = tb.FFNNorm.Forward(x)
	if err != nil {
		return nil, err
	}

	ffnOut, err := tb.FFN.Forward(normed)
	if err != nil {
		return nil, err
	}

	// Residual connection: x = x + ffn(norm(x))
	x, err = ops.Add(x, ffnOut)
	if err != nil {
		return nil, err
	}

	return x, nil
}

// Parameters returns all trainable parameters.
func (tb *TransformerBlock) Parameters() []*tensor.Tensor {
	var params []*tensor.Tensor
	params = append(params, tb.AttnNorm.Parameters()...)
	params = append(params, tb.Attn.Parameters()...)
	params = append(params, tb.FFNNorm.Parameters()...)
	params = append(params, tb.FFN.Parameters()...)
	return params
}
