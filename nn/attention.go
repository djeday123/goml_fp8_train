package nn

import (
	"fmt"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/ops"
	"github.com/djeday123/goml/tensor"
)

// MultiHeadAttention implements multi-head self-attention.
// Architecture follows LLaMA style: separate Q, K, V projections, no output bias.
type MultiHeadAttention struct {
	Wq *Linear // [dim, dim]
	Wk *Linear // [dim, dim]
	Wv *Linear // [dim, dim]
	Wo *Linear // [dim, dim]

	NumHeads int
	HeadDim  int
	Dim      int

	// KV Cache for inference
	CacheK *tensor.Tensor // [batch, numHeads, cachedLen, headDim]
	CacheV *tensor.Tensor // [batch, numHeads, cachedLen, headDim]
}

// NewMultiHeadAttention creates an MHA layer.
func NewMultiHeadAttention(dim, numHeads int, device backend.Device) (*MultiHeadAttention, error) {
	if dim%numHeads != 0 {
		return nil, fmt.Errorf("dim %d not divisible by numHeads %d", dim, numHeads)
	}
	headDim := dim / numHeads

	wq, err := NewLinear(dim, dim, false, device)
	if err != nil {
		return nil, err
	}
	wk, err := NewLinear(dim, dim, false, device)
	if err != nil {
		return nil, err
	}
	wv, err := NewLinear(dim, dim, false, device)
	if err != nil {
		return nil, err
	}
	wo, err := NewLinear(dim, dim, false, device)
	if err != nil {
		return nil, err
	}

	return &MultiHeadAttention{
		Wq: wq, Wk: wk, Wv: wv, Wo: wo,
		NumHeads: numHeads, HeadDim: headDim, Dim: dim,
	}, nil
}

// Forward runs multi-head attention.
// x shape: [batch, seqLen, dim] → [batch, seqLen, dim]
func (mha *MultiHeadAttention) Forward(x *tensor.Tensor, causal bool) (*tensor.Tensor, error) {
	shape := x.Shape()
	batch := shape[0]
	seqLen := shape[1]

	// Project Q, K, V: [batch, seqLen, dim]
	q, err := mha.Wq.Forward(x)
	if err != nil {
		return nil, fmt.Errorf("Wq: %w", err)
	}
	k, err := mha.Wk.Forward(x)
	if err != nil {
		return nil, fmt.Errorf("Wk: %w", err)
	}
	v, err := mha.Wv.Forward(x)
	if err != nil {
		return nil, fmt.Errorf("Wv: %w", err)
	}

	// Reshape to [batch, seqLen, numHeads, headDim]
	q, err = q.View(tensor.Shape{batch, seqLen, mha.NumHeads, mha.HeadDim})
	if err != nil {
		return nil, err
	}
	k, err = k.View(tensor.Shape{batch, seqLen, mha.NumHeads, mha.HeadDim})
	if err != nil {
		return nil, err
	}
	v, err = v.View(tensor.Shape{batch, seqLen, mha.NumHeads, mha.HeadDim})
	if err != nil {
		return nil, err
	}

	// Transpose to [batch, numHeads, seqLen, headDim]
	q, err = q.Transpose([]int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}
	k, err = k.Transpose([]int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}
	v, err = v.Transpose([]int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}

	// Apply RoPE to Q and K
	q, err = applyRoPE(q, mha.HeadDim)
	if err != nil {
		return nil, fmt.Errorf("rope q: %w", err)
	}
	k, err = applyRoPE(k, mha.HeadDim)
	if err != nil {
		return nil, fmt.Errorf("rope k: %w", err)
	}

	// Make Q, K, V contiguous for attention kernel
	q, err = makeContiguous(q)
	if err != nil {
		return nil, err
	}
	k, err = makeContiguous(k)
	if err != nil {
		return nil, err
	}
	v, err = makeContiguous(v)
	if err != nil {
		return nil, err
	}

	// Scaled dot-product attention
	attnOut, err := ops.ScaledDotProductAttention(q, k, v, mha.NumHeads, causal)
	if err != nil {
		return nil, fmt.Errorf("attention: %w", err)
	}

	// Transpose back: [batch, numHeads, seqLen, headDim] → [batch, seqLen, numHeads, headDim]
	attnOut, err = attnOut.Transpose([]int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}

	// Make contiguous and reshape to [batch, seqLen, dim]
	attnOut, err = makeContiguous(attnOut)
	if err != nil {
		return nil, err
	}
	attnOut, err = attnOut.View(tensor.Shape{batch, seqLen, mha.Dim})
	if err != nil {
		return nil, err
	}

	// Output projection
	out, err := mha.Wo.Forward(attnOut)
	if err != nil {
		return nil, fmt.Errorf("Wo: %w", err)
	}

	return out, nil
}

// Parameters returns all trainable parameters.
func (mha *MultiHeadAttention) Parameters() []*tensor.Tensor {
	var params []*tensor.Tensor
	params = append(params, mha.Wq.Parameters()...)
	params = append(params, mha.Wk.Parameters()...)
	params = append(params, mha.Wv.Parameters()...)
	params = append(params, mha.Wo.Parameters()...)
	return params
}

// applyRoPE applies rotary positional embeddings.
func applyRoPE(x *tensor.Tensor, headDim int) (*tensor.Tensor, error) {
	bk, err := backend.GetForDevice(x.Device())
	if err != nil {
		return nil, err
	}

	store, err := bk.Alloc(x.NumElements() * int(x.DType().Size()))
	if err != nil {
		return nil, err
	}

	err = bk.RoPE(store, x.Storage(), x.Shape(), headDim, 10000.0, x.DType())
	if err != nil {
		store.Free()
		return nil, err
	}

	return tensor.NewTensor(store, x.Shape(), x.DType()), nil
}

// makeContiguous copies a non-contiguous tensor into contiguous memory.
func makeContiguous(t *tensor.Tensor) (*tensor.Tensor, error) {
	if t.IsContiguous() {
		return t, nil
	}

	bk, err := backend.GetForDevice(t.Device())
	if err != nil {
		return nil, err
	}

	n := t.NumElements()
	byteLen := n * int(t.DType().Size())
	store, err := bk.Alloc(byteLen)
	if err != nil {
		return nil, err
	}

	// Copy element by element following strides
	shape := t.Shape()
	strides := t.Strides()
	ndim := len(shape)
	indices := make([]int, ndim)

	srcBytes := t.Storage().Bytes()
	dstBytes := store.Bytes()

	elemSize := int(t.DType().Size())

	for i := 0; i < n; i++ {
		// Compute source byte offset from strides
		srcOffset := 0
		for d := 0; d < ndim; d++ {
			srcOffset += indices[d] * strides[d]
		}

		// Copy one element
		dstOff := i * elemSize
		copy(dstBytes[dstOff:dstOff+elemSize], srcBytes[srcOffset:srcOffset+elemSize])

		// Increment indices
		for d := ndim - 1; d >= 0; d-- {
			indices[d]++
			if indices[d] < shape[d] {
				break
			}
			indices[d] = 0
		}
	}

	return tensor.NewTensor(store, shape, t.DType()), nil
}
