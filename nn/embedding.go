package nn

import (
	"math/rand"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/tensor"
)

// Embedding is a lookup table for token embeddings.
type Embedding struct {
	Weight    *tensor.Tensor // [vocabSize, embedDim]
	VocabSize int
	EmbedDim  int
}

// NewEmbedding creates an embedding layer with normal initialization.
func NewEmbedding(vocabSize, embedDim int, device backend.Device) (*Embedding, error) {
	data := make([]float32, vocabSize*embedDim)
	for i := range data {
		data[i] = float32(rand.NormFloat64() * 0.02)
	}

	w, err := tensor.FromSlice(data, tensor.Shape{vocabSize, embedDim})
	if err != nil {
		return nil, err
	}
	w.SetRequiresGrad(true)

	return &Embedding{Weight: w, VocabSize: vocabSize, EmbedDim: embedDim}, nil
}

// Forward looks up embeddings for given token indices.
// indices shape: [seqLen] (int64) → output: [seqLen, embedDim]
func (e *Embedding) Forward(indices *tensor.Tensor) (*tensor.Tensor, error) {
	seqLen := indices.NumElements()

	bk, err := backend.GetForDevice(indices.Device())
	if err != nil {
		return nil, err
	}

	outStore, err := bk.Alloc(seqLen * e.EmbedDim * int(tensor.Float32.Size()))
	if err != nil {
		return nil, err
	}

	err = bk.Embedding(outStore, e.Weight.Storage(), indices.Storage(),
		e.VocabSize, e.EmbedDim, seqLen, tensor.Float32)
	if err != nil {
		return nil, err
	}

	return tensor.NewTensor(outStore, tensor.Shape{seqLen, e.EmbedDim}, tensor.Float32), nil
}

// Parameters returns trainable parameters.
func (e *Embedding) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{e.Weight}
}
