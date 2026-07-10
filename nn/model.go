package nn

import (
	"fmt"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/tensor"
)

// ModelConfig defines the architecture hyperparameters.
type ModelConfig struct {
	VocabSize    int     // vocabulary size
	Dim          int     // model dimension (embedding size)
	NumLayers    int     // number of transformer blocks
	NumHeads     int     // number of attention heads
	FFNHiddenDim int     // FFN intermediate size
	UseSwiGLU    bool    // use SwiGLU activation in FFN
	MaxSeqLen    int     // maximum sequence length
	NormEps      float64 // layer norm epsilon
}

// SmallConfig returns config for a ~25M parameter model (for testing).
func SmallConfig() ModelConfig {
	return ModelConfig{
		VocabSize:    32000,
		Dim:          256,
		NumLayers:    6,
		NumHeads:     8,
		FFNHiddenDim: 688, // (2/3)*4*256 ≈ 688 for SwiGLU
		UseSwiGLU:    true,
		MaxSeqLen:    512,
		NormEps:      1e-5,
	}
}

// TinyConfig returns config for a ~3M parameter model (for quick testing).
func TinyConfig() ModelConfig {
	return ModelConfig{
		VocabSize:    256, // byte-level
		Dim:          64,
		NumLayers:    2,
		NumHeads:     4,
		FFNHiddenDim: 172, // (2/3)*4*64
		UseSwiGLU:    true,
		MaxSeqLen:    128,
		NormEps:      1e-5,
	}
}

// LLM is the complete language model.
type LLM struct {
	Config   ModelConfig
	TokEmbed *Embedding          // token embedding
	Layers   []*TransformerBlock // transformer layers
	Norm     *LayerNorm          // final layer norm
	Output   *Linear             // lm_head: dim → vocab
}

// NewLLM creates a language model from config.
func NewLLM(cfg ModelConfig, device backend.Device) (*LLM, error) {
	tokEmbed, err := NewEmbedding(cfg.VocabSize, cfg.Dim, device)
	if err != nil {
		return nil, fmt.Errorf("embedding: %w", err)
	}

	layers := make([]*TransformerBlock, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		layer, err := NewTransformerBlock(cfg.Dim, cfg.NumHeads, cfg.FFNHiddenDim, cfg.UseSwiGLU, device)
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
		layers[i] = layer
	}

	norm, err := NewLayerNorm(cfg.Dim, cfg.NormEps, device)
	if err != nil {
		return nil, fmt.Errorf("final norm: %w", err)
	}

	output, err := NewLinear(cfg.Dim, cfg.VocabSize, false, device)
	if err != nil {
		return nil, fmt.Errorf("output: %w", err)
	}

	return &LLM{
		Config:   cfg,
		TokEmbed: tokEmbed,
		Layers:   layers,
		Norm:     norm,
		Output:   output,
	}, nil
}

// Forward runs the full model.
// tokens: [batch, seqLen] (int64) → logits: [batch, seqLen, vocabSize]
func (m *LLM) Forward(tokens *tensor.Tensor) (*tensor.Tensor, error) {
	shape := tokens.Shape()
	batch := shape[0]
	seqLen := shape[1]

	// Token embedding: [batch, seqLen] → [batch, seqLen, dim]
	// Process each batch element separately then combine
	var embeddings []*tensor.Tensor
	for b := 0; b < batch; b++ {
		// Get this batch's tokens
		batchTokens, err := sliceBatch(tokens, b, seqLen)
		if err != nil {
			return nil, fmt.Errorf("batch slice: %w", err)
		}

		emb, err := m.TokEmbed.Forward(batchTokens)
		if err != nil {
			return nil, fmt.Errorf("embedding: %w", err)
		}
		embeddings = append(embeddings, emb)
	}

	// Stack into [batch, seqLen, dim]
	x, err := stackBatch(embeddings, batch, seqLen, m.Config.Dim)
	if err != nil {
		return nil, fmt.Errorf("stack: %w", err)
	}

	// Pass through transformer layers
	for i, layer := range m.Layers {
		x, err = layer.Forward(x)
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
	}

	// Final layer norm
	x, err = m.Norm.Forward(x)
	if err != nil {
		return nil, fmt.Errorf("final norm: %w", err)
	}

	// Project to vocabulary: [batch, seqLen, dim] → [batch, seqLen, vocabSize]
	logits, err := m.Output.Forward(x)
	if err != nil {
		return nil, fmt.Errorf("output: %w", err)
	}

	return logits, nil
}

// CountParameters returns total number of trainable parameters.
func (m *LLM) CountParameters() int {
	total := 0
	for _, p := range m.Parameters() {
		total += p.NumElements()
	}
	return total
}

// Parameters returns all trainable parameters.
func (m *LLM) Parameters() []*tensor.Tensor {
	var params []*tensor.Tensor
	params = append(params, m.TokEmbed.Parameters()...)
	for _, layer := range m.Layers {
		params = append(params, layer.Parameters()...)
	}
	params = append(params, m.Norm.Parameters()...)
	params = append(params, m.Output.Parameters()...)
	return params
}

// sliceBatch extracts one batch element's tokens.
func sliceBatch(tokens *tensor.Tensor, batchIdx, seqLen int) (*tensor.Tensor, error) {
	allData := tokens.ToInt64Slice()
	start := batchIdx * seqLen
	batchData := make([]int64, seqLen)
	copy(batchData, allData[start:start+seqLen])
	return tensor.FromSlice(batchData, tensor.Shape{seqLen})
}

// stackBatch combines batch embeddings into [batch, seqLen, dim].
func stackBatch(embeddings []*tensor.Tensor, batch, seqLen, dim int) (*tensor.Tensor, error) {
	totalSize := batch * seqLen * dim
	data := make([]float32, totalSize)

	for b := 0; b < batch; b++ {
		embData := embeddings[b].ToFloat32Slice()
		copy(data[b*seqLen*dim:(b+1)*seqLen*dim], embData)
	}

	return tensor.FromSlice(data, tensor.Shape{batch, seqLen, dim})
}

// Softmax sampling utilities

// ArgMax returns the index of the maximum value in a 1D tensor.
func ArgMax(t *tensor.Tensor) int {
	data := t.ToFloat32Slice()
	maxIdx := 0
	maxVal := data[0]
	for i, v := range data {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

// TopKSample samples from top-k logits with temperature.
func TopKSample(logits *tensor.Tensor, k int, temperature float32) int {
	data := logits.ToFloat32Slice()
	n := len(data)

	// Apply temperature
	if temperature != 1.0 {
		for i := range data {
			data[i] /= temperature
		}
	}

	// Find top-k indices
	type indexVal struct {
		idx int
		val float32
	}
	items := make([]indexVal, n)
	for i, v := range data {
		items[i] = indexVal{i, v}
	}

	// Partial sort: get top k
	for i := 0; i < k && i < n; i++ {
		maxJ := i
		for j := i + 1; j < n; j++ {
			if items[j].val > items[maxJ].val {
				maxJ = j
			}
		}
		items[i], items[maxJ] = items[maxJ], items[i]
	}

	if k > n {
		k = n
	}

	// Softmax over top-k
	maxVal := items[0].val
	sumExp := float32(0)
	probs := make([]float32, k)
	for i := 0; i < k; i++ {
		probs[i] = float32Exp(items[i].val - maxVal)
		sumExp += probs[i]
	}
	for i := range probs {
		probs[i] /= sumExp
	}

	// Sample from distribution
	r := float32Rand()
	cumSum := float32(0)
	for i := 0; i < k; i++ {
		cumSum += probs[i]
		if r < cumSum {
			return items[i].idx
		}
	}
	return items[k-1].idx
}

func float32Exp(x float32) float32 {
	if x > 88 {
		return 3.4e38
	}
	if x < -88 {
		return 0
	}
	// Fast approximation
	return float32(1) + x + x*x/2 + x*x*x/6
}

func float32Rand() float32 {
	// Simple LCG for sampling (not cryptographic)
	return float32(pseudoRand()) / float32(1<<31)
}

var prngState uint32 = 42

func pseudoRand() uint32 {
	prngState = prngState*1103515245 + 12345
	return (prngState >> 16) & 0x7FFF
}
