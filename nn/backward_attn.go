package nn

import (
	"math"

	"github.com/djeday123/goml/tensor"
)

// ---- Attention Backward ----

// AttentionCache stores intermediate values needed for backward pass.
type AttentionCache struct {
	X       *tensor.Tensor // input
	Q, K, V *tensor.Tensor // after projection, shape [batch, heads, seq, headDim]
	Scores  []float32      // attention weights after softmax
}

// ForwardWithCache runs attention and saves intermediates for backward.
func (mha *MultiHeadAttention) ForwardWithCache(x *tensor.Tensor, causal bool) (*tensor.Tensor, *AttentionCache, error) {
	shape := x.Shape()
	batch := shape[0]
	seqLen := shape[1]

	// Project Q, K, V
	q, _ := mha.Wq.Forward(x)
	k, _ := mha.Wk.Forward(x)
	v, _ := mha.Wv.Forward(x)

	// Reshape to [batch*seqLen, numHeads, headDim] then rearrange
	qFlat := q.ToFloat32Slice()
	kFlat := k.ToFloat32Slice()
	vFlat := v.ToFloat32Slice()

	// Rearrange to [batch, heads, seq, headDim]
	qArr := rearrangeBSHD(qFlat, batch, seqLen, mha.NumHeads, mha.HeadDim)
	kArr := rearrangeBSHD(kFlat, batch, seqLen, mha.NumHeads, mha.HeadDim)
	vArr := rearrangeBSHD(vFlat, batch, seqLen, mha.NumHeads, mha.HeadDim)

	// Apply RoPE
	applyRoPEInPlace(qArr, batch, mha.NumHeads, seqLen, mha.HeadDim, 10000.0)
	applyRoPEInPlace(kArr, batch, mha.NumHeads, seqLen, mha.HeadDim, 10000.0)

	// Attention: scores = Q @ K^T / sqrt(d)
	scale := float32(1.0 / math.Sqrt(float64(mha.HeadDim)))
	allScores := make([]float32, batch*mha.NumHeads*seqLen*seqLen)
	outArr := make([]float32, batch*mha.NumHeads*seqLen*mha.HeadDim)

	for b := 0; b < batch; b++ {
		for h := 0; h < mha.NumHeads; h++ {
			bhOff := (b*mha.NumHeads + h) * seqLen * mha.HeadDim
			scOff := (b*mha.NumHeads + h) * seqLen * seqLen

			// Q @ K^T
			for i := 0; i < seqLen; i++ {
				for j := 0; j < seqLen; j++ {
					dot := float32(0)
					for d := 0; d < mha.HeadDim; d++ {
						dot += qArr[bhOff+i*mha.HeadDim+d] * kArr[bhOff+j*mha.HeadDim+d]
					}
					allScores[scOff+i*seqLen+j] = dot * scale
					if causal && j > i {
						allScores[scOff+i*seqLen+j] = -1e9
					}
				}
			}

			// Softmax per row
			for i := 0; i < seqLen; i++ {
				maxVal := float32(-math.MaxFloat32)
				for j := 0; j < seqLen; j++ {
					if allScores[scOff+i*seqLen+j] > maxVal {
						maxVal = allScores[scOff+i*seqLen+j]
					}
				}
				sumExp := float32(0)
				for j := 0; j < seqLen; j++ {
					allScores[scOff+i*seqLen+j] = float32(math.Exp(float64(allScores[scOff+i*seqLen+j] - maxVal)))
					sumExp += allScores[scOff+i*seqLen+j]
				}
				for j := 0; j < seqLen; j++ {
					allScores[scOff+i*seqLen+j] /= sumExp
				}
			}

			// Attn @ V
			for i := 0; i < seqLen; i++ {
				for d := 0; d < mha.HeadDim; d++ {
					sum := float32(0)
					for j := 0; j < seqLen; j++ {
						sum += allScores[scOff+i*seqLen+j] * vArr[bhOff+j*mha.HeadDim+d]
					}
					outArr[bhOff+i*mha.HeadDim+d] = sum
				}
			}
		}
	}

	// Rearrange back to [batch, seq, dim]
	outFlat := rearrangeBHSD(outArr, batch, seqLen, mha.NumHeads, mha.HeadDim)

	outTensor, _ := tensor.FromSlice(outFlat, tensor.Shape{batch, seqLen, mha.Dim})

	// Output projection
	result, _ := mha.Wo.Forward(outTensor)

	// Save cache
	qT, _ := tensor.FromSlice(qArr, tensor.Shape{batch, mha.NumHeads, seqLen, mha.HeadDim})
	kT, _ := tensor.FromSlice(kArr, tensor.Shape{batch, mha.NumHeads, seqLen, mha.HeadDim})
	vT, _ := tensor.FromSlice(vArr, tensor.Shape{batch, mha.NumHeads, seqLen, mha.HeadDim})

	cache := &AttentionCache{X: x, Q: qT, K: kT, V: vT, Scores: allScores}

	return result, cache, nil
}

// Backward computes gradients for attention.
func (mha *MultiHeadAttention) Backward(cache *AttentionCache, dout *tensor.Tensor) (*tensor.Tensor, error) {
	shape := cache.X.Shape()
	batch := shape[0]
	seqLen := shape[1]

	// Backward through Wo
	// Need to recompute attn output for Wo backward
	// First, get the pre-Wo tensor
	qArr := cache.Q.ToFloat32Slice()
	kArr := cache.K.ToFloat32Slice()
	vArr := cache.V.ToFloat32Slice()
	scores := cache.Scores

	// Recompute attn output
	outArr := make([]float32, batch*mha.NumHeads*seqLen*mha.HeadDim)
	for b := 0; b < batch; b++ {
		for h := 0; h < mha.NumHeads; h++ {
			bhOff := (b*mha.NumHeads + h) * seqLen * mha.HeadDim
			scOff := (b*mha.NumHeads + h) * seqLen * seqLen
			for i := 0; i < seqLen; i++ {
				for d := 0; d < mha.HeadDim; d++ {
					sum := float32(0)
					for j := 0; j < seqLen; j++ {
						sum += scores[scOff+i*seqLen+j] * vArr[bhOff+j*mha.HeadDim+d]
					}
					outArr[bhOff+i*mha.HeadDim+d] = sum
				}
			}
		}
	}
	outFlat := rearrangeBHSD(outArr, batch, seqLen, mha.NumHeads, mha.HeadDim)
	attnOutTensor, _ := tensor.FromSlice(outFlat, tensor.Shape{batch, seqLen, mha.Dim})

	// dAttnOut from Wo backward
	dAttnOut, _ := mha.Wo.Backward(attnOutTensor, dout)
	dAttnOutData := dAttnOut.ToFloat32Slice()

	// Rearrange dAttnOut to [batch, heads, seq, headDim]
	dOutArr := rearrangeBSHD(dAttnOutData, batch, seqLen, mha.NumHeads, mha.HeadDim)

	// Backward through attention: dScores, dV, dQ, dK
	scale := float32(1.0 / math.Sqrt(float64(mha.HeadDim)))
	dQArr := make([]float32, len(qArr))
	dKArr := make([]float32, len(kArr))
	dVArr := make([]float32, len(vArr))

	for b := 0; b < batch; b++ {
		for h := 0; h < mha.NumHeads; h++ {
			bhOff := (b*mha.NumHeads + h) * seqLen * mha.HeadDim
			scOff := (b*mha.NumHeads + h) * seqLen * seqLen

			// dV = scores^T @ dOut
			for j := 0; j < seqLen; j++ {
				for d := 0; d < mha.HeadDim; d++ {
					sum := float32(0)
					for i := 0; i < seqLen; i++ {
						sum += scores[scOff+i*seqLen+j] * dOutArr[bhOff+i*mha.HeadDim+d]
					}
					dVArr[bhOff+j*mha.HeadDim+d] = sum
				}
			}

			// dScores = dOut @ V^T
			dScores := make([]float32, seqLen*seqLen)
			for i := 0; i < seqLen; i++ {
				for j := 0; j < seqLen; j++ {
					sum := float32(0)
					for d := 0; d < mha.HeadDim; d++ {
						sum += dOutArr[bhOff+i*mha.HeadDim+d] * vArr[bhOff+j*mha.HeadDim+d]
					}
					dScores[i*seqLen+j] = sum
				}
			}

			// Backward through softmax: dPre = scores * (dScores - sum(dScores * scores))
			for i := 0; i < seqLen; i++ {
				dot := float32(0)
				for j := 0; j < seqLen; j++ {
					dot += dScores[i*seqLen+j] * scores[scOff+i*seqLen+j]
				}
				for j := 0; j < seqLen; j++ {
					dScores[i*seqLen+j] = scores[scOff+i*seqLen+j] * (dScores[i*seqLen+j] - dot) * scale
				}
			}

			// dQ = dPre @ K
			for i := 0; i < seqLen; i++ {
				for d := 0; d < mha.HeadDim; d++ {
					sum := float32(0)
					for j := 0; j < seqLen; j++ {
						sum += dScores[i*seqLen+j] * kArr[bhOff+j*mha.HeadDim+d]
					}
					dQArr[bhOff+i*mha.HeadDim+d] = sum
				}
			}

			// dK = dPre^T @ Q
			for j := 0; j < seqLen; j++ {
				for d := 0; d < mha.HeadDim; d++ {
					sum := float32(0)
					for i := 0; i < seqLen; i++ {
						sum += dScores[i*seqLen+j] * qArr[bhOff+i*mha.HeadDim+d]
					}
					dKArr[bhOff+j*mha.HeadDim+d] = sum
				}
			}
		}
	}

	// RoPE backward (reverse rotation — same as forward but with -angle)
	ropeBackwardInPlace(dQArr, batch, mha.NumHeads, seqLen, mha.HeadDim, 10000.0)
	ropeBackwardInPlace(dKArr, batch, mha.NumHeads, seqLen, mha.HeadDim, 10000.0)

	// Rearrange back to [batch, seq, dim]
	dQFlat := rearrangeBHSD(dQArr, batch, seqLen, mha.NumHeads, mha.HeadDim)
	dKFlat := rearrangeBHSD(dKArr, batch, seqLen, mha.NumHeads, mha.HeadDim)
	dVFlat := rearrangeBHSD(dVArr, batch, seqLen, mha.NumHeads, mha.HeadDim)

	dQ, _ := tensor.FromSlice(dQFlat, tensor.Shape{batch, seqLen, mha.Dim})
	dK, _ := tensor.FromSlice(dKFlat, tensor.Shape{batch, seqLen, mha.Dim})
	dV, _ := tensor.FromSlice(dVFlat, tensor.Shape{batch, seqLen, mha.Dim})

	// Backward through Wq, Wk, Wv
	dx1, _ := mha.Wq.Backward(cache.X, dQ)
	dx2, _ := mha.Wk.Backward(cache.X, dK)
	dx3, _ := mha.Wv.Backward(cache.X, dV)

	// dx = dx1 + dx2 + dx3
	return addTensors(addTensors(dx1, dx2), dx3), nil
}

// ---- TransformerBlock Backward ----

// BlockCache stores intermediates for transformer block backward.
type BlockCache struct {
	X         *tensor.Tensor // input to block
	Normed1   *tensor.Tensor // after first layernorm
	AttnOut   *tensor.Tensor // after attention (before residual)
	PostAttn  *tensor.Tensor // after first residual
	Normed2   *tensor.Tensor // after second layernorm
	FFNOut    *tensor.Tensor // after FFN (before residual)
	AttnCache *AttentionCache
}

// ForwardWithCache runs transformer block and saves intermediates.
func (tb *TransformerBlock) ForwardWithCache(x *tensor.Tensor) (*tensor.Tensor, *BlockCache, error) {
	cache := &BlockCache{X: x}

	normed1, _ := tb.AttnNorm.Forward(x)
	cache.Normed1 = normed1

	attnOut, attnCache, err := tb.Attn.ForwardWithCache(normed1, true)
	if err != nil {
		return nil, nil, err
	}
	cache.AttnOut = attnOut
	cache.AttnCache = attnCache

	postAttn := addTensors(x, attnOut)
	cache.PostAttn = postAttn

	normed2, _ := tb.FFNNorm.Forward(postAttn)
	cache.Normed2 = normed2

	ffnOut, _ := tb.FFN.Forward(normed2)
	cache.FFNOut = ffnOut

	out := addTensors(postAttn, ffnOut)
	return out, cache, nil
}

// Backward computes gradients for transformer block.
func (tb *TransformerBlock) Backward(cache *BlockCache, dout *tensor.Tensor) (*tensor.Tensor, error) {
	// dout flows through second residual
	dFFNOut := dout
	dPostAttn := dout // copy via residual

	// Backward through FFN
	dNormed2, _ := tb.FFN.Backward(cache.Normed2, dFFNOut)

	// Backward through FFNNorm
	dPostAttn2, _ := tb.FFNNorm.Backward(cache.PostAttn, dNormed2)
	dPostAttn = addTensors(dPostAttn, dPostAttn2)

	// dPostAttn flows through first residual
	dAttnOut := dPostAttn
	dX := dPostAttn // copy via residual

	// Backward through Attention
	dNormed1, _ := tb.Attn.Backward(cache.AttnCache, dAttnOut)

	// Backward through AttnNorm
	dX2, _ := tb.AttnNorm.Backward(cache.X, dNormed1)
	dX = addTensors(dX, dX2)

	return dX, nil
}

// ---- LLM Backward ----

// LLMCache stores all intermediates for full model backward.
type LLMCache struct {
	Tokens      *tensor.Tensor
	Embeddings  *tensor.Tensor // [batch, seqLen, dim]
	BlockCaches []*BlockCache
	Normed      *tensor.Tensor // after final norm
}

// ForwardWithCache runs full LLM and saves intermediates.
func (m *LLM) ForwardWithCache(tokens *tensor.Tensor) (*tensor.Tensor, *LLMCache, error) {
	shape := tokens.Shape()
	batch := shape[0]
	seqLen := shape[1]

	cache := &LLMCache{Tokens: tokens}

	// Embedding
	var embeddings []*tensor.Tensor
	for b := 0; b < batch; b++ {
		bt, _ := sliceBatch(tokens, b, seqLen)
		emb, _ := m.TokEmbed.Forward(bt)
		embeddings = append(embeddings, emb)
	}
	x, _ := stackBatch(embeddings, batch, seqLen, m.Config.Dim)
	cache.Embeddings = x

	// Transformer blocks
	cache.BlockCaches = make([]*BlockCache, len(m.Layers))
	var err error
	for i, layer := range m.Layers {
		var bc *BlockCache
		x, bc, err = layer.ForwardWithCache(x)
		if err != nil {
			return nil, nil, err
		}
		cache.BlockCaches[i] = bc
	}

	// Final norm
	normed, _ := m.Norm.Forward(x)
	cache.Normed = normed

	// Output projection
	logits, _ := m.Output.Forward(normed)

	return logits, cache, nil
}

// Backward runs full model backward from logits gradient.
func (m *LLM) Backward(cache *LLMCache, dLogits *tensor.Tensor) error {
	// Backward through output projection
	dNormed, _ := m.Output.Backward(cache.Normed, dLogits)

	// Backward through final norm
	// Need the input to final norm = output of last transformer block
	lastBlockOut := cache.BlockCaches[len(cache.BlockCaches)-1]
	// Reconstruct: lastBlockOut = PostAttn + FFNOut
	finalNormInput := addTensors(lastBlockOut.PostAttn, lastBlockOut.FFNOut)
	dx, _ := m.Norm.Backward(finalNormInput, dNormed)

	// Backward through transformer blocks (reverse order)
	for i := len(m.Layers) - 1; i >= 0; i-- {
		var err error
		dx, err = m.Layers[i].Backward(cache.BlockCaches[i], dx)
		if err != nil {
			return err
		}
	}

	// Backward through embedding
	for b := 0; b < cache.Tokens.Shape()[0]; b++ {
		seqLen := cache.Tokens.Shape()[1]
		bt, _ := sliceBatch(cache.Tokens, b, seqLen)

		// Extract this batch's embedding gradient
		dxData := dx.ToFloat32Slice()
		dim := m.Config.Dim
		batchGrad := dxData[b*seqLen*dim : (b+1)*seqLen*dim]
		dEmb, _ := tensor.FromSlice(batchGrad, tensor.Shape{seqLen, dim})

		m.TokEmbed.Backward(bt, dEmb)
	}

	return nil
}

// ---- Utility functions ----

// rearrangeBSHD: [batch*seq, heads*headDim] → [batch, heads, seq, headDim] (flat)
func rearrangeBSHD(data []float32, batch, seqLen, numHeads, headDim int) []float32 {
	out := make([]float32, len(data))
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			for h := 0; h < numHeads; h++ {
				for d := 0; d < headDim; d++ {
					srcIdx := (b*seqLen+s)*numHeads*headDim + h*headDim + d
					dstIdx := ((b*numHeads+h)*seqLen+s)*headDim + d
					out[dstIdx] = data[srcIdx]
				}
			}
		}
	}
	return out
}

// rearrangeBHSD: [batch, heads, seq, headDim] → [batch, seq, heads*headDim] (flat)
func rearrangeBHSD(data []float32, batch, seqLen, numHeads, headDim int) []float32 {
	out := make([]float32, len(data))
	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			for s := 0; s < seqLen; s++ {
				for d := 0; d < headDim; d++ {
					srcIdx := ((b*numHeads+h)*seqLen+s)*headDim + d
					dstIdx := (b*seqLen+s)*numHeads*headDim + h*headDim + d
					out[dstIdx] = data[srcIdx]
				}
			}
		}
	}
	return out
}

func applyRoPEInPlace(data []float32, batch, numHeads, seqLen, headDim int, base float64) {
	halfDim := headDim / 2
	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			for pos := 0; pos < seqLen; pos++ {
				off := ((b*numHeads+h)*seqLen + pos) * headDim
				for i := 0; i < halfDim; i++ {
					freq := 1.0 / math.Pow(base, float64(2*i)/float64(headDim))
					angle := float64(pos) * freq
					cos := float32(math.Cos(angle))
					sin := float32(math.Sin(angle))
					x0 := data[off+i]
					x1 := data[off+halfDim+i]
					data[off+i] = x0*cos - x1*sin
					data[off+halfDim+i] = x0*sin + x1*cos
				}
			}
		}
	}
}

func ropeBackwardInPlace(data []float32, batch, numHeads, seqLen, headDim int, base float64) {
	halfDim := headDim / 2
	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			for pos := 0; pos < seqLen; pos++ {
				off := ((b*numHeads+h)*seqLen + pos) * headDim
				for i := 0; i < halfDim; i++ {
					freq := 1.0 / math.Pow(base, float64(2*i)/float64(headDim))
					angle := float64(pos) * freq
					cos := float32(math.Cos(angle))
					sin := float32(-math.Sin(angle)) // negative angle for backward
					x0 := data[off+i]
					x1 := data[off+halfDim+i]
					data[off+i] = x0*cos - x1*sin
					data[off+halfDim+i] = x0*sin + x1*cos
				}
			}
		}
	}
}
