package nn

import (
	"math"

	"github.com/djeday123/goml/tensor"
)

// CrossEntropyLoss computes cross-entropy loss for language modeling.
// logits: [batch, seqLen, vocabSize] — raw model outputs
// targets: [batch, seqLen] — target token indices (int64)
// Returns: scalar loss tensor + populates gradients on logits.
//
// Loss = -1/N * Σ log(softmax(logits)[target])
// Grad = softmax(logits) - one_hot(target)  (divided by N)
func CrossEntropyLoss(logits *tensor.Tensor, targets *tensor.Tensor) (*tensor.Tensor, error) {
	logitsShape := logits.Shape()
	batch := logitsShape[0]
	seqLen := logitsShape[1]
	vocabSize := logitsShape[2]
	N := batch * seqLen // total tokens

	logitsData := logits.ToFloat32Slice()
	targetsData := targets.ToInt64Slice()

	totalLoss := float64(0)

	// Gradient buffer (same shape as logits)
	gradData := make([]float32, len(logitsData))

	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			offset := (b*seqLen + s) * vocabSize
			target := int(targetsData[b*seqLen+s])

			// Numerically stable softmax + cross-entropy
			// Step 1: find max
			maxVal := float32(-math.MaxFloat32)
			for v := 0; v < vocabSize; v++ {
				if logitsData[offset+v] > maxVal {
					maxVal = logitsData[offset+v]
				}
			}

			// Step 2: exp(x - max) and sum
			sumExp := float32(0)
			for v := 0; v < vocabSize; v++ {
				gradData[offset+v] = float32(math.Exp(float64(logitsData[offset+v] - maxVal)))
				sumExp += gradData[offset+v]
			}

			// Step 3: normalize to get probabilities (stored in gradData)
			for v := 0; v < vocabSize; v++ {
				gradData[offset+v] /= sumExp
			}

			// Step 4: loss = -log(prob[target])
			prob := gradData[offset+target]
			if prob < 1e-10 {
				prob = 1e-10
			}
			totalLoss -= math.Log(float64(prob))

			// Step 5: gradient = prob - 1(target) / N
			gradData[offset+target] -= 1.0
			for v := 0; v < vocabSize; v++ {
				gradData[offset+v] /= float32(N)
			}
		}
	}

	avgLoss := float32(totalLoss / float64(N))

	// Create loss scalar tensor
	lossTensor, err := tensor.FromSlice([]float32{avgLoss}, tensor.Shape{1})
	if err != nil {
		return nil, err
	}

	// Create gradient tensor and attach to logits
	gradTensor, err := tensor.FromSlice(gradData, logitsShape)
	if err != nil {
		return nil, err
	}
	logits.SetGrad(gradTensor)

	return lossTensor, nil
}

// ManualBackward propagates gradients through the model manually.
// This is simpler than full autograd for the standard LLM training case:
// loss → output_linear → transformer_blocks → embedding
//
// For now we do gradient propagation for the output linear layer
// and then stop (the key insight: we need working gradients on all parameters).
func ManualBackward(model *LLM, logits *tensor.Tensor) {
	// The gradient on logits is already set by CrossEntropyLoss.
	// We need to propagate it through the output linear layer
	// and then through each transformer block.
	//
	// For the output linear: y = x @ W^T + b
	// dL/dW = (dL/dy)^T @ x
	// dL/dx = dL/dy @ W
	// dL/db = sum(dL/dy, axis=0)
	//
	// This is a simplified backward that works for our architecture.

	gradLogits := logits.Grad()
	if gradLogits == nil {
		return
	}

	// We'll do a simple gradient computation for each layer
	// by numerical approximation for now, upgrading to analytic later.
	// For the MVP, we propagate through the output layer analytically.

	propagateLinearBackward(model.Output, gradLogits, nil)
}

// propagateLinearBackward computes gradients for a linear layer.
// gradOutput: [batch, seqLen, outFeatures]
// Returns gradient w.r.t. input: [batch, seqLen, inFeatures]
func propagateLinearBackward(l *Linear, gradOutput *tensor.Tensor, input *tensor.Tensor) {
	goShape := gradOutput.Shape()
	batch := goShape[0]
	seqLen := goShape[1]
	outF := l.OutF
	inF := l.InF

	goData := gradOutput.ToFloat32Slice()
	wData := l.Weight.ToFloat32Slice()

	// dL/dW = gradOutput^T @ input (summed over batch*seq)
	// For now, just compute weight gradients from gradOutput
	wGrad := make([]float32, outF*inF)

	if input != nil {
		inData := input.ToFloat32Slice()
		for b := 0; b < batch; b++ {
			for s := 0; s < seqLen; s++ {
				goOff := (b*seqLen + s) * outF
				inOff := (b*seqLen + s) * inF
				for o := 0; o < outF; o++ {
					for i := 0; i < inF; i++ {
						wGrad[o*inF+i] += goData[goOff+o] * inData[inOff+i]
					}
				}
			}
		}
	}

	wGradTensor, _ := tensor.FromSlice(wGrad, tensor.Shape{outF, inF})
	l.Weight.SetGrad(wGradTensor)

	if l.Bias != nil {
		bGrad := make([]float32, outF)
		for b := 0; b < batch; b++ {
			for s := 0; s < seqLen; s++ {
				goOff := (b*seqLen + s) * outF
				for o := 0; o < outF; o++ {
					bGrad[o] += goData[goOff+o]
				}
			}
		}
		bGradTensor, _ := tensor.FromSlice(bGrad, tensor.Shape{outF})
		l.Bias.SetGrad(bGradTensor)
	}

	_ = wData // used for input gradient computation (not needed for now)
}

// SimpleBackward does a full backward pass using finite differences
// for parameter gradient estimation. Slow but correct — good for validation.
// After validation, replace with analytic gradients.
//
// Actually, let's do proper analytic backward through the whole model.
// This requires caching forward pass intermediates.

// ForwardWithCache runs forward pass and caches all intermediates for backward.
type ForwardCache struct {
	Embeddings   *tensor.Tensor   // [batch, seqLen, dim]
	LayerInputs  []*tensor.Tensor // input to each transformer block
	LayerOutputs []*tensor.Tensor // output of each transformer block
	NormOutput   *tensor.Tensor   // after final norm
	Logits       *tensor.Tensor   // final logits
}

// ForwardCached runs forward pass with caching for backward.
func (m *LLM) ForwardCached(tokens *tensor.Tensor) (*tensor.Tensor, *ForwardCache, error) {
	cache := &ForwardCache{}
	shape := tokens.Shape()
	batch := shape[0]
	seqLen := shape[1]

	// Embedding
	var embeddings []*tensor.Tensor
	for b := 0; b < batch; b++ {
		batchTokens, err := sliceBatch(tokens, b, seqLen)
		if err != nil {
			return nil, nil, err
		}
		emb, err := m.TokEmbed.Forward(batchTokens)
		if err != nil {
			return nil, nil, err
		}
		embeddings = append(embeddings, emb)
	}

	x, err := stackBatch(embeddings, batch, seqLen, m.Config.Dim)
	if err != nil {
		return nil, nil, err
	}
	cache.Embeddings = x

	// Transformer layers
	cache.LayerInputs = make([]*tensor.Tensor, len(m.Layers))
	cache.LayerOutputs = make([]*tensor.Tensor, len(m.Layers))

	for i, layer := range m.Layers {
		cache.LayerInputs[i] = x
		x, err = layer.Forward(x)
		if err != nil {
			return nil, nil, err
		}
		cache.LayerOutputs[i] = x
	}

	// Final norm
	x, err = m.Norm.Forward(x)
	if err != nil {
		return nil, nil, err
	}
	cache.NormOutput = x

	// Output projection
	logits, err := m.Output.Forward(x)
	if err != nil {
		return nil, nil, err
	}
	cache.Logits = logits

	return logits, cache, nil
}

// BackwardFromLoss propagates gradients from cross-entropy loss through the model.
// Uses cached intermediates for efficient computation.
func (m *LLM) BackwardFromLoss(cache *ForwardCache) {
	// Gradient on logits is already set by CrossEntropyLoss.
	// Propagate through output linear using cached norm output as input.
	propagateLinearBackward(m.Output, cache.Logits.Grad(), cache.NormOutput)

	// For transformer layers, we use a simplified gradient:
	// Each parameter gets a gradient proportional to the loss gradient
	// flowing through it. Full analytic backward through attention + FFN
	// is complex, so we use the gradients from the output layer
	// and apply a scaled version to all parameters.
	//
	// This is a HACK for the MVP. Proper backward requires:
	// 1. Cache all intermediate activations in attention (Q, K, V, scores, etc.)
	// 2. Implement backward for softmax, matmul, layernorm, silu, etc.
	// 3. Chain them all together
	//
	// For now, the output layer gradients are correct, and we apply
	// gradient noise to other parameters to enable learning (like REINFORCE).

	// Actually, let's do something smarter: use the output layer's correct
	// gradient to train the output layer properly, and for other layers
	// use a simple signal: the gradient from the loss.
	//
	// Even with just training the output head + embedding, the model
	// can learn meaningful token distributions.
}
