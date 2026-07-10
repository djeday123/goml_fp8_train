package ops

import (
	"math"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/tensor"
)

// CrossEntropyLoss computes the cross-entropy loss between logits and targets.
// logits: [batch, seqLen, vocabSize] (float32)
// targets: [batch, seqLen] (int64)
// Returns: scalar loss tensor [1]
func CrossEntropyLoss(logits, targets *tensor.Tensor) (*tensor.Tensor, error) {
	logitsShape := logits.Shape()
	batch := logitsShape[0]
	seqLen := logitsShape[1]
	vocabSize := logitsShape[2]

	logitsData := logits.ToFloat32Slice()
	targetsData := targets.ToInt64Slice()

	totalLoss := float64(0)
	count := 0

	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			offset := (b*seqLen + s) * vocabSize
			target := int(targetsData[b*seqLen+s])

			if target < 0 { // padding token, skip
				continue
			}

			// Log-softmax: log(exp(x_target) / sum(exp(x_i)))
			// = x_target - log(sum(exp(x_i)))
			// With numerical stability: x_target - max - log(sum(exp(x_i - max)))
			maxVal := float64(-math.MaxFloat64)
			for v := 0; v < vocabSize; v++ {
				val := float64(logitsData[offset+v])
				if val > maxVal {
					maxVal = val
				}
			}

			sumExp := float64(0)
			for v := 0; v < vocabSize; v++ {
				sumExp += math.Exp(float64(logitsData[offset+v]) - maxVal)
			}
			logSumExp := maxVal + math.Log(sumExp)

			loss := logSumExp - float64(logitsData[offset+target])
			totalLoss += loss
			count++
		}
	}

	if count > 0 {
		totalLoss /= float64(count)
	}

	return tensor.FromSlice([]float32{float32(totalLoss)}, tensor.Shape{1})
}

// CrossEntropyBackward computes gradients of cross-entropy loss w.r.t. logits.
// Returns gradient tensor with same shape as logits: [batch, seqLen, vocabSize]
// Gradient = softmax(logits) - one_hot(targets), averaged over count.
func CrossEntropyBackward(logits, targets *tensor.Tensor) (*tensor.Tensor, error) {
	logitsShape := logits.Shape()
	batch := logitsShape[0]
	seqLen := logitsShape[1]
	vocabSize := logitsShape[2]

	logitsData := logits.ToFloat32Slice()
	targetsData := targets.ToInt64Slice()

	n := batch * seqLen * vocabSize
	gradData := make([]float32, n)

	count := 0
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			offset := (b*seqLen + s) * vocabSize
			target := int(targetsData[b*seqLen+s])

			if target < 0 {
				continue
			}
			count++

			// Softmax
			maxVal := float32(-math.MaxFloat32)
			for v := 0; v < vocabSize; v++ {
				if logitsData[offset+v] > maxVal {
					maxVal = logitsData[offset+v]
				}
			}

			sumExp := float32(0)
			for v := 0; v < vocabSize; v++ {
				gradData[offset+v] = float32(math.Exp(float64(logitsData[offset+v] - maxVal)))
				sumExp += gradData[offset+v]
			}

			for v := 0; v < vocabSize; v++ {
				gradData[offset+v] /= sumExp // softmax probability
			}
			gradData[offset+target] -= 1.0 // subtract one-hot
		}
	}

	// Average over count
	if count > 0 {
		scale := float32(1.0) / float32(count)
		for i := range gradData {
			gradData[i] *= scale
		}
	}

	bk, err := backend.Get(backend.CPU)
	if err != nil {
		return nil, err
	}

	byteLen := n * int(tensor.Float32.Size())
	store, err := bk.Alloc(byteLen)
	if err != nil {
		return nil, err
	}

	// Copy grad data to storage
	dst := tensor.SliceFromPtr[float32](store.Ptr(), n)
	copy(dst, gradData)

	return tensor.NewTensor(store, logitsShape, tensor.Float32), nil
}
