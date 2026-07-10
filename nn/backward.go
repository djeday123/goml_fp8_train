package nn

import (
	"math"

	"github.com/djeday123/goml/tensor"
)

// ---- Linear Backward ----

// Backward computes gradients for Linear layer.
// dout: [batch, seqLen, outF] → dx: [batch, seqLen, inF]
// Also sets gradients on Weight and Bias.
func (l *Linear) Backward(x, dout *tensor.Tensor) (*tensor.Tensor, error) {
	xShape := x.Shape()
	batch := 1
	if len(xShape) == 3 {
		batch = xShape[0] * xShape[1]
	} else if len(xShape) == 2 {
		batch = xShape[0]
	}

	xData := x.ToFloat32Slice()
	doutData := dout.ToFloat32Slice()
	wData := l.Weight.ToFloat32Slice()

	// dx = dout @ W  (not transposed, since W is [outF, inF])
	dxData := make([]float32, batch*l.InF)
	for b := 0; b < batch; b++ {
		for i := 0; i < l.InF; i++ {
			sum := float32(0)
			for o := 0; o < l.OutF; o++ {
				sum += doutData[b*l.OutF+o] * wData[o*l.InF+i]
			}
			dxData[b*l.InF+i] = sum
		}
	}

	// dW = dout^T @ x → [outF, inF]
	dWData := make([]float32, l.OutF*l.InF)
	for b := 0; b < batch; b++ {
		for o := 0; o < l.OutF; o++ {
			for i := 0; i < l.InF; i++ {
				dWData[o*l.InF+i] += doutData[b*l.OutF+o] * xData[b*l.InF+i]
			}
		}
	}

	// Set weight gradient
	dW, err := tensor.FromSlice(dWData, tensor.Shape{l.OutF, l.InF})
	if err != nil {
		return nil, err
	}
	accumulateGrad(l.Weight, dW)

	// dBias = sum(dout, axis=0)
	if l.Bias != nil {
		dBData := make([]float32, l.OutF)
		for b := 0; b < batch; b++ {
			for o := 0; o < l.OutF; o++ {
				dBData[o] += doutData[b*l.OutF+o]
			}
		}
		dB, err := tensor.FromSlice(dBData, tensor.Shape{l.OutF})
		if err != nil {
			return nil, err
		}
		accumulateGrad(l.Bias, dB)
	}

	dx, err := tensor.FromSlice(dxData, xShape)
	if err != nil {
		return nil, err
	}
	return dx, nil
}

// ---- LayerNorm Backward ----

// Backward computes gradients for LayerNorm.
func (ln *LayerNorm) Backward(x, dout *tensor.Tensor) (*tensor.Tensor, error) {
	xData := x.ToFloat32Slice()
	doutData := dout.ToFloat32Slice()
	gammaData := ln.Gamma.ToFloat32Slice()

	shape := x.Shape()
	normSize := shape[len(shape)-1]
	batchSize := x.NumElements() / normSize

	dxData := make([]float32, len(xData))
	dGamma := make([]float32, normSize)
	dBeta := make([]float32, normSize)

	for b := 0; b < batchSize; b++ {
		off := b * normSize

		// Recompute forward stats
		mean := float32(0)
		for i := 0; i < normSize; i++ {
			mean += xData[off+i]
		}
		mean /= float32(normSize)

		variance := float32(0)
		for i := 0; i < normSize; i++ {
			d := xData[off+i] - mean
			variance += d * d
		}
		variance /= float32(normSize)
		invStd := float32(1.0 / math.Sqrt(float64(variance)+ln.Eps))

		// Normalized values
		xNorm := make([]float32, normSize)
		for i := 0; i < normSize; i++ {
			xNorm[i] = (xData[off+i] - mean) * invStd
		}

		// dGamma, dBeta
		for i := 0; i < normSize; i++ {
			dGamma[i] += doutData[off+i] * xNorm[i]
			dBeta[i] += doutData[off+i]
		}

		// dx
		// dx = (1/N) * invStd * (N*dy_hat - sum(dy_hat) - xnorm*sum(dy_hat*xnorm))
		// where dy_hat = dout * gamma
		dyHat := make([]float32, normSize)
		sumDyHat := float32(0)
		sumDyHatXnorm := float32(0)
		for i := 0; i < normSize; i++ {
			dyHat[i] = doutData[off+i] * gammaData[i]
			sumDyHat += dyHat[i]
			sumDyHatXnorm += dyHat[i] * xNorm[i]
		}

		invN := float32(1.0) / float32(normSize)
		for i := 0; i < normSize; i++ {
			dxData[off+i] = invStd * invN * (float32(normSize)*dyHat[i] - sumDyHat - xNorm[i]*sumDyHatXnorm)
		}
	}

	dG, _ := tensor.FromSlice(dGamma, tensor.Shape{normSize})
	dB, _ := tensor.FromSlice(dBeta, tensor.Shape{normSize})
	accumulateGrad(ln.Gamma, dG)
	accumulateGrad(ln.Beta, dB)

	dx, _ := tensor.FromSlice(dxData, shape)
	return dx, nil
}

// ---- FeedForward Backward ----

// Backward computes gradients for FFN.
func (ff *FeedForward) Backward(x, dout *tensor.Tensor) (*tensor.Tensor, error) {
	if ff.UseSwiGLU {
		return ff.backwardSwiGLU(x, dout)
	}
	return ff.backwardStandard(x, dout)
}

func (ff *FeedForward) backwardSwiGLU(x, dout *tensor.Tensor) (*tensor.Tensor, error) {
	// Forward: gate = SiLU(W1(x)), up = W3(x), hidden = gate * up, out = W2(hidden)

	// Recompute forward
	w1out, _ := ff.W1.Forward(x)
	gate := siluForward(w1out)
	up, _ := ff.W3.Forward(x)
	hidden := mulElementwise(gate, up)

	// Backward through W2
	dHidden, err := ff.W2.Backward(hidden, dout)
	if err != nil {
		return nil, err
	}

	// Backward through gate * up
	dHiddenData := dHidden.ToFloat32Slice()
	gateData := gate.ToFloat32Slice()
	upData := up.ToFloat32Slice()

	dGateData := make([]float32, len(gateData))
	dUpData := make([]float32, len(upData))
	for i := range dHiddenData {
		dGateData[i] = dHiddenData[i] * upData[i]
		dUpData[i] = dHiddenData[i] * gateData[i]
	}

	dGate, _ := tensor.FromSlice(dGateData, gate.Shape())
	dUp, _ := tensor.FromSlice(dUpData, up.Shape())

	// Backward through SiLU
	dSilu := siluBackward(w1out, dGate)

	// Backward through W1 and W3
	dx1, err := ff.W1.Backward(x, dSilu)
	if err != nil {
		return nil, err
	}
	dx3, err := ff.W3.Backward(x, dUp)
	if err != nil {
		return nil, err
	}

	// dx = dx1 + dx3
	return addTensors(dx1, dx3), nil
}

func (ff *FeedForward) backwardStandard(x, dout *tensor.Tensor) (*tensor.Tensor, error) {
	// Forward: h = GELU(W1(x)), out = W2(h)
	w1out, _ := ff.W1.Forward(x)
	h := geluForward(w1out)

	// Backward through W2
	dH, err := ff.W2.Backward(h, dout)
	if err != nil {
		return nil, err
	}

	// Backward through GELU
	dGelu := geluBackward(w1out, dH)

	// Backward through W1
	return ff.W1.Backward(x, dGelu)
}

// ---- Embedding Backward ----

// Backward computes gradients for Embedding.
// dout: [seqLen, embedDim] → scatters gradients back to Weight
func (e *Embedding) Backward(indices *tensor.Tensor, dout *tensor.Tensor) error {
	iData := indices.ToInt64Slice()
	doutData := dout.ToFloat32Slice()
	seqLen := len(iData)

	dWData := make([]float32, e.VocabSize*e.EmbedDim)
	for s := 0; s < seqLen; s++ {
		idx := int(iData[s])
		for d := 0; d < e.EmbedDim; d++ {
			dWData[idx*e.EmbedDim+d] += doutData[s*e.EmbedDim+d]
		}
	}

	dW, _ := tensor.FromSlice(dWData, tensor.Shape{e.VocabSize, e.EmbedDim})
	accumulateGrad(e.Weight, dW)
	return nil
}

// ---- Helper functions ----

func accumulateGrad(param, grad *tensor.Tensor) {
	if param.Grad() == nil {
		param.SetGrad(grad)
	} else {
		pGrad := param.Grad().ToFloat32Slice()
		gData := grad.ToFloat32Slice()
		for i := range pGrad {
			pGrad[i] += gData[i]
		}
	}
}

func siluForward(x *tensor.Tensor) *tensor.Tensor {
	data := x.ToFloat32Slice()
	out := make([]float32, len(data))
	for i, v := range data {
		sig := float32(1.0 / (1.0 + math.Exp(float64(-v))))
		out[i] = v * sig
	}
	t, _ := tensor.FromSlice(out, x.Shape())
	return t
}

func siluBackward(x, dout *tensor.Tensor) *tensor.Tensor {
	xData := x.ToFloat32Slice()
	dData := dout.ToFloat32Slice()
	out := make([]float32, len(xData))
	for i, v := range xData {
		sig := float32(1.0 / (1.0 + math.Exp(float64(-v))))
		// d(x*sigmoid(x))/dx = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
		//                    = sigmoid(x) * (1 + x*(1-sigmoid(x)))
		out[i] = dData[i] * sig * (1 + v*(1-sig))
	}
	t, _ := tensor.FromSlice(out, x.Shape())
	return t
}

func geluForward(x *tensor.Tensor) *tensor.Tensor {
	data := x.ToFloat32Slice()
	c := float32(math.Sqrt(2.0 / math.Pi))
	out := make([]float32, len(data))
	for i, v := range data {
		out[i] = 0.5 * v * (1 + float32(math.Tanh(float64(c*(v+0.044715*v*v*v)))))
	}
	t, _ := tensor.FromSlice(out, x.Shape())
	return t
}

func geluBackward(x, dout *tensor.Tensor) *tensor.Tensor {
	xData := x.ToFloat32Slice()
	dData := dout.ToFloat32Slice()
	c := float64(math.Sqrt(2.0 / math.Pi))
	out := make([]float32, len(xData))
	for i, v := range xData {
		vf := float64(v)
		inner := c * (vf + 0.044715*vf*vf*vf)
		tanh := math.Tanh(inner)
		dtanh := 1 - tanh*tanh
		dinner := c * (1 + 3*0.044715*vf*vf)
		grad := 0.5*(1+tanh) + 0.5*vf*dtanh*dinner
		out[i] = dData[i] * float32(grad)
	}
	t, _ := tensor.FromSlice(out, x.Shape())
	return t
}

func mulElementwise(a, b *tensor.Tensor) *tensor.Tensor {
	aData := a.ToFloat32Slice()
	bData := b.ToFloat32Slice()
	out := make([]float32, len(aData))
	for i := range out {
		out[i] = aData[i] * bData[i]
	}
	t, _ := tensor.FromSlice(out, a.Shape())
	return t
}

func addTensors(a, b *tensor.Tensor) *tensor.Tensor {
	aData := a.ToFloat32Slice()
	bData := b.ToFloat32Slice()
	out := make([]float32, len(aData))
	for i := range out {
		out[i] = aData[i] + bData[i]
	}
	t, _ := tensor.FromSlice(out, a.Shape())
	return t
}
