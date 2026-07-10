package optim

import (
	"math"

	"github.com/djeday123/goml/tensor"
)

// AdamW implements the AdamW optimizer (decoupled weight decay).
// Standard hyperparameters follow the original paper + LLM best practices.
type AdamW struct {
	Params      []*tensor.Tensor
	LR          float64 // learning rate
	Beta1       float64 // first moment decay (default 0.9)
	Beta2       float64 // second moment decay (default 0.95 for LLMs)
	Eps         float64 // numerical stability (default 1e-8)
	WeightDecay float64 // L2 regularization (default 0.1 for LLMs)
	MaxGradNorm float64 // gradient clipping (0 = disabled)

	// State
	m    [][]float32 // first moment (mean of gradients)
	v    [][]float32 // second moment (mean of squared gradients)
	step int
}

// NewAdamW creates an optimizer with LLM-tuned defaults.
func NewAdamW(params []*tensor.Tensor, lr float64) *AdamW {
	m := make([][]float32, len(params))
	v := make([][]float32, len(params))
	for i, p := range params {
		n := p.NumElements()
		m[i] = make([]float32, n)
		v[i] = make([]float32, n)
	}

	return &AdamW{
		Params:      params,
		LR:          lr,
		Beta1:       0.9,
		Beta2:       0.95,
		Eps:         1e-8,
		WeightDecay: 0.1,
		MaxGradNorm: 1.0,
		m:           m,
		v:           v,
	}
}

// Step performs one optimization step.
// Gradients must be set on each parameter tensor before calling.
func (opt *AdamW) Step() {
	opt.step++

	// Gradient clipping (global norm)
	if opt.MaxGradNorm > 0 {
		opt.clipGradNorm()
	}

	// Bias correction factors
	bc1 := 1.0 - math.Pow(opt.Beta1, float64(opt.step))
	bc2 := 1.0 - math.Pow(opt.Beta2, float64(opt.step))

	lr := opt.LR

	for i, param := range opt.Params {
		if param.Grad() == nil {
			continue
		}

		pData := param.ToFloat32Slice()
		gData := param.Grad().ToFloat32Slice()
		m := opt.m[i]
		v := opt.v[i]

		for j := 0; j < len(pData); j++ {
			g := gData[j]

			// Update moments
			m[j] = float32(opt.Beta1)*m[j] + float32(1-opt.Beta1)*g
			v[j] = float32(opt.Beta2)*v[j] + float32(1-opt.Beta2)*g*g

			// Bias-corrected moments
			mHat := float64(m[j]) / bc1
			vHat := float64(v[j]) / bc2

			// Adam update
			update := mHat / (math.Sqrt(vHat) + opt.Eps)

			// Decoupled weight decay (AdamW)
			pData[j] -= float32(lr) * (float32(update) + float32(opt.WeightDecay)*pData[j])
		}
	}
}

// ZeroGrad clears all gradients.
func (opt *AdamW) ZeroGrad() {
	for _, p := range opt.Params {
		if p.Grad() != nil {
			gData := p.Grad().ToFloat32Slice()
			for i := range gData {
				gData[i] = 0
			}
		}
	}
}

// clipGradNorm clips gradients by global L2 norm.
func (opt *AdamW) clipGradNorm() {
	// Compute global norm
	totalNorm := float64(0)
	for _, p := range opt.Params {
		if p.Grad() == nil {
			continue
		}
		gData := p.Grad().ToFloat32Slice()
		for _, g := range gData {
			totalNorm += float64(g) * float64(g)
		}
	}
	totalNorm = math.Sqrt(totalNorm)

	if totalNorm <= opt.MaxGradNorm {
		return
	}

	// Scale gradients
	scale := float32(opt.MaxGradNorm / totalNorm)
	for _, p := range opt.Params {
		if p.Grad() == nil {
			continue
		}
		gData := p.Grad().ToFloat32Slice()
		for i := range gData {
			gData[i] *= scale
		}
	}
}

// GetLR returns current learning rate.
func (opt *AdamW) GetLR() float64 {
	return opt.LR
}

// SetLR updates the learning rate (for scheduling).
func (opt *AdamW) SetLR(lr float64) {
	opt.LR = lr
}

// CosineSchedule computes learning rate with warmup + cosine decay.
func CosineSchedule(step, warmupSteps, totalSteps int, maxLR, minLR float64) float64 {
	if step < warmupSteps {
		// Linear warmup
		return maxLR * float64(step) / float64(warmupSteps)
	}

	// Cosine decay
	progress := float64(step-warmupSteps) / float64(totalSteps-warmupSteps)
	if progress > 1.0 {
		progress = 1.0
	}
	return minLR + 0.5*(maxLR-minLR)*(1.0+math.Cos(math.Pi*progress))
}
