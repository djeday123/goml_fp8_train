package nn

import (
	"math"

	"github.com/djeday123/goml/tensor"
)

// AdamW implements the AdamW optimizer (decoupled weight decay).
// AdamW: θ = θ - lr * (m_hat / (sqrt(v_hat) + eps) + wd * θ)
type AdamW struct {
	Params      []*tensor.Tensor
	LR          float64
	Beta1       float64
	Beta2       float64
	Eps         float64
	WeightDecay float64
	Step        int

	// State: first and second moment estimates per parameter
	M [][]float32 // first moment (mean of gradients)
	V [][]float32 // second moment (mean of squared gradients)
}

// NewAdamW creates an AdamW optimizer.
func NewAdamW(params []*tensor.Tensor, lr, beta1, beta2, eps, weightDecay float64) *AdamW {
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
		Beta1:       beta1,
		Beta2:       beta2,
		Eps:         eps,
		WeightDecay: weightDecay,
		M:           m,
		V:           v,
	}
}

// DefaultAdamW creates AdamW with standard hyperparameters.
func DefaultAdamW(params []*tensor.Tensor, lr float64) *AdamW {
	return NewAdamW(params, lr, 0.9, 0.999, 1e-8, 0.01)
}

// ZeroGrad clears all parameter gradients.
func (opt *AdamW) ZeroGrad() {
	for _, p := range opt.Params {
		if g := p.Grad(); g != nil {
			gData := g.ToFloat32Slice()
			for i := range gData {
				gData[i] = 0
			}
		}
	}
}

// StepUpdate performs one optimization step.
func (opt *AdamW) StepUpdate() {
	opt.Step++
	t := float64(opt.Step)

	// Bias correction factors
	bc1 := 1.0 - math.Pow(opt.Beta1, t)
	bc2 := 1.0 - math.Pow(opt.Beta2, t)

	lr := opt.LR

	for i, p := range opt.Params {
		grad := p.Grad()
		if grad == nil {
			continue
		}

		pData := p.ToFloat32Slice()
		gData := grad.ToFloat32Slice()
		mData := opt.M[i]
		vData := opt.V[i]

		for j := 0; j < len(pData); j++ {
			g := gData[j]

			// Update biased first moment: m = β1*m + (1-β1)*g
			mData[j] = float32(opt.Beta1)*mData[j] + float32(1-opt.Beta1)*g

			// Update biased second moment: v = β2*v + (1-β2)*g²
			vData[j] = float32(opt.Beta2)*vData[j] + float32(1-opt.Beta2)*g*g

			// Bias-corrected estimates
			mHat := float64(mData[j]) / bc1
			vHat := float64(vData[j]) / bc2

			// AdamW update: param -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * param)
			update := mHat / (math.Sqrt(vHat) + opt.Eps)
			pData[j] -= float32(lr) * (float32(update) + float32(opt.WeightDecay)*pData[j])
		}
	}
}

// SetLR updates the learning rate (for scheduling).
func (opt *AdamW) SetLR(lr float64) {
	opt.LR = lr
}

// CosineSchedule returns the learning rate for a cosine annealing schedule.
// lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * step / total_steps))
func CosineSchedule(step, warmupSteps, totalSteps int, maxLR, minLR float64) float64 {
	if step < warmupSteps {
		// Linear warmup
		return maxLR * float64(step) / float64(warmupSteps)
	}
	// Cosine decay
	progress := float64(step-warmupSteps) / float64(totalSteps-warmupSteps)
	if progress > 1 {
		progress = 1
	}
	return minLR + 0.5*(maxLR-minLR)*(1+math.Cos(math.Pi*progress))
}
