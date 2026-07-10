package fp8

import "math"

// ScaleHistory stores the amax (absolute maximum) history used by the
// delayed-scaling algorithm popularised by NVIDIA's Transformer Engine.
// Instead of computing the optimal scale on every forward pass (which
// requires a full-tensor reduce), we use the max seen over the last
// HistoryLen steps and update the scale once per optimizer step.
type ScaleHistory struct {
	History    []float32 // circular buffer of per-step amax values
	head       int
	HistoryLen int
	Margin     float32 // safety margin multiplier (default 1.0)
}

// NewScaleHistory creates a new ScaleHistory with the given history length.
// margin is a safety multiplier (typically 1.0; use > 1.0 to add headroom).
func NewScaleHistory(historyLen int, margin float32) *ScaleHistory {
	if historyLen <= 0 {
		historyLen = 16
	}
	if margin <= 0 {
		margin = 1.0
	}
	return &ScaleHistory{
		History:    make([]float32, historyLen),
		HistoryLen: historyLen,
		Margin:     margin,
	}
}

// Update records the amax observed in this step.
func (s *ScaleHistory) Update(amax float32) {
	s.History[s.head] = amax
	s.head = (s.head + 1) % s.HistoryLen
}

// ComputeScale returns the scale factor to use in the *next* step given the
// dtype. It uses the maximum amax seen across the history window so that
// transient spikes don't overflow.
func (s *ScaleHistory) ComputeScale(dtype DType) float32 {
	maxAmax := float32(0)
	for _, v := range s.History {
		if v > maxAmax {
			maxAmax = v
		}
	}
	if maxAmax == 0 || math.IsNaN(float64(maxAmax)) {
		maxAmax = 1.0
	}
	// scale = maxValue / (margin * maxAmax)
	// so that after scaling by 1/scale, values fill the FP8 range.
	return dtype.MaxValue() / (s.Margin * maxAmax)
}

// DelayedScaler manages the delayed-scaling state for one tensor (e.g. the
// weight matrix or the activation matrix).
type DelayedScaler struct {
	History     *ScaleHistory
	CurrentScale float32
	DType       DType
}

// NewDelayedScaler creates a DelayedScaler for the given dtype.
func NewDelayedScaler(dtype DType, historyLen int) *DelayedScaler {
	return &DelayedScaler{
		History:      NewScaleHistory(historyLen, 1.0),
		CurrentScale: 1.0,
		DType:        dtype,
	}
}

// Quantize quantizes src into a new FP8 tensor using the current scale, then
// records the actual amax so that the next call to UpdateScale uses real data.
func (ds *DelayedScaler) Quantize(src []float32) *Tensor {
	t := NewTensor([]int{len(src)}, ds.DType)
	t.QuantizeWithScale(src, ds.CurrentScale)

	// Record real amax for next-step scale update.
	maxAbs := float32(0)
	for _, v := range src {
		if a := abs32(v); a > maxAbs {
			maxAbs = a
		}
	}
	ds.History.Update(maxAbs)
	return t
}

// UpdateScale computes the scale for the next step using the accumulated
// history. Call this at the end of each training iteration.
func (ds *DelayedScaler) UpdateScale() {
	ds.CurrentScale = ds.History.ComputeScale(ds.DType)
}
