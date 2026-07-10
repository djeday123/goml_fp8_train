package fp8

import (
	"fmt"
	"math"
)

// Tensor is an n-dimensional array stored in FP8. Values are quantized with a
// per-tensor scale factor:  real_value ≈ fp8_value * scale_inv
// where scale_inv = 1 / scale.
type Tensor struct {
	Data     []uint8 // raw FP8 bytes
	Shape    []int
	DType    DType
	Scale    float32 // scale factor applied before quantization
	ScaleInv float32 // 1/Scale, used during dequantization
}

// NewTensor allocates a zero-initialised FP8 tensor with the given shape.
func NewTensor(shape []int, dtype DType) *Tensor {
	n := numel(shape)
	return &Tensor{
		Data:     make([]uint8, n),
		Shape:    append([]int(nil), shape...),
		DType:    dtype,
		Scale:    1.0,
		ScaleInv: 1.0,
	}
}

// Numel returns the total number of elements.
func (t *Tensor) Numel() int { return len(t.Data) }

// QuantizeFrom fills t with values from f32, choosing the scale automatically
// so that the maximum absolute value maps to the format's max representable
// magnitude.  The scale and scale_inv fields are updated in place.
func (t *Tensor) QuantizeFrom(f32 []float32) {
	if len(f32) != len(t.Data) {
		panic(fmt.Sprintf("fp8: QuantizeFrom size mismatch: want %d got %d", len(t.Data), len(f32)))
	}
	// Compute max absolute value.
	maxAbs := float32(0)
	for _, v := range f32 {
		if a := abs32(v); a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs == 0 || math.IsNaN(float64(maxAbs)) {
		maxAbs = 1.0
	}
	t.Scale = maxAbs / t.DType.MaxValue()
	if t.Scale == 0 {
		t.Scale = 1.0
	}
	t.ScaleInv = 1.0 / t.Scale

	// Quantize each element.
	switch t.DType {
	case E4M3FN:
		for i, v := range f32 {
			t.Data[i] = QuantizeE4M3(v * t.ScaleInv)
		}
	case E5M2:
		for i, v := range f32 {
			t.Data[i] = QuantizeE5M2(v * t.ScaleInv)
		}
	}
}

// Dequantize expands t back to float32, applying the stored scale factor.
func (t *Tensor) Dequantize() []float32 {
	out := make([]float32, len(t.Data))
	switch t.DType {
	case E4M3FN:
		for i, b := range t.Data {
			out[i] = DequantizeE4M3(b) * t.Scale
		}
	case E5M2:
		for i, b := range t.Data {
			out[i] = DequantizeE5M2(b) * t.Scale
		}
	}
	return out
}

// QuantizeWithScale quantizes using a pre-determined scale (used during
// delayed-scaling so the scale is fixed from the previous iteration).
func (t *Tensor) QuantizeWithScale(f32 []float32, scale float32) {
	if len(f32) != len(t.Data) {
		panic(fmt.Sprintf("fp8: QuantizeWithScale size mismatch: want %d got %d", len(t.Data), len(f32)))
	}
	t.Scale = scale
	if scale == 0 {
		t.Scale = 1.0
	}
	t.ScaleInv = 1.0 / t.Scale
	scaleInv := t.ScaleInv
	switch t.DType {
	case E4M3FN:
		for i, v := range f32 {
			t.Data[i] = QuantizeE4M3(v * scaleInv)
		}
	case E5M2:
		for i, v := range f32 {
			t.Data[i] = QuantizeE5M2(v * scaleInv)
		}
	}
}

// helper

func numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
