package fp8_test

import (
	"math"
	"testing"

	"github.com/djeday123/goml_fp8_train/fp8"
)

// ─── E4M3FN round-trip tests ────────────────────────────────────────────────

func TestQuantizeE4M3_Zero(t *testing.T) {
	if b := fp8.QuantizeE4M3(0); b != 0 {
		t.Errorf("QuantizeE4M3(0) = 0x%02x, want 0x00", b)
	}
}

func TestQuantizeE4M3_NaN(t *testing.T) {
	b := fp8.QuantizeE4M3(float32(math.NaN()))
	if b != 0x7F {
		t.Errorf("QuantizeE4M3(NaN) = 0x%02x, want 0x7F", b)
	}
}

func TestQuantizeE4M3_MaxValue(t *testing.T) {
	// 448 is the max representable value in E4M3FN.
	b := fp8.QuantizeE4M3(448.0)
	v := fp8.DequantizeE4M3(b)
	if math.Abs(float64(v-448.0)) > 1.0 {
		t.Errorf("round-trip 448.0: got %.3f", v)
	}
}

func TestQuantizeE4M3_Negative(t *testing.T) {
	b := fp8.QuantizeE4M3(-1.0)
	v := fp8.DequantizeE4M3(b)
	if math.Abs(float64(v+1.0)) > 0.1 {
		t.Errorf("round-trip -1.0: got %.3f", v)
	}
}

func TestQuantizeE4M3_Clamp(t *testing.T) {
	// Values above 448 should clamp to 448.
	b := fp8.QuantizeE4M3(1e6)
	v := fp8.DequantizeE4M3(b)
	if v > 448.0+1e-3 {
		t.Errorf("clamp test: %.3f > 448", v)
	}
}

func TestRoundTripE4M3(t *testing.T) {
	cases := []float32{0.5, 1.0, 2.0, 4.0, 16.0, 100.0, -0.5, -2.0}
	for _, want := range cases {
		b := fp8.QuantizeE4M3(want)
		got := fp8.DequantizeE4M3(b)
		// Allow up to 12.5% relative error (3 mantissa bits → 1/8 resolution).
		if math.Abs(float64(got-want)) > 0.125*math.Abs(float64(want))+1e-4 {
			t.Errorf("E4M3 round-trip %.4f → 0x%02x → %.4f (err %.4f)",
				want, b, got, math.Abs(float64(got-want)))
		}
	}
}

// ─── E5M2 round-trip tests ────────────────────────────────────────────────

func TestQuantizeE5M2_Zero(t *testing.T) {
	if b := fp8.QuantizeE5M2(0); b != 0 {
		t.Errorf("QuantizeE5M2(0) = 0x%02x, want 0x00", b)
	}
}

func TestRoundTripE5M2(t *testing.T) {
	cases := []float32{0.5, 1.0, 2.0, 4.0, 256.0, -1.0, -64.0}
	for _, want := range cases {
		b := fp8.QuantizeE5M2(want)
		got := fp8.DequantizeE5M2(b)
		// E5M2 has 2 mantissa bits → 25% relative resolution.
		if math.Abs(float64(got-want)) > 0.26*math.Abs(float64(want))+1e-4 {
			t.Errorf("E5M2 round-trip %.4f → 0x%02x → %.4f (err %.4f)",
				want, b, got, math.Abs(float64(got-want)))
		}
	}
}

func TestQuantizeE5M2_Inf(t *testing.T) {
	b := fp8.QuantizeE5M2(float32(math.Inf(1)))
	v := fp8.DequantizeE5M2(b)
	if !math.IsInf(float64(v), 1) {
		t.Errorf("E5M2(+Inf) dequant = %.3f, want +Inf", v)
	}
}

// ─── Tensor quantization tests ────────────────────────────────────────────

func TestTensorQuantizeFrom(t *testing.T) {
	src := []float32{1.0, -1.0, 0.5, -0.5, 2.0}
	ten := fp8.NewTensor([]int{5}, fp8.E4M3FN)
	ten.QuantizeFrom(src)

	dst := ten.Dequantize()
	for i, want := range src {
		err := math.Abs(float64(dst[i] - want))
		// Expect < 12.5% relative error after quantisation.
		maxErr := 0.125*math.Abs(float64(want)) + 1e-3
		if err > maxErr {
			t.Errorf("tensor[%d]: want %.4f got %.4f (err %.4f > %.4f)",
				i, want, dst[i], err, maxErr)
		}
	}
}

func TestTensorScaleInv(t *testing.T) {
	// The scale should be chosen so that max(abs(src)) / scale ≈ fp8_max.
	src := []float32{100.0, -200.0, 50.0}
	ten := fp8.NewTensor([]int{3}, fp8.E4M3FN)
	ten.QuantizeFrom(src)
	if ten.Scale <= 0 {
		t.Errorf("scale should be positive, got %.6f", ten.Scale)
	}
	if math.Abs(float64(ten.Scale*ten.ScaleInv)-1.0) > 1e-5 {
		t.Errorf("scale * scale_inv should be 1, got %.6f", ten.Scale*ten.ScaleInv)
	}
}

// ─── DelayedScaler tests ──────────────────────────────────────────────────

func TestDelayedScaler(t *testing.T) {
	ds := fp8.NewDelayedScaler(fp8.E4M3FN, 4)

	// Feed 4 steps with amax = 10 each time.
	src := []float32{5.0, -10.0, 3.0}
	for i := 0; i < 4; i++ {
		_ = ds.Quantize(src)
		ds.UpdateScale()
	}
	// After 4 steps the scale should be calibrated so that 10 maps near
	// E4M3FN max (448). Scale ≈ 448 / 10 = 44.8.
	expectedScale := float32(448.0 / 10.0)
	if math.Abs(float64(ds.CurrentScale-expectedScale)) > 1.0 {
		t.Errorf("delayed scaler: got scale %.3f, want ~%.3f",
			ds.CurrentScale, expectedScale)
	}
}

// ─── FP8 GEMM correctness test ────────────────────────────────────────────

// TestGEMM_Identity verifies that A * B = C for a trivial case where the
// error from FP8 quantisation is bounded.
func TestGEMM_Identity(t *testing.T) {
	// A = [[1,0],[0,1]] (identity), B = [[2,3],[4,5]]
	// Expected C = [[2,3],[4,5]]
	aF32 := []float32{1, 0, 0, 1}
	bF32 := []float32{2, 3, 4, 5}

	aT := fp8.NewTensor([]int{4}, fp8.E4M3FN)
	aT.QuantizeFrom(aF32)
	bT := fp8.NewTensor([]int{4}, fp8.E4M3FN)
	bT.QuantizeFrom(bF32)

	c := fp8.GEMM(
		aT.Data, aT.Scale, fp8.E4M3FN,
		bT.Data, bT.Scale, fp8.E4M3FN,
		2, 2, 2,
	)
	expected := []float32{2, 3, 4, 5}
	for i, want := range expected {
		if math.Abs(float64(c[i]-want)) > 0.5 {
			t.Errorf("GEMM identity c[%d] = %.3f, want %.3f", i, c[i], want)
		}
	}
}

// ─── FP8 Linear layer tests ───────────────────────────────────────────────

func TestLinearForwardShape(t *testing.T) {
	batchSize, in, out := 4, 8, 16
	layer := fp8.NewLinear(in, out)
	x := make([]float32, batchSize*in)
	for i := range x {
		x[i] = float32(i) * 0.01
	}
	y, err := layer.Forward(x, batchSize)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if len(y) != batchSize*out {
		t.Errorf("Forward output length %d, want %d", len(y), batchSize*out)
	}
}

func TestLinearBackwardShape(t *testing.T) {
	batchSize, in, out := 4, 8, 16
	layer := fp8.NewLinear(in, out)
	// Use non-zero inputs so that dW = X^T * dY is non-zero.
	x := make([]float32, batchSize*in)
	for i := range x {
		x[i] = float32(i+1) * 0.05
	}
	y, err := layer.Forward(x, batchSize)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	dY := make([]float32, len(y))
	for i := range dY {
		dY[i] = 0.1
	}
	dX, err := layer.Backward(dY, batchSize)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	if len(dX) != batchSize*in {
		t.Errorf("Backward dX length %d, want %d", len(dX), batchSize*in)
	}
	// Gradient should have been accumulated.
	hasGrad := false
	for _, g := range layer.GradWeight {
		if g != 0 {
			hasGrad = true
			break
		}
	}
	if !hasGrad {
		t.Error("Backward did not accumulate any gradient into GradWeight")
	}
}

func TestLinearForwardError(t *testing.T) {
	layer := fp8.NewLinear(8, 16)
	// Wrong input size should return an error.
	_, err := layer.Forward([]float32{1, 2, 3}, 4) // 3 != 4*8
	if err == nil {
		t.Error("expected error for wrong input size, got nil")
	}
}
