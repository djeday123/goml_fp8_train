// Package fp8 implements FP8 (8-bit floating point) training primitives for
// GoML. It targets NVIDIA Hopper (H100/H200) GPUs which provide native FP8
// tensor-core acceleration, yielding ~652 TFLOPS forward and ~285 TFLOPS
// backward for large matrix workloads.
//
// Two FP8 formats are supported:
//   - E4M3FN  (1 sign, 4 exponent, 3 mantissa bits) — used for activations and
//     weights during the forward pass; larger mantissa preserves more precision.
//   - E5M2    (1 sign, 5 exponent, 2 mantissa bits) — used for gradients during
//     the backward pass; larger exponent range handles the wider gradient
//     distribution.
package fp8

import (
	"math"
)

// DType identifies an FP8 numerical format.
type DType uint8

const (
	// E4M3FN is the NVIDIA fp8_e4m3fn format (finite, no NaN encoding for
	// negative zero). Max representable value: 448.
	E4M3FN DType = iota
	// E5M2 is the NVIDIA fp8_e5m2 format. Max representable value: 57344.
	E5M2
)

// String returns a human-readable name for the dtype.
func (d DType) String() string {
	switch d {
	case E4M3FN:
		return "fp8_e4m3fn"
	case E5M2:
		return "fp8_e5m2"
	default:
		return "unknown_fp8"
	}
}

// MaxValue returns the largest positive finite value representable in the
// given FP8 format (used to choose the initial scale factor).
func (d DType) MaxValue() float32 {
	switch d {
	case E4M3FN:
		return 448.0
	case E5M2:
		return 57344.0
	default:
		return 0
	}
}

// QuantizeE4M3 converts a float32 value to FP8 E4M3FN bit representation.
// The exponent bias is 7; special values follow the OFP8 spec:
//   - Positive/negative infinity → max finite value (no inf in E4M3FN)
//   - NaN → 0x7F (S=0, E=1111, M=111)
func QuantizeE4M3(v float32) uint8 {
	if math.IsNaN(float64(v)) {
		return 0x7F
	}
	if v == 0 {
		return 0
	}
	sign := uint8(0)
	if v < 0 {
		sign = 0x80
		v = -v
	}
	if math.IsInf(float64(v), 1) {
		// E4M3FN has no Inf; clamp to max finite value.
		return sign | 0x7E // 0 1111 110 = 448
	}

	// Clamp to max representable value.
	const maxE4M3 = float32(448.0)
	if v > maxE4M3 {
		v = maxE4M3
	}

	// IEEE 754 single precision bit layout.
	bits := math.Float32bits(v)
	fpExp := int32((bits>>23)&0xFF) - 127 // unbiased exponent
	fpMan := bits & 0x7FFFFF              // 23-bit mantissa

	// E4M3FN bias is 7; representable exponent range is [-6, 8].
	const bias = 7
	const expMin = -6
	const expMax = 8

	if fpExp < expMin-3 {
		// Below sub-normal range — round to zero.
		return sign
	}

	var e4m3Exp int32
	var e4m3Man uint8

	if fpExp >= expMin {
		// Normal number.
		e4m3Exp = fpExp + bias
		if e4m3Exp > expMax+bias {
			e4m3Exp = expMax + bias
			e4m3Man = 0x6 // max mantissa
		} else {
			e4m3Man = uint8(fpMan >> 20) // top 3 mantissa bits
			// Round to nearest even.
			if (fpMan>>19)&1 == 1 && (fpMan&0x7FFFF != 0 || e4m3Man&1 == 1) {
				e4m3Man++
				if e4m3Man >= 8 {
					e4m3Man = 0
					e4m3Exp++
				}
			}
			if e4m3Exp > expMax+bias {
				e4m3Exp = expMax + bias
				e4m3Man = 0x6
			}
		}
	} else {
		// Sub-normal: shift mantissa.
		shift := uint32(expMin - fpExp)
		sub := (fpMan | 0x800000) >> (shift + 20)
		e4m3Man = uint8(sub & 0x7)
		e4m3Exp = 0
	}

	return sign | (uint8(e4m3Exp) << 3) | e4m3Man
}

// DequantizeE4M3 converts an FP8 E4M3FN bit representation to float32.
func DequantizeE4M3(b uint8) float32 {
	if b == 0x7F || b == 0xFF {
		return float32(math.NaN())
	}
	sign := float32(1.0)
	if b&0x80 != 0 {
		sign = -1.0
	}
	exp := int32((b >> 3) & 0x0F) // 4-bit exponent
	man := int32(b & 0x07)        // 3-bit mantissa

	const bias = 7
	if exp == 0 {
		// Sub-normal
		if man == 0 {
			return sign * 0.0
		}
		return sign * float32(man) * float32(math.Pow(2, float64(1-bias-3)))
	}
	// Normal
	f := float32(1.0+float64(man)/8.0) * float32(math.Pow(2, float64(exp-bias)))
	return sign * f
}

// QuantizeE5M2 converts a float32 value to FP8 E5M2 bit representation.
// Exponent bias is 15; the format supports Inf and NaN.
func QuantizeE5M2(v float32) uint8 {
	if math.IsNaN(float64(v)) {
		return 0x7F
	}
	sign := uint8(0)
	if v < 0 {
		sign = 0x80
		v = -v
	}
	if math.IsInf(float64(v), 1) {
		return sign | 0x7C // 0 11111 00 = Inf in E5M2
	}

	const maxE5M2 = float32(57344.0)
	if v > maxE5M2 {
		v = maxE5M2
	}
	if v == 0 {
		return sign
	}

	bits := math.Float32bits(v)
	fpExp := int32((bits>>23)&0xFF) - 127
	fpMan := bits & 0x7FFFFF

	const bias = 15
	const expMin = -14
	const expMax = 15

	if fpExp < expMin-2 {
		return sign
	}

	var e5m2Exp int32
	var e5m2Man uint8

	if fpExp >= expMin {
		e5m2Exp = fpExp + bias
		if e5m2Exp > expMax+bias {
			e5m2Exp = expMax + bias
			e5m2Man = 0x3
		} else {
			e5m2Man = uint8(fpMan >> 21) // top 2 mantissa bits
			if (fpMan>>20)&1 == 1 && (fpMan&0xFFFFF != 0 || e5m2Man&1 == 1) {
				e5m2Man++
				if e5m2Man >= 4 {
					e5m2Man = 0
					e5m2Exp++
				}
			}
			if e5m2Exp > expMax+bias {
				e5m2Exp = expMax + bias
				e5m2Man = 0x3
			}
		}
	} else {
		shift := uint32(expMin - fpExp)
		sub := (fpMan | 0x800000) >> (shift + 21)
		e5m2Man = uint8(sub & 0x3)
		e5m2Exp = 0
	}

	return sign | (uint8(e5m2Exp) << 2) | e5m2Man
}

// DequantizeE5M2 converts an FP8 E5M2 bit representation to float32.
func DequantizeE5M2(b uint8) float32 {
	sign := float32(1.0)
	if b&0x80 != 0 {
		sign = -1.0
	}
	exp := int32((b >> 2) & 0x1F) // 5-bit exponent
	man := int32(b & 0x03)        // 2-bit mantissa

	const bias = 15
	if exp == 0x1F {
		if man == 0 {
			return sign * float32(math.Inf(1))
		}
		return float32(math.NaN())
	}
	if exp == 0 {
		if man == 0 {
			return sign * 0.0
		}
		return sign * float32(man) * float32(math.Pow(2, float64(1-bias-2)))
	}
	f := float32(1.0+float64(man)/4.0) * float32(math.Pow(2, float64(exp-bias)))
	return sign * f
}
