package core

import (
	"fmt"
	"unsafe"
)

// DType represents the data type of tensor elements.
type DType uint8

const (
	Float16 DType = iota
	Float32
	Float64
	BFloat16
	Int8
	Int16
	Int32
	Int64
	Uint8
	Bool
)

// Size returns the byte size of one element.
func (d DType) Size() uintptr {
	switch d {
	case Float16, BFloat16, Int16:
		return 2
	case Float32, Int32:
		return 4
	case Float64, Int64:
		return 8
	case Int8, Uint8, Bool:
		return 1
	default:
		panic(fmt.Sprintf("unknown dtype: %d", d))
	}
}

func (d DType) String() string {
	names := [...]string{
		"float16", "float32", "float64", "bfloat16",
		"int8", "int16", "int32", "int64", "uint8", "bool",
	}
	if int(d) < len(names) {
		return names[d]
	}
	return fmt.Sprintf("dtype(%d)", d)
}

// IsFloat returns true for floating point types.
func (d DType) IsFloat() bool {
	return d == Float16 || d == Float32 || d == Float64 || d == BFloat16
}

// BFloat16Value represents a bfloat16 number (brain floating point).
// Stored as uint16, upper 16 bits of float32.
type BFloat16Value uint16

func BFloat16FromFloat32(f float32) BFloat16Value {
	bits := *(*uint32)(unsafe.Pointer(&f))
	return BFloat16Value(bits >> 16)
}

func (b BFloat16Value) Float32() float32 {
	bits := uint32(b) << 16
	return *(*float32)(unsafe.Pointer(&bits))
}
