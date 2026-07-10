package tensor

import "github.com/djeday123/goml/core"

// Re-export core types so tensor.Shape, tensor.DType etc. still work.
type Shape = core.Shape
type Strides = core.Strides
type DType = core.DType
type BFloat16Value = core.BFloat16Value

const (
	Float16  = core.Float16
	Float32  = core.Float32
	Float64  = core.Float64
	BFloat16 = core.BFloat16
	Int8     = core.Int8
	Int16    = core.Int16
	Int32    = core.Int32
	Int64    = core.Int64
	Uint8    = core.Uint8
	Bool     = core.Bool
)

var (
	ContiguousStrides   = core.ContiguousStrides
	IsContiguous        = core.IsContiguous
	BroadcastShapes     = core.BroadcastShapes
	FlatIndex           = core.FlatIndex
	Permute             = core.Permute
	BFloat16FromFloat32 = core.BFloat16FromFloat32
)
