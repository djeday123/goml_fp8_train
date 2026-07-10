package cpu

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
)

// Backend implements backend.Backend for CPU.
type Backend struct{}

func init() {
	backend.Register(&Backend{})
}

func (b *Backend) Name() string                   { return "cpu" }
func (b *Backend) DeviceType() backend.DeviceType { return backend.CPU }

// ---- Memory ----

func (b *Backend) Alloc(byteLen int) (backend.Storage, error) {
	return newStorage(byteLen), nil
}

func (b *Backend) Free(s backend.Storage) {
	s.Free()
}

func (b *Backend) Copy(dst, src backend.Storage, byteLen int) error {
	d := asBytes(dst, byteLen)
	s := asBytes(src, byteLen)
	copy(d, s)
	return nil
}

func (b *Backend) ToDevice(dst backend.Device, src backend.Storage) (backend.Storage, error) {
	if dst.Type != backend.CPU {
		return nil, fmt.Errorf("cpu backend can only transfer to cpu")
	}
	newStore := newStorage(src.ByteLen())
	copy(asBytes(newStore, src.ByteLen()), asBytes(src, src.ByteLen()))
	return newStore, nil
}

// ---- Unary ops ----

func (b *Backend) Neg(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return unaryOp(dst, src, shape, dtype, func(x float32) float32 { return -x })
}

func (b *Backend) Abs(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return unaryOp(dst, src, shape, dtype, func(x float32) float32 {
		if x < 0 {
			return -x
		}
		return x
	})
}

func (b *Backend) Exp(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return unaryOp(dst, src, shape, dtype, func(x float32) float32 {
		return float32(math.Exp(float64(x)))
	})
}

func (b *Backend) Log(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return unaryOp(dst, src, shape, dtype, func(x float32) float32 {
		return float32(math.Log(float64(x)))
	})
}

func (b *Backend) Sqrt(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return unaryOp(dst, src, shape, dtype, func(x float32) float32 {
		return float32(math.Sqrt(float64(x)))
	})
}

func (b *Backend) Tanh(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return unaryOp(dst, src, shape, dtype, func(x float32) float32 {
		return float32(math.Tanh(float64(x)))
	})
}

func (b *Backend) Relu(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return unaryOp(dst, src, shape, dtype, func(x float32) float32 {
		if x > 0 {
			return x
		}
		return 0
	})
}

func (b *Backend) Gelu(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	c := float32(math.Sqrt(2.0 / math.Pi))
	return unaryOp(dst, src, shape, dtype, func(x float32) float32 {
		return 0.5 * x * (1 + float32(math.Tanh(float64(c*(x+0.044715*x*x*x)))))
	})
}

func (b *Backend) Sigmoid(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return unaryOp(dst, src, shape, dtype, func(x float32) float32 {
		return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
	})
}

func (b *Backend) Silu(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return unaryOp(dst, src, shape, dtype, func(x float32) float32 {
		return x / (1.0 + float32(math.Exp(float64(-x))))
	})
}

// ---- Binary ops ----

func (b *Backend) Add(dst, a, bStore backend.Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error {
	return binaryOp(dst, a, bStore, shapeA, shapeB, shapeOut, dtype, func(x, y float32) float32 { return x + y })
}

func (b *Backend) Sub(dst, a, bStore backend.Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error {
	return binaryOp(dst, a, bStore, shapeA, shapeB, shapeOut, dtype, func(x, y float32) float32 { return x - y })
}

func (b *Backend) Mul(dst, a, bStore backend.Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error {
	return binaryOp(dst, a, bStore, shapeA, shapeB, shapeOut, dtype, func(x, y float32) float32 { return x * y })
}

func (b *Backend) Div(dst, a, bStore backend.Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error {
	return binaryOp(dst, a, bStore, shapeA, shapeB, shapeOut, dtype, func(x, y float32) float32 { return x / y })
}

// ---- Reduction ops ----

func (b *Backend) Sum(dst, src backend.Storage, shape core.Shape, axes []int, keepDim bool, dtype core.DType) error {
	return reduceOp(dst, src, shape, axes, keepDim, dtype, 0, func(acc, x float32) float32 { return acc + x })
}

func (b *Backend) Max(dst, src backend.Storage, shape core.Shape, axes []int, keepDim bool, dtype core.DType) error {
	return reduceOp(dst, src, shape, axes, keepDim, dtype, -math.MaxFloat32, func(acc, x float32) float32 {
		if x > acc {
			return x
		}
		return acc
	})
}

func (b *Backend) Mean(dst, src backend.Storage, shape core.Shape, axes []int, keepDim bool, dtype core.DType) error {
	// Mean = Sum / count
	count := 1
	for _, a := range axes {
		count *= shape[a]
	}
	return reduceOp(dst, src, shape, axes, keepDim, dtype, 0, func(acc, x float32) float32 { return acc + x/float32(count) })
}

// ---- MatMul ----

func (b *Backend) MatMul(dst, a, bStore backend.Storage, shapeA, shapeB core.Shape, dtype core.DType) error {
	if dtype != core.Float32 {
		return fmt.Errorf("matmul: only float32 supported on cpu, got %s", dtype)
	}

	ndimA := len(shapeA)
	ndimB := len(shapeB)
	M := shapeA[ndimA-2]
	K := shapeA[ndimA-1]
	N := shapeB[ndimB-1]

	// Compute batch size
	batchSize := 1
	for i := 0; i < ndimA-2; i++ {
		batchSize *= shapeA[i]
	}

	aData := f32Slice(a, batchSize*M*K)
	bData := f32Slice(bStore, batchSize*K*N)
	cData := f32Slice(dst, batchSize*M*N)

	for batch := 0; batch < batchSize; batch++ {
		aOff := batch * M * K
		bOff := batch * K * N
		cOff := batch * M * N
		matmulF32(
			cData[cOff:cOff+M*N],
			aData[aOff:aOff+M*K],
			bData[bOff:bOff+K*N],
			M, K, N,
		)
	}
	return nil
}

// matmulF32 performs C = A @ B with tiling for cache efficiency.
func matmulF32(c, a, b []float32, M, K, N int) {
	// Clear output
	for i := range c {
		c[i] = 0
	}

	const tileSize = 32

	for i0 := 0; i0 < M; i0 += tileSize {
		iEnd := min(i0+tileSize, M)
		for k0 := 0; k0 < K; k0 += tileSize {
			kEnd := min(k0+tileSize, K)
			for j0 := 0; j0 < N; j0 += tileSize {
				jEnd := min(j0+tileSize, N)
				// Micro-kernel: tile multiplication
				for i := i0; i < iEnd; i++ {
					for k := k0; k < kEnd; k++ {
						aik := a[i*K+k]
						for j := j0; j < jEnd; j++ {
							c[i*N+j] += aik * b[k*N+j]
						}
					}
				}
			}
		}
	}
}

// ---- Softmax ----

func (b *Backend) Softmax(dst, src backend.Storage, shape core.Shape, axis int, dtype core.DType) error {
	if dtype != core.Float32 {
		return fmt.Errorf("softmax: only float32 supported")
	}

	n := shape.NumElements()
	srcData := f32Slice(src, n)
	dstData := f32Slice(dst, n)

	axisSize := shape[axis]
	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= shape[i]
	}
	innerSize := 1
	for i := axis + 1; i < len(shape); i++ {
		innerSize *= shape[i]
	}

	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			// Find max for numerical stability
			maxVal := float32(-math.MaxFloat32)
			for a := 0; a < axisSize; a++ {
				idx := outer*axisSize*innerSize + a*innerSize + inner
				if srcData[idx] > maxVal {
					maxVal = srcData[idx]
				}
			}
			// Exp and sum
			sumExp := float32(0)
			for a := 0; a < axisSize; a++ {
				idx := outer*axisSize*innerSize + a*innerSize + inner
				v := float32(math.Exp(float64(srcData[idx] - maxVal)))
				dstData[idx] = v
				sumExp += v
			}
			// Normalize
			for a := 0; a < axisSize; a++ {
				idx := outer*axisSize*innerSize + a*innerSize + inner
				dstData[idx] /= sumExp
			}
		}
	}
	return nil
}

// ---- LayerNorm ----

func (b *Backend) LayerNorm(dst, src, gamma, beta backend.Storage, shape core.Shape, normAxis int, eps float64, dtype core.DType) error {
	if dtype != core.Float32 {
		return fmt.Errorf("layernorm: only float32 supported")
	}

	n := shape.NumElements()
	srcData := f32Slice(src, n)
	dstData := f32Slice(dst, n)

	normSize := 1
	for i := normAxis; i < len(shape); i++ {
		normSize *= shape[i]
	}
	batchSize := n / normSize

	var gammaData, betaData []float32
	if gamma != nil {
		gammaData = f32Slice(gamma, normSize)
	}
	if beta != nil {
		betaData = f32Slice(beta, normSize)
	}

	for batch := 0; batch < batchSize; batch++ {
		off := batch * normSize

		// Compute mean
		mean := float32(0)
		for i := 0; i < normSize; i++ {
			mean += srcData[off+i]
		}
		mean /= float32(normSize)

		// Compute variance
		variance := float32(0)
		for i := 0; i < normSize; i++ {
			d := srcData[off+i] - mean
			variance += d * d
		}
		variance /= float32(normSize)

		invStd := float32(1.0 / math.Sqrt(float64(variance)+eps))

		for i := 0; i < normSize; i++ {
			normalized := (srcData[off+i] - mean) * invStd
			if gammaData != nil {
				normalized *= gammaData[i]
			}
			if betaData != nil {
				normalized += betaData[i]
			}
			dstData[off+i] = normalized
		}
	}
	return nil
}

// ---- Embedding ----

func (b *Backend) Embedding(dst, weight, indices backend.Storage, vocabSize, embedDim, seqLen int, dtype core.DType) error {
	if dtype != core.Float32 {
		return fmt.Errorf("embedding: only float32 supported")
	}

	wData := f32Slice(weight, vocabSize*embedDim)
	iData := i64Slice(indices, seqLen)
	oData := f32Slice(dst, seqLen*embedDim)

	for s := 0; s < seqLen; s++ {
		idx := int(iData[s])
		if idx < 0 || idx >= vocabSize {
			return fmt.Errorf("embedding index %d out of range [0, %d)", idx, vocabSize)
		}
		copy(oData[s*embedDim:(s+1)*embedDim], wData[idx*embedDim:(idx+1)*embedDim])
	}
	return nil
}

// ---- RoPE ----

func (b *Backend) RoPE(dst, src backend.Storage, shape core.Shape, headDim int, base float64, dtype core.DType) error {
	if dtype != core.Float32 {
		return fmt.Errorf("rope: only float32 supported")
	}

	n := shape.NumElements()
	srcData := f32Slice(src, n)
	dstData := f32Slice(dst, n)

	// shape: [batch, heads, seq, headDim]
	batch := shape[0]
	heads := shape[1]
	seqLen := shape[2]

	halfDim := headDim / 2

	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			for pos := 0; pos < seqLen; pos++ {
				off := ((b*heads+h)*seqLen + pos) * headDim
				for i := 0; i < halfDim; i++ {
					freq := 1.0 / math.Pow(base, float64(2*i)/float64(headDim))
					angle := float64(pos) * freq
					cos := float32(math.Cos(angle))
					sin := float32(math.Sin(angle))

					x0 := srcData[off+i]
					x1 := srcData[off+halfDim+i]
					dstData[off+i] = x0*cos - x1*sin
					dstData[off+halfDim+i] = x0*sin + x1*cos
				}
			}
		}
	}
	return nil
}

// ---- Scaled Dot-Product Attention ----

func (b *Backend) ScaledDotProductAttention(
	dst, q, k, v backend.Storage,
	batchSize, numHeads, seqLen, headDim int,
	causal bool, dtype core.DType,
) error {
	if dtype != core.Float32 {
		return fmt.Errorf("attention: only float32 supported")
	}

	total := batchSize * numHeads * seqLen * headDim
	qData := f32Slice(q, total)
	kData := f32Slice(k, total)
	vData := f32Slice(v, total)
	oData := f32Slice(dst, total)

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	scores := make([]float32, seqLen*seqLen)

	for b := 0; b < batchSize; b++ {
		for h := 0; h < numHeads; h++ {
			bhOff := (b*numHeads + h) * seqLen * headDim

			// Q @ K^T -> scores [seqLen, seqLen]
			for i := 0; i < seqLen; i++ {
				for j := 0; j < seqLen; j++ {
					dot := float32(0)
					for d := 0; d < headDim; d++ {
						dot += qData[bhOff+i*headDim+d] * kData[bhOff+j*headDim+d]
					}
					scores[i*seqLen+j] = dot * scale
					if causal && j > i {
						scores[i*seqLen+j] = -1e9
					}
				}
			}

			// Softmax over last dim
			for i := 0; i < seqLen; i++ {
				maxVal := float32(-math.MaxFloat32)
				for j := 0; j < seqLen; j++ {
					if scores[i*seqLen+j] > maxVal {
						maxVal = scores[i*seqLen+j]
					}
				}
				sumExp := float32(0)
				for j := 0; j < seqLen; j++ {
					scores[i*seqLen+j] = float32(math.Exp(float64(scores[i*seqLen+j] - maxVal)))
					sumExp += scores[i*seqLen+j]
				}
				for j := 0; j < seqLen; j++ {
					scores[i*seqLen+j] /= sumExp
				}
			}

			// Attn @ V -> output
			for i := 0; i < seqLen; i++ {
				for d := 0; d < headDim; d++ {
					sum := float32(0)
					for j := 0; j < seqLen; j++ {
						sum += scores[i*seqLen+j] * vData[bhOff+j*headDim+d]
					}
					oData[bhOff+i*headDim+d] = sum
				}
			}
		}
	}
	return nil
}

// ---- Fill ops ----

func (b *Backend) Fill(dst backend.Storage, shape core.Shape, value float64, dtype core.DType) error {
	n := shape.NumElements()
	switch dtype {
	case core.Float32:
		data := f32Slice(dst, n)
		v := float32(value)
		for i := range data {
			data[i] = v
		}
	case core.Float64:
		data := f64Slice(dst, n)
		for i := range data {
			data[i] = value
		}
	case core.Int32:
		data := i32Slice(dst, n)
		v := int32(value)
		for i := range data {
			data[i] = v
		}
	case core.Int64:
		data := i64Slice(dst, n)
		v := int64(value)
		for i := range data {
			data[i] = v
		}
	default:
		return fmt.Errorf("fill: unsupported dtype %s", dtype)
	}
	return nil
}

func (b *Backend) Arange(dst backend.Storage, start, step float64, n int, dtype core.DType) error {
	switch dtype {
	case core.Float32:
		data := f32Slice(dst, n)
		for i := range data {
			data[i] = float32(start + float64(i)*step)
		}
	case core.Int64:
		data := i64Slice(dst, n)
		for i := range data {
			data[i] = int64(start + float64(i)*step)
		}
	default:
		return fmt.Errorf("arange: unsupported dtype %s", dtype)
	}
	return nil
}

// ---- Where ----

func (b *Backend) Where(dst, cond, a, bStore backend.Storage, shape core.Shape, dtype core.DType) error {
	n := shape.NumElements()
	condData := cond.Bytes()[:n]

	switch dtype {
	case core.Float32:
		aData := f32Slice(a, n)
		bData := f32Slice(bStore, n)
		dData := f32Slice(dst, n)
		for i := 0; i < n; i++ {
			if condData[i] != 0 {
				dData[i] = aData[i]
			} else {
				dData[i] = bData[i]
			}
		}
	default:
		return fmt.Errorf("where: unsupported dtype %s", dtype)
	}
	return nil
}

// ---- Helpers ----

func asBytes(s backend.Storage, n int) []byte {
	b := s.Bytes()
	if b != nil {
		return b[:n]
	}
	return unsafe.Slice((*byte)(s.Ptr()), n)
}

func f32Slice(s backend.Storage, n int) []float32 {
	b := s.Bytes()
	if len(b) > 0 {
		return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), n)
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(s.Ptr())), n)
}

func f64Slice(s backend.Storage, n int) []float64 {
	b := s.Bytes()
	if len(b) > 0 {
		return unsafe.Slice((*float64)(unsafe.Pointer(&b[0])), n)
	}
	return unsafe.Slice((*float64)(unsafe.Pointer(s.Ptr())), n)
}

func i32Slice(s backend.Storage, n int) []int32 {
	b := s.Bytes()
	if len(b) > 0 {
		return unsafe.Slice((*int32)(unsafe.Pointer(&b[0])), n)
	}
	return unsafe.Slice((*int32)(unsafe.Pointer(s.Ptr())), n)
}

func i64Slice(s backend.Storage, n int) []int64 {
	b := s.Bytes()
	if len(b) > 0 {
		return unsafe.Slice((*int64)(unsafe.Pointer(&b[0])), n)
	}
	return unsafe.Slice((*int64)(unsafe.Pointer(s.Ptr())), n)
}

// unaryOp applies a scalar function element-wise (float32 only for now).
func unaryOp(dst, src backend.Storage, shape core.Shape, dtype core.DType, fn func(float32) float32) error {
	if dtype != core.Float32 {
		return fmt.Errorf("unary op: only float32 supported, got %s", dtype)
	}
	n := shape.NumElements()
	srcData := f32Slice(src, n)
	dstData := f32Slice(dst, n)
	for i := 0; i < n; i++ {
		dstData[i] = fn(srcData[i])
	}
	return nil
}

// binaryOp applies a binary function element-wise with broadcasting.
func binaryOp(dst, aStore, bStore backend.Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType, fn func(float32, float32) float32) error {
	if dtype != core.Float32 {
		return fmt.Errorf("binary op: only float32 supported, got %s", dtype)
	}

	nOut := shapeOut.NumElements()
	nA := shapeA.NumElements()
	nB := shapeB.NumElements()
	aData := f32Slice(aStore, nA)
	bData := f32Slice(bStore, nB)
	dData := f32Slice(dst, nOut)

	// Fast path: same shape, no broadcasting needed
	if shapeA.Equal(shapeB) {
		for i := 0; i < nOut; i++ {
			dData[i] = fn(aData[i], bData[i])
		}
		return nil
	}

	// General broadcasting
	ndim := len(shapeOut)
	indices := make([]int, ndim)

	for i := 0; i < nOut; i++ {
		// Compute broadcast indices for A and B
		idxA := 0
		idxB := 0
		strideA := 1
		strideB := 1
		for d := ndim - 1; d >= 0; d-- {
			dimA := 1
			dimB := 1
			offA := d - (ndim - len(shapeA))
			offB := d - (ndim - len(shapeB))
			if offA >= 0 {
				dimA = shapeA[offA]
			}
			if offB >= 0 {
				dimB = shapeB[offB]
			}

			aIdx := indices[d]
			bIdx := indices[d]
			if dimA == 1 {
				aIdx = 0
			}
			if dimB == 1 {
				bIdx = 0
			}

			if offA >= 0 {
				idxA += aIdx * strideA
				strideA *= dimA
			}
			if offB >= 0 {
				idxB += bIdx * strideB
				strideB *= dimB
			}
		}

		dData[i] = fn(aData[idxA], bData[idxB])

		// Increment indices
		for d := ndim - 1; d >= 0; d-- {
			indices[d]++
			if indices[d] < shapeOut[d] {
				break
			}
			indices[d] = 0
		}
	}
	return nil
}

// reduceOp performs a reduction along given axes.
func reduceOp(dst, src backend.Storage, shape core.Shape, axes []int, keepDim bool, dtype core.DType, init float32, fn func(float32, float32) float32) error {
	if dtype != core.Float32 {
		return fmt.Errorf("reduce op: only float32 supported, got %s", dtype)
	}

	n := shape.NumElements()
	srcData := f32Slice(src, n)

	// Compute output shape
	outShape := make(core.Shape, 0, len(shape))
	axisSet := make(map[int]bool)
	for _, a := range axes {
		axisSet[a] = true
	}
	for i, d := range shape {
		if axisSet[i] {
			if keepDim {
				outShape = append(outShape, 1)
			}
		} else {
			outShape = append(outShape, d)
		}
	}
	if len(outShape) == 0 {
		outShape = core.Shape{1}
	}

	nOut := outShape.NumElements()
	dstData := f32Slice(dst, nOut)

	// Initialize output
	for i := range dstData {
		dstData[i] = init
	}

	// Iterate over all source elements
	ndim := len(shape)
	indices := make([]int, ndim)

	for i := 0; i < n; i++ {
		// Compute output index
		outIdx := 0
		outStride := 1
		for d := len(outShape) - 1; d >= 0; d-- {
			// Map source dim to output dim
			srcDim := d
			if !keepDim {
				// Count how many reduced axes are before this output dim
				skip := 0
				for _, a := range axes {
					if a <= d+skip {
						skip++
					}
				}
				srcDim = d + skip
			}
			idx := indices[srcDim]
			if axisSet[srcDim] {
				idx = 0
			}
			outIdx += idx * outStride
			outStride *= outShape[d]
		}

		dstData[outIdx] = fn(dstData[outIdx], srcData[i])

		// Increment indices
		for d := ndim - 1; d >= 0; d-- {
			indices[d]++
			if indices[d] < shape[d] {
				break
			}
			indices[d] = 0
		}
	}

	return nil
}
