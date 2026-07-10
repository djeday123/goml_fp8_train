package cuda

// CUDA backend operations -- all Backend interface methods for compute.
// Delegates to PTX kernels (via launch) and cuBLAS (for MatMul).

import (
	"fmt"
	"unsafe"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
)

// ----------------------------------------------------------------
// Unary ops -- delegate to PTX kernels
// ----------------------------------------------------------------

func (b *Backend) launchUnary(kernel string, dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	if err := b.ensureInit(); err != nil {
		return err
	}
	n := uint32(shape.NumElements())
	dstPtr := devPtr(dst)
	srcPtr := devPtr(src)
	params := []unsafe.Pointer{
		unsafe.Pointer(&dstPtr),
		unsafe.Pointer(&srcPtr),
		unsafe.Pointer(&n),
	}
	return b.launch(kernel, gridSize1D(int(n), 256), 1, 1, 256, 1, 1, params)
}

func (b *Backend) Neg(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.launchUnary("neg_f32", dst, src, shape, dtype)
}

func (b *Backend) Exp(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.launchUnary("exp_f32", dst, src, shape, dtype)
}

func (b *Backend) Silu(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.launchUnary("silu_f32", dst, src, shape, dtype)
}

func (b *Backend) Abs(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.launchUnary("abs_f32", dst, src, shape, dtype)
}

func (b *Backend) Log(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.launchUnary("log_f32", dst, src, shape, dtype)
}

func (b *Backend) Sqrt(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.launchUnary("sqrt_f32", dst, src, shape, dtype)
}

func (b *Backend) Tanh(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.launchUnary("tanh_f32", dst, src, shape, dtype)
}

func (b *Backend) Relu(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.launchUnary("relu_f32", dst, src, shape, dtype)
}

func (b *Backend) Gelu(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.launchUnary("gelu_f32", dst, src, shape, dtype)
}

func (b *Backend) Sigmoid(dst, src backend.Storage, shape core.Shape, dtype core.DType) error {
	return b.launchUnary("sigmoid_f32", dst, src, shape, dtype)
}

// ----------------------------------------------------------------
// Binary ops -- PTX kernels (flat element-wise, no broadcasting yet)
// ----------------------------------------------------------------

func (b *Backend) launchBinary(kernel string, dst, a, bStore backend.Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error {
	if err := b.ensureInit(); err != nil {
		return err
	}
	// TODO: broadcasting support (shapeA != shapeB)
	// For now: require same shape
	n := uint32(shapeOut.NumElements())
	dstPtr := devPtr(dst)
	aPtr := devPtr(a)
	bPtr := devPtr(bStore)
	params := []unsafe.Pointer{
		unsafe.Pointer(&dstPtr),
		unsafe.Pointer(&aPtr),
		unsafe.Pointer(&bPtr),
		unsafe.Pointer(&n),
	}
	return b.launch(kernel, gridSize1D(int(n), 256), 1, 1, 256, 1, 1, params)
}

func (b *Backend) Add(dst, a, bStore backend.Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error {
	return b.launchBinary("add_f32", dst, a, bStore, shapeA, shapeB, shapeOut, dtype)
}

func (b *Backend) Sub(dst, a, bStore backend.Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error {
	return b.launchBinary("sub_f32", dst, a, bStore, shapeA, shapeB, shapeOut, dtype)
}

func (b *Backend) Mul(dst, a, bStore backend.Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error {
	return b.launchBinary("mul_f32", dst, a, bStore, shapeA, shapeB, shapeOut, dtype)
}

func (b *Backend) Div(dst, a, bStore backend.Storage, shapeA, shapeB, shapeOut core.Shape, dtype core.DType) error {
	return b.launchBinary("div_f32", dst, a, bStore, shapeA, shapeB, shapeOut, dtype)
}

// ----------------------------------------------------------------
// Reduction ops
// ----------------------------------------------------------------

func (b *Backend) Sum(dst, src backend.Storage, shape core.Shape, axes []int, keepDim bool, dtype core.DType) error {
	return b.launchReduce("sum_reduce_f32", dst, src, shape, axes, dtype)
}

func (b *Backend) Max(dst, src backend.Storage, shape core.Shape, axes []int, keepDim bool, dtype core.DType) error {
	return b.launchReduce("max_reduce_f32", dst, src, shape, axes, dtype)
}

func (b *Backend) Mean(dst, src backend.Storage, shape core.Shape, axes []int, keepDim bool, dtype core.DType) error {
	// Mean = Sum / count
	if err := b.launchReduce("sum_reduce_f32", dst, src, shape, axes, dtype); err != nil {
		return err
	}
	// Compute element count along reduction axes
	count := 1
	for _, ax := range axes {
		if ax < 0 {
			ax = len(shape) + ax
		}
		count *= shape[ax]
	}
	// Compute output size
	ndst := 1
	for i := range shape {
		isReduced := false
		for _, ax := range axes {
			a := ax
			if a < 0 {
				a = len(shape) + a
			}
			if i == a {
				isReduced = true
			}
		}
		if !isReduced {
			ndst *= shape[i]
		}
	}
	invCount := float32(1.0 / float64(count))
	return b.scaleInPlace(dst, ndst, invCount)
}

// launchReduce reshapes tensor for per-row reduction and launches kernel.
func (b *Backend) launchReduce(kernel string, dst, src backend.Storage, shape core.Shape, axes []int, dtype core.DType) error {
	if err := b.ensureInit(); err != nil {
		return err
	}

	ndim := len(shape)

	// Normalize negative axes
	normAxes := make([]int, len(axes))
	for i, ax := range axes {
		if ax < 0 {
			ax = ndim + ax
		}
		normAxes[i] = ax
	}

	// Compute row_size (product of reduced dims) and num_rows (product of kept dims)
	rowSize := uint32(1)
	numRows := uint32(1)
	for i, s := range shape {
		isReduced := false
		for _, ax := range normAxes {
			if i == ax {
				isReduced = true
			}
		}
		if isReduced {
			rowSize *= uint32(s)
		} else {
			numRows *= uint32(s)
		}
	}

	dstPtr := devPtr(dst)
	srcPtr := devPtr(src)
	params := []unsafe.Pointer{
		unsafe.Pointer(&dstPtr),
		unsafe.Pointer(&srcPtr),
		unsafe.Pointer(&rowSize),
		unsafe.Pointer(&numRows),
	}
	return b.launch(kernel, numRows, 1, 1, 256, 1, 1, params)
}

// scaleInPlace multiplies n elements by scalar. Uses mul_f32 with a filled buffer.
func (b *Backend) scaleInPlace(s backend.Storage, n int, scale float32) error {
	tmp, err := b.pool.Get(n * 4)
	if err != nil {
		return err
	}
	defer b.pool.Put(tmp)

	nu := uint32(n)
	tmpPtr := devPtr(tmp)
	params := []unsafe.Pointer{
		unsafe.Pointer(&tmpPtr),
		unsafe.Pointer(&nu),
		unsafe.Pointer(&scale),
	}
	if err := b.launch("fill_f32", gridSize1D(n, 256), 1, 1, 256, 1, 1, params); err != nil {
		return err
	}

	dstPtr := devPtr(s)
	params2 := []unsafe.Pointer{
		unsafe.Pointer(&dstPtr),
		unsafe.Pointer(&dstPtr),
		unsafe.Pointer(&tmpPtr),
		unsafe.Pointer(&nu),
	}
	return b.launch("mul_f32", gridSize1D(n, 256), 1, 1, 256, 1, 1, params2)
}

// ----------------------------------------------------------------
// MatMul -- cuBLAS
// ----------------------------------------------------------------

func (b *Backend) MatMul(dst, a, bStore backend.Storage, shapeA, shapeB core.Shape, dtype core.DType) error {
	if err := b.ensureInit(); err != nil {
		return err
	}
	if dtype != core.Float32 {
		return fmt.Errorf("CUDA MatMul: only float32 supported currently, got %s", dtype)
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

	if batchSize == 1 {
		return b.cublas.MatMulF32(devPtr(dst), devPtr(a), devPtr(bStore), M, K, N)
	}
	return b.cublas.BatchedMatMulF32(devPtr(dst), devPtr(a), devPtr(bStore), batchSize, M, K, N)
}

// ----------------------------------------------------------------
// Softmax
// ----------------------------------------------------------------

func (b *Backend) Softmax(dst, src backend.Storage, shape core.Shape, axis int, dtype core.DType) error {
	if err := b.ensureInit(); err != nil {
		return err
	}

	ndim := len(shape)
	if axis < 0 {
		axis = ndim + axis
	}

	innerSize := uint32(shape[axis])
	outerSize := uint32(1)
	for i := 0; i < axis; i++ {
		outerSize *= uint32(shape[i])
	}
	for i := axis + 1; i < ndim; i++ {
		outerSize *= uint32(shape[i])
	}

	dstPtr := devPtr(dst)
	srcPtr := devPtr(src)
	params := []unsafe.Pointer{
		unsafe.Pointer(&dstPtr),
		unsafe.Pointer(&srcPtr),
		unsafe.Pointer(&innerSize),
		unsafe.Pointer(&outerSize),
	}
	return b.launch("softmax_f32", outerSize, 1, 1, 256, 1, 1, params)
}

// ----------------------------------------------------------------
// LayerNorm
// ----------------------------------------------------------------

func (b *Backend) LayerNorm(dst, src, gamma, beta backend.Storage, shape core.Shape, normAxis int, eps float64, dtype core.DType) error {
	if err := b.ensureInit(); err != nil {
		return err
	}

	ndim := len(shape)
	if normAxis < 0 {
		normAxis = ndim + normAxis
	}

	normSize := uint32(1)
	for i := normAxis; i < ndim; i++ {
		normSize *= uint32(shape[i])
	}
	numRows := uint32(1)
	for i := 0; i < normAxis; i++ {
		numRows *= uint32(shape[i])
	}

	epsF32 := float32(eps)
	dstPtr := devPtr(dst)
	srcPtr := devPtr(src)
	gammaPtr := devPtr(gamma)
	betaPtr := devPtr(beta)

	params := []unsafe.Pointer{
		unsafe.Pointer(&dstPtr),
		unsafe.Pointer(&srcPtr),
		unsafe.Pointer(&gammaPtr),
		unsafe.Pointer(&betaPtr),
		unsafe.Pointer(&normSize),
		unsafe.Pointer(&numRows),
		unsafe.Pointer(&epsF32),
	}
	return b.launch("layernorm_f32", numRows, 1, 1, 256, 1, 1, params)
}

// ----------------------------------------------------------------
// Embedding
// ----------------------------------------------------------------

func (b *Backend) Embedding(dst, weight, indices backend.Storage, vocabSize, embedDim, seqLen int, dtype core.DType) error {
	if err := b.ensureInit(); err != nil {
		return err
	}

	dim := uint32(embedDim)
	dstPtr := devPtr(dst)
	wPtr := devPtr(weight)
	iPtr := devPtr(indices)
	params := []unsafe.Pointer{
		unsafe.Pointer(&dstPtr),
		unsafe.Pointer(&wPtr),
		unsafe.Pointer(&iPtr),
		unsafe.Pointer(&dim),
	}

	blockDim := uint32(256)
	if uint32(embedDim) < blockDim {
		blockDim = uint32(embedDim)
	}
	return b.launch("embedding_f32", uint32(seqLen), 1, 1, blockDim, 1, 1, params)
}

// ----------------------------------------------------------------
// RoPE
// ----------------------------------------------------------------

func (b *Backend) RoPE(dst, src backend.Storage, shape core.Shape, headDim int, base float64, dtype core.DType) error {
	if err := b.ensureInit(); err != nil {
		return err
	}

	// shape: [batch, heads, seq, headDim]
	batch := shape[0]
	heads := shape[1]
	seqLen := shape[2]

	seqLenU32 := uint32(seqLen)
	headDimU32 := uint32(headDim)
	numHeadsU32 := uint32(heads)
	baseF32 := float32(base)
	dstPtr := devPtr(dst)
	srcPtr := devPtr(src)

	params := []unsafe.Pointer{
		unsafe.Pointer(&dstPtr),
		unsafe.Pointer(&srcPtr),
		unsafe.Pointer(&seqLenU32),
		unsafe.Pointer(&headDimU32),
		unsafe.Pointer(&numHeadsU32),
		unsafe.Pointer(&baseF32),
	}

	gridX := uint32(batch * heads * seqLen)
	halfDim := headDim / 2
	blockX := uint32(256)
	if uint32(halfDim) < blockX {
		blockX = uint32(halfDim)
	}

	return b.launch("rope_f32", gridX, 1, 1, blockX, 1, 1, params)
}

// ----------------------------------------------------------------
// ScaledDotProductAttention -- composed from MatMul + Softmax
// ----------------------------------------------------------------

func (b *Backend) ScaledDotProductAttention(
	dst, q, k, v backend.Storage,
	batchSize, numHeads, seqLen, headDim int,
	causal bool, dtype core.DType,
) error {
	if err := b.ensureInit(); err != nil {
		return err
	}

	BH := batchSize * numHeads
	scale := float32InvSqrt(float32(headDim))

	// Alloc temp: scores [BH, seqLen, seqLen]
	scoreBytes := BH * seqLen * seqLen * 4
	scores, err := b.pool.Get(scoreBytes)
	if err != nil {
		return err
	}
	defer b.pool.Put(scores)

	// scores = Q @ K^T (batched, with scaling folded into alpha)
	alpha := scale
	beta := float32(0.0)
	for i := 0; i < BH; i++ {
		qOff := uintptr(i * seqLen * headDim * 4)
		kOff := uintptr(i * seqLen * headDim * 4)
		sOff := uintptr(i * seqLen * seqLen * 4)
		status := cublasSgemm_v2(
			b.cublas.handle,
			CUBLAS_OP_T, CUBLAS_OP_N,
			int32(seqLen), int32(seqLen), int32(headDim),
			unsafe.Pointer(&alpha),
			devPtr(k)+kOff, int32(headDim),
			devPtr(q)+qOff, int32(headDim),
			unsafe.Pointer(&beta),
			scores.DevicePtr()+sOff, int32(seqLen),
		)
		if status != CUBLAS_STATUS_SUCCESS {
			return fmt.Errorf("attention Q@K^T batch %d: %s", i, status.Error())
		}
	}

	// Causal mask: set scores[i][j] = -1e9 where j > i
	if causal && seqLen > 1 {
		maskBytes := seqLen * seqLen * 4
		mask, err2 := b.pool.Get(maskBytes)
		if err2 != nil {
			return err2
		}
		defer b.pool.Put(mask)

		// Build mask on host: 1.0 for j<=i, 0.0 for j>i
		hostMask := make([]byte, maskBytes)
		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				val := float32(1.0)
				if j > i {
					val = 0.0
				}
				off := (i*seqLen + j) * 4
				*(*float32)(unsafe.Pointer(&hostMask[off])) = val
			}
		}
		if err := CopyHtoD(mask, hostMask); err != nil {
			return err
		}

		// Fill -inf buffer
		negInf := float32(-1e9)
		negInfBuf, err2 := b.pool.Get(seqLen * seqLen * 4)
		if err2 != nil {
			return err2
		}
		defer b.pool.Put(negInfBuf)

		nn := uint32(seqLen * seqLen)
		negInfPtr := negInfBuf.DevicePtr()
		fillParams := []unsafe.Pointer{
			unsafe.Pointer(&negInfPtr),
			unsafe.Pointer(&nn),
			unsafe.Pointer(&negInf),
		}
		b.launch("fill_f32", gridSize1D(int(nn), 256), 1, 1, 256, 1, 1, fillParams)

		maskPtr := mask.DevicePtr()
		for i := 0; i < BH; i++ {
			sOff := uintptr(i * seqLen * seqLen * 4)
			sPtr := scores.DevicePtr() + sOff
			whereParams := []unsafe.Pointer{
				unsafe.Pointer(&sPtr),
				unsafe.Pointer(&maskPtr),
				unsafe.Pointer(&sPtr),
				unsafe.Pointer(&negInfPtr),
				unsafe.Pointer(&nn),
			}
			if err := b.launch("where_f32", gridSize1D(int(nn), 256), 1, 1, 256, 1, 1, whereParams); err != nil {
				return err
			}
		}
	}

	// Softmax along last axis
	rowSize := uint32(seqLen)
	numRows := uint32(BH * seqLen)
	sPtr := scores.DevicePtr()
	smParams := []unsafe.Pointer{
		unsafe.Pointer(&sPtr),
		unsafe.Pointer(&sPtr), // in-place
		unsafe.Pointer(&rowSize),
		unsafe.Pointer(&numRows),
	}
	if err := b.launch("softmax_f32", numRows, 1, 1, 256, 1, 1, smParams); err != nil {
		return err
	}

	// dst = scores @ V (batched)
	alpha = float32(1.0)
	for i := 0; i < BH; i++ {
		sOff := uintptr(i * seqLen * seqLen * 4)
		vOff := uintptr(i * seqLen * headDim * 4)
		dOff := uintptr(i * seqLen * headDim * 4)
		status := cublasSgemm_v2(
			b.cublas.handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			int32(headDim), int32(seqLen), int32(seqLen),
			unsafe.Pointer(&alpha),
			devPtr(v)+vOff, int32(headDim),
			scores.DevicePtr()+sOff, int32(seqLen),
			unsafe.Pointer(&beta),
			devPtr(dst)+dOff, int32(headDim),
		)
		if status != CUBLAS_STATUS_SUCCESS {
			return fmt.Errorf("attention scores@V batch %d: %s", i, status.Error())
		}
	}

	return nil
}

// float32InvSqrt returns 1/sqrt(x) using fast inverse sqrt (Quake III style).
func float32InvSqrt(x float32) float32 {
	i := *(*uint32)(unsafe.Pointer(&x))
	i = 0x5f3759df - (i >> 1)
	y := *(*float32)(unsafe.Pointer(&i))
	y = y * (1.5 - (x*0.5)*y*y) // one Newton iteration
	return y
}

// ----------------------------------------------------------------
// Fill / Arange / Where
// ----------------------------------------------------------------

func (b *Backend) Fill(dst backend.Storage, shape core.Shape, value float64, dtype core.DType) error {
	if err := b.ensureInit(); err != nil {
		return err
	}

	n := uint32(shape.NumElements())
	val := float32(value)
	dstPtr := devPtr(dst)
	params := []unsafe.Pointer{
		unsafe.Pointer(&dstPtr),
		unsafe.Pointer(&n),
		unsafe.Pointer(&val),
	}
	return b.launch("fill_f32", gridSize1D(int(n), 256), 1, 1, 256, 1, 1, params)
}

func (b *Backend) Arange(dst backend.Storage, start, step float64, n int, dtype core.DType) error {
	if err := b.ensureInit(); err != nil {
		return err
	}
	startF32 := float32(start)
	stepF32 := float32(step)
	nu := uint32(n)
	dstPtr := devPtr(dst)
	params := []unsafe.Pointer{
		unsafe.Pointer(&dstPtr),
		unsafe.Pointer(&startF32),
		unsafe.Pointer(&stepF32),
		unsafe.Pointer(&nu),
	}
	return b.launch("arange_f32", gridSize1D(n, 256), 1, 1, 256, 1, 1, params)
}

func (b *Backend) Where(dst, cond, a, bStore backend.Storage, shape core.Shape, dtype core.DType) error {
	if err := b.ensureInit(); err != nil {
		return err
	}
	n := uint32(shape.NumElements())
	dstPtr := devPtr(dst)
	condPtr := devPtr(cond)
	aPtr := devPtr(a)
	bPtr := devPtr(bStore)
	params := []unsafe.Pointer{
		unsafe.Pointer(&dstPtr),
		unsafe.Pointer(&condPtr),
		unsafe.Pointer(&aPtr),
		unsafe.Pointer(&bPtr),
		unsafe.Pointer(&n),
	}
	return b.launch("where_f32", gridSize1D(int(n), 256), 1, 1, 256, 1, 1, params)
}
