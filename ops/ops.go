package ops

import (
	"fmt"
	"math"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/tensor"
)

// ---- Autograd function implementations ----

type addGradFn struct {
	a, b *tensor.Tensor
}

func (f *addGradFn) Name() string             { return "AddBackward" }
func (f *addGradFn) Inputs() []*tensor.Tensor { return []*tensor.Tensor{f.a, f.b} }
func (f *addGradFn) Backward(grad *tensor.Tensor) []*tensor.Tensor {
	// d(a+b)/da = 1, d(a+b)/db = 1
	// TODO: handle broadcasting reduction
	return []*tensor.Tensor{grad, grad}
}

type mulGradFn struct {
	a, b *tensor.Tensor
}

func (f *mulGradFn) Name() string             { return "MulBackward" }
func (f *mulGradFn) Inputs() []*tensor.Tensor { return []*tensor.Tensor{f.a, f.b} }
func (f *mulGradFn) Backward(grad *tensor.Tensor) []*tensor.Tensor {
	// d(a*b)/da = b, d(a*b)/db = a
	gradA, _ := Mul(grad, f.b)
	gradB, _ := Mul(grad, f.a)
	return []*tensor.Tensor{gradA, gradB}
}

type matmulGradFn struct {
	a, b *tensor.Tensor
}

func (f *matmulGradFn) Name() string             { return "MatMulBackward" }
func (f *matmulGradFn) Inputs() []*tensor.Tensor { return []*tensor.Tensor{f.a, f.b} }
func (f *matmulGradFn) Backward(grad *tensor.Tensor) []*tensor.Tensor {
	// d(A@B)/dA = grad @ B^T
	// d(A@B)/dB = A^T @ grad
	bT, _ := f.b.T()
	aT, _ := f.a.T()
	gradA, _ := MatMul(grad, bT)
	gradB, _ := MatMul(aT, grad)
	return []*tensor.Tensor{gradA, gradB}
}

type reluGradFn struct {
	input *tensor.Tensor
}

func (f *reluGradFn) Name() string             { return "ReluBackward" }
func (f *reluGradFn) Inputs() []*tensor.Tensor { return []*tensor.Tensor{f.input} }
func (f *reluGradFn) Backward(grad *tensor.Tensor) []*tensor.Tensor {
	// d(relu)/dx = 1 if x > 0 else 0
	inputData := f.input.ToFloat32Slice()
	gradData := grad.ToFloat32Slice()
	out := make([]float32, len(gradData))
	for i := range out {
		if inputData[i] > 0 {
			out[i] = gradData[i]
		}
	}
	result, _ := tensor.FromSlice(out, grad.Shape())
	return []*tensor.Tensor{result}
}

// ---- Public API ----

func getBackend(t *tensor.Tensor) (backend.Backend, error) {
	return backend.GetForDevice(t.Device())
}

func allocOutput(shape tensor.Shape, dtype tensor.DType, device backend.Device) (backend.Storage, error) {
	bk, err := backend.GetForDevice(device)
	if err != nil {
		return nil, err
	}
	return bk.Alloc(shape.NumElements() * int(dtype.Size()))
}

func needsGrad(tensors ...*tensor.Tensor) bool {
	for _, t := range tensors {
		if t.RequiresGrad() {
			return true
		}
	}
	return false
}

// Add performs element-wise addition.
func Add(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	bk, err := getBackend(a)
	if err != nil {
		return nil, err
	}

	outShape, err := tensor.BroadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}

	store, err := allocOutput(outShape, a.DType(), a.Device())
	if err != nil {
		return nil, err
	}

	if err := bk.Add(store, a.Storage(), b.Storage(), a.Shape(), b.Shape(), outShape, a.DType()); err != nil {
		return nil, err
	}

	out := tensor.NewTensor(store, outShape, a.DType())
	if needsGrad(a, b) {
		out.SetRequiresGrad(true)
		out.SetGradFn(&addGradFn{a: a, b: b})
	}
	return out, nil
}

// Mul performs element-wise multiplication.
func Mul(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	bk, err := getBackend(a)
	if err != nil {
		return nil, err
	}

	outShape, err := tensor.BroadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}

	store, err := allocOutput(outShape, a.DType(), a.Device())
	if err != nil {
		return nil, err
	}

	if err := bk.Mul(store, a.Storage(), b.Storage(), a.Shape(), b.Shape(), outShape, a.DType()); err != nil {
		return nil, err
	}

	out := tensor.NewTensor(store, outShape, a.DType())
	if needsGrad(a, b) {
		out.SetRequiresGrad(true)
		out.SetGradFn(&mulGradFn{a: a, b: b})
	}
	return out, nil
}

// MatMul performs matrix multiplication.
func MatMul(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	bk, err := getBackend(a)
	if err != nil {
		return nil, err
	}

	// Ensure contiguous layout before sending to backend
	origA, origB := a, b
	if !a.IsContiguous() {
		a, err = a.Contiguous()
		if err != nil {
			return nil, fmt.Errorf("matmul: contiguous A: %w", err)
		}
	}
	if !b.IsContiguous() {
		b, err = b.Contiguous()
		if err != nil {
			return nil, fmt.Errorf("matmul: contiguous B: %w", err)
		}
	}

	shapeA := a.Shape()
	shapeB := b.Shape()
	ndimA := len(shapeA)
	ndimB := len(shapeB)

	M := shapeA[ndimA-2]
	N := shapeB[ndimB-1]

	// Output shape: batch dims + [M, N]
	outShape := make(tensor.Shape, ndimA)
	copy(outShape, shapeA[:ndimA-2])
	outShape[ndimA-2] = M
	outShape[ndimA-1] = N

	store, err := allocOutput(outShape, a.DType(), a.Device())
	if err != nil {
		return nil, err
	}

	if err := bk.MatMul(store, a.Storage(), b.Storage(), shapeA, shapeB, a.DType()); err != nil {
		return nil, err
	}

	out := tensor.NewTensor(store, outShape, a.DType())
	if needsGrad(origA, origB) {
		out.SetRequiresGrad(true)
		out.SetGradFn(&matmulGradFn{a: origA, b: origB})
	}
	return out, nil
}

// Relu applies rectified linear unit.
func Relu(t *tensor.Tensor) (*tensor.Tensor, error) {
	bk, err := getBackend(t)
	if err != nil {
		return nil, err
	}

	store, err := allocOutput(t.Shape(), t.DType(), t.Device())
	if err != nil {
		return nil, err
	}

	if err := bk.Relu(store, t.Storage(), t.Shape(), t.DType()); err != nil {
		return nil, err
	}

	out := tensor.NewTensor(store, t.Shape(), t.DType())
	if needsGrad(t) {
		out.SetRequiresGrad(true)
		out.SetGradFn(&reluGradFn{input: t})
	}
	return out, nil
}

// softmaxGradFn implements backward for softmax along axis.
// dx = s * (g - sum(g*s, axis=axis, keepdim=true))
type softmaxGradFn struct {
	out  *tensor.Tensor // saved softmax output
	axis int
}

func (f *softmaxGradFn) Name() string             { return "SoftmaxBackward" }
func (f *softmaxGradFn) Inputs() []*tensor.Tensor { return []*tensor.Tensor{f.out} }
func (f *softmaxGradFn) Backward(grad *tensor.Tensor) []*tensor.Tensor {
	shape := grad.Shape()
	rank := len(shape)
	axis := f.axis
	if axis < 0 {
		axis += rank
	}
	gData := grad.ToFloat32Slice()
	sData := f.out.ToFloat32Slice()

	// Compute strides for flat indexing along `axis`.
	innerSize := 1
	for i := axis + 1; i < rank; i++ {
		innerSize *= shape[i]
	}
	axisSize := shape[axis]
	outerSize := len(gData) / (axisSize * innerSize)

	dx := make([]float32, len(gData))
	for o := 0; o < outerSize; o++ {
		for in := 0; in < innerSize; in++ {
			// First pass: dot = sum_j(g_j * s_j) along axis
			dot := float32(0)
			for j := 0; j < axisSize; j++ {
				idx := (o*axisSize+j)*innerSize + in
				dot += gData[idx] * sData[idx]
			}
			// Second pass: dx_j = s_j * (g_j - dot)
			for j := 0; j < axisSize; j++ {
				idx := (o*axisSize+j)*innerSize + in
				dx[idx] = sData[idx] * (gData[idx] - dot)
			}
		}
	}
	result, _ := tensor.FromSlice(dx, shape)
	return []*tensor.Tensor{result}
}

// Softmax applies softmax along the given axis.
func Softmax(t *tensor.Tensor, axis int) (*tensor.Tensor, error) {
	bk, err := getBackend(t)
	if err != nil {
		return nil, err
	}

	store, err := allocOutput(t.Shape(), t.DType(), t.Device())
	if err != nil {
		return nil, err
	}

	if err := bk.Softmax(store, t.Storage(), t.Shape(), axis, t.DType()); err != nil {
		return nil, err
	}

	out := tensor.NewTensor(store, t.Shape(), t.DType())
	if needsGrad(t) {
		out.SetRequiresGrad(true)
		out.SetGradFn(&softmaxGradFn{out: out, axis: axis})
	}
	return out, nil
}

// LayerNorm applies layer normalization.
func LayerNorm(x, gamma, beta *tensor.Tensor, normAxis int, eps float64) (*tensor.Tensor, error) {
	bk, err := getBackend(x)
	if err != nil {
		return nil, err
	}

	store, err := allocOutput(x.Shape(), x.DType(), x.Device())
	if err != nil {
		return nil, err
	}

	var gs, bs backend.Storage
	if gamma != nil {
		gs = gamma.Storage()
	}
	if beta != nil {
		bs = beta.Storage()
	}

	if err := bk.LayerNorm(store, x.Storage(), gs, bs, x.Shape(), normAxis, eps, x.DType()); err != nil {
		return nil, err
	}

	return tensor.NewTensor(store, x.Shape(), x.DType()), nil
}

// geluGradFn implements backward for GELU (tanh approximation).
// y = 0.5*x*(1 + tanh(u)), where u = sqrt(2/π) * (x + 0.044715*x^3)
// dy/dx = 0.5*(1 + tanh(u)) + 0.5*x * (1 - tanh²(u)) * c * (1 + 3*0.044715*x²)
type geluGradFn struct {
	input *tensor.Tensor
}

func (f *geluGradFn) Name() string             { return "GeluBackward" }
func (f *geluGradFn) Inputs() []*tensor.Tensor { return []*tensor.Tensor{f.input} }
func (f *geluGradFn) Backward(grad *tensor.Tensor) []*tensor.Tensor {
	xData := f.input.ToFloat32Slice()
	gData := grad.ToFloat32Slice()
	const c = 0.7978845608028654 // sqrt(2/π)
	dx := make([]float32, len(gData))
	for i, x := range xData {
		x64 := float64(x)
		u := c * (x64 + 0.044715*x64*x64*x64)
		tu := math.Tanh(u)
		dudx := c * (1 + 3*0.044715*x64*x64)
		dy := 0.5*(1+tu) + 0.5*x64*(1-tu*tu)*dudx
		dx[i] = gData[i] * float32(dy)
	}
	result, _ := tensor.FromSlice(dx, grad.Shape())
	return []*tensor.Tensor{result}
}

// Gelu applies GELU activation.
func Gelu(t *tensor.Tensor) (*tensor.Tensor, error) {
	bk, err := getBackend(t)
	if err != nil {
		return nil, err
	}

	store, err := allocOutput(t.Shape(), t.DType(), t.Device())
	if err != nil {
		return nil, err
	}

	if err := bk.Gelu(store, t.Storage(), t.Shape(), t.DType()); err != nil {
		return nil, err
	}

	out := tensor.NewTensor(store, t.Shape(), t.DType())
	if needsGrad(t) {
		out.SetRequiresGrad(true)
		out.SetGradFn(&geluGradFn{input: t})
	}
	return out, nil
}

// siluGradFn implements backward for SiLU/Swish: y = x*sigmoid(x).
// dy/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
type siluGradFn struct {
	input *tensor.Tensor
}

func (f *siluGradFn) Name() string             { return "SiluBackward" }
func (f *siluGradFn) Inputs() []*tensor.Tensor { return []*tensor.Tensor{f.input} }
func (f *siluGradFn) Backward(grad *tensor.Tensor) []*tensor.Tensor {
	xData := f.input.ToFloat32Slice()
	gData := grad.ToFloat32Slice()
	dx := make([]float32, len(gData))
	for i, x := range xData {
		sig := 1.0 / (1.0 + math.Exp(float64(-x)))
		dy := sig * (1 + float64(x)*(1-sig))
		dx[i] = gData[i] * float32(dy)
	}
	result, _ := tensor.FromSlice(dx, grad.Shape())
	return []*tensor.Tensor{result}
}

// Silu applies SiLU (Swish) activation.
func Silu(t *tensor.Tensor) (*tensor.Tensor, error) {
	bk, err := getBackend(t)
	if err != nil {
		return nil, err
	}

	store, err := allocOutput(t.Shape(), t.DType(), t.Device())
	if err != nil {
		return nil, err
	}

	if err := bk.Silu(store, t.Storage(), t.Shape(), t.DType()); err != nil {
		return nil, err
	}

	out := tensor.NewTensor(store, t.Shape(), t.DType())
	if needsGrad(t) {
		out.SetRequiresGrad(true)
		out.SetGradFn(&siluGradFn{input: t})
	}
	return out, nil
}

// ScaledDotProductAttention computes multi-head attention.
func ScaledDotProductAttention(q, k, v *tensor.Tensor, numHeads int, causal bool) (*tensor.Tensor, error) {
	bk, err := getBackend(q)
	if err != nil {
		return nil, err
	}

	shape := q.Shape()
	batchSize := shape[0]
	seqLen := shape[2]
	headDim := shape[3]

	store, err := allocOutput(shape, q.DType(), q.Device())
	if err != nil {
		return nil, err
	}

	if err := bk.ScaledDotProductAttention(
		store, q.Storage(), k.Storage(), v.Storage(),
		batchSize, numHeads, seqLen, headDim,
		causal, q.DType(),
	); err != nil {
		return nil, err
	}

	return tensor.NewTensor(store, shape, q.DType()), nil
}
