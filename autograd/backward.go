package autograd

import (
	"github.com/djeday123/goml/tensor"
)

// Backward computes gradients for all leaf tensors that require grad.
// loss must be a scalar tensor (1 element).
func Backward(loss *tensor.Tensor) error {
	if loss.NumElements() != 1 {
		panic("backward requires scalar loss")
	}

	// Initialize grad of loss as 1.0
	onesGrad, err := tensor.Ones(loss.Shape(), loss.DType(), loss.Device())
	if err != nil {
		return err
	}

	// Topological sort (reverse)
	visited := make(map[*tensor.Tensor]bool)
	var order []*tensor.Tensor
	var topoSort func(t *tensor.Tensor)
	topoSort = func(t *tensor.Tensor) {
		if visited[t] {
			return
		}
		visited[t] = true
		if t.GradFn() != nil {
			for _, input := range t.GradFn().Inputs() {
				topoSort(input)
			}
		}
		order = append(order, t)
	}
	topoSort(loss)

	// Assign initial gradient
	gradMap := make(map[*tensor.Tensor]*tensor.Tensor)
	gradMap[loss] = onesGrad

	// Backward pass in reverse topological order
	for i := len(order) - 1; i >= 0; i-- {
		t := order[i]
		grad, ok := gradMap[t]
		if !ok || t.GradFn() == nil {
			continue
		}

		inputGrads := t.GradFn().Backward(grad)
		inputs := t.GradFn().Inputs()

		for j, input := range inputs {
			if j >= len(inputGrads) || inputGrads[j] == nil {
				continue
			}
			if existing, ok := gradMap[input]; ok {
				// Accumulate gradients
				accumulated, err := AccumulateGrad(existing, inputGrads[j])
				if err != nil {
					return err
				}
				gradMap[input] = accumulated
			} else {
				gradMap[input] = inputGrads[j]
			}
		}
	}

	// Assign gradients to leaf tensors
	for t, grad := range gradMap {
		if t.IsLeaf() && t.RequiresGrad() {
			setGrad(t, grad)
		}
	}

	return nil
}

// setGrad assigns the gradient to a tensor using the exported method.
func setGrad(t *tensor.Tensor, grad *tensor.Tensor) {
	t.SetGrad(grad)
}

// AccumulateGrad adds two gradient tensors element-wise.
func AccumulateGrad(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	// Use the ops package to add - but to avoid circular imports,
	// we do it at the storage level directly
	return AddTensors(a, b)
}
