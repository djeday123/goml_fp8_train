package cuda

import (
	"fmt"
	"sync"

	"github.com/ebitengine/purego"
)

// Transformer kernel bindings via purego (no CGo)

var transformer struct {
	once sync.Once
	err  error
	lib  uintptr

	rmsnorm   func(x, weight, y uintptr, rows, hidden int32, eps float32, stream uintptr) int32
	swiglu    func(gate, up, y uintptr, n int32, stream uintptr) int32
	rope      func(x, pos, y uintptr, batch, seqLen, numHeads, headDim int32, thetaBase float32, stream uintptr) int32
	attention func(Q, K, V, O uintptr, batch, numHeads, seqLen, headDim, causal int32, stream uintptr) int32
}

func initTransformer() error {
	transformer.once.Do(func() {
		lib, err := purego.Dlopen(resolveLib("libtransformer.so"), purego.RTLD_LAZY|purego.RTLD_GLOBAL)
		if err != nil {
			transformer.err = fmt.Errorf("transformer: dlopen: %w", err)
			return
		}
		transformer.lib = lib

		purego.RegisterLibFunc(&transformer.rmsnorm, lib, "rmsnorm_forward")
		purego.RegisterLibFunc(&transformer.swiglu, lib, "swiglu_forward")
		purego.RegisterLibFunc(&transformer.rope, lib, "rope_forward")
		purego.RegisterLibFunc(&transformer.attention, lib, "attention_forward")
	})
	return transformer.err
}

// RMSNorm computes y = x * weight / sqrt(mean(x²) + eps)
//
// x:      [rows, hidden] FP16 device ptr
// weight: [hidden]       FP16 device ptr
// y:      [rows, hidden] FP16 device ptr
func RMSNorm(x, weight, y uintptr, rows, hidden int, eps float32, stream uintptr) error {
	if err := initTransformer(); err != nil {
		return err
	}
	rc := transformer.rmsnorm(x, weight, y, int32(rows), int32(hidden), eps, stream)
	if rc != 0 {
		return fmt.Errorf("rmsnorm: CUDA error %d", rc)
	}
	return nil
}

// SwiGLU computes y = SiLU(gate) * up
//
// gate: [n] FP16 device ptr
// up:   [n] FP16 device ptr
// y:    [n] FP16 device ptr
func SwiGLU(gate, up, y uintptr, n int, stream uintptr) error {
	if err := initTransformer(); err != nil {
		return err
	}
	rc := transformer.swiglu(gate, up, y, int32(n), stream)
	if rc != 0 {
		return fmt.Errorf("swiglu: CUDA error %d", rc)
	}
	return nil
}

// RoPE applies rotary positional embeddings.
//
// x:   [batch, seqLen, numHeads, headDim] FP16 device ptr
// pos: [seqLen]                           int32 device ptr
// y:   [batch, seqLen, numHeads, headDim] FP16 device ptr (can alias x)
// thetaBase: 10000.0 (standard) or 500000.0 (extended context)
func RoPE(x, pos, y uintptr, batch, seqLen, numHeads, headDim int, thetaBase float32, stream uintptr) error {
	if err := initTransformer(); err != nil {
		return err
	}
	rc := transformer.rope(x, pos, y, int32(batch), int32(seqLen), int32(numHeads), int32(headDim), thetaBase, stream)
	if rc != 0 {
		return fmt.Errorf("rope: CUDA error %d", rc)
	}
	return nil
}

// Attention computes scaled dot-product attention with optional causal mask.
//
// Q,K,V: [batch, numHeads, seqLen, headDim] FP16 device ptr
// O:     [batch, numHeads, seqLen, headDim] FP16 device ptr
// causal: true for causal (autoregressive) mask
//
// Note: basic O(s²) implementation. For seqLen > 2048, use FlashAttention.
func Attention(Q, K, V, O uintptr, batch, numHeads, seqLen, headDim int, causal bool, stream uintptr) error {
	if err := initTransformer(); err != nil {
		return err
	}
	c := int32(0)
	if causal {
		c = 1
	}
	rc := transformer.attention(Q, K, V, O, int32(batch), int32(numHeads), int32(seqLen), int32(headDim), c, stream)
	if rc != 0 {
		return fmt.Errorf("attention: CUDA error %d", rc)
	}
	return nil
}
