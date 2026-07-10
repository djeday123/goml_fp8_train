/*
 * libfa_sm120 — FlashAttention FP8 forward для NVIDIA sm_120a (Blackwell consumer).
 *
 * Production champion: v121r (mean 647T bh=64 sl=8192, mean 652T bh=128 sl=8192).
 * Disp: hd=64 (v89/v80b), hd=128 (v121r peak, v121 window, v122 wave-tail,
 *       v118 mid-grid, v117b small-batch wnd, v96b boundary).
 *
 * ABI: C linkage, opaque context, error codes (no exceptions, no printf).
 * Thread-safe across distinct contexts. Single-GPU.
 *
 * Все указатели — device memory (CUDA). Inputs Q,K,V e4m3 (uint8), output O fp16.
 * Layout: tightly packed [batch_heads, seq_len, head_dim] row-major.
 *
 * Supported: hd ∈ {64, 128}; window ∈ {0..seq_len}; causal ∈ {0, 1}.
 * NOT supported (yet): backward, hd ∉ {64,128}, sparse, non-dense strides,
 *                      multi-GPU, mixed-precision other than e4m3/fp16.
 */
#ifndef FA_SM120_H_
#define FA_SM120_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
  #define FA_API __attribute__((visibility("default")))
#else
  #define FA_API
#endif

typedef struct fa_ctx fa_ctx_t;

typedef enum {
    FA_OK                       = 0,
    FA_ERR_INVALID_ARG          = 1,
    FA_ERR_UNSUPPORTED_ARCH     = 2,   /* GPU is not sm_120a */
    FA_ERR_UNSUPPORTED_HD       = 3,   /* head_dim not in {64, 128} */
    FA_ERR_UNSUPPORTED_SHAPE    = 4,   /* seq_len < head_dim, etc. */
    FA_ERR_CUDA                 = 5,   /* CUDA runtime error (see fa_last_cuda_error) */
    FA_ERR_OOM                  = 6,
    FA_ERR_INTERNAL             = 7
} fa_status_t;

/* CUDA stream type — forward-declared to avoid <cuda_runtime.h> in client. */
typedef struct CUstream_st* fa_stream_t;

/*
 * fa_create(): probe device, verify sm_120a, prepare dispatcher state.
 *   *ctx_out: receives opaque pointer. NULL on failure.
 * Returns FA_ERR_UNSUPPORTED_ARCH on non-sm_120a cards (consumer should fallback).
 */
FA_API fa_status_t fa_create(fa_ctx_t** ctx_out);

/*
 * fa_forward(): single FA forward call.
 *   q, k, v       — device ptrs, FP8 e4m3 (uint8). Shape [BH, S, HD], row-major.
 *   o             — device ptr, FP16 output. Shape [BH, S, HD].
 *   batch_heads   — flattened B*H (we don't track batch/heads separately).
 *   seq_len       — sequence length S.
 *   head_dim      — HD ∈ {64, 128}.
 *   causal        — 0=no mask, 1=causal upper-triangular mask.
 *   window        — 0=no window; >0 = sliding window length (each Q sees K in [q-window+1, q]).
 *   scale         — softmax pre-scale (typically 1/sqrt(HD)).
 *   stream        — CUDA stream (0 = default).
 *
 * Combinations:
 *   causal=0 window=0  → full attention
 *   causal=1 window=0  → causal mask
 *   causal=0 window>0  → bidirectional sliding window (rare)
 *   causal=1 window>0  → causal sliding window
 */
FA_API fa_status_t fa_forward(fa_ctx_t* ctx,
                       const void* q, const void* k, const void* v,
                       void* o,
                       int batch_heads, int seq_len, int head_dim,
                       int causal, int window,
                       float scale, fa_stream_t stream);

/*
 * fa_destroy(): release all internal state. Returns FA_OK even on null ctx.
 */
FA_API fa_status_t fa_destroy(fa_ctx_t* ctx);

/*
 * Diagnostic helpers (no global state, single-context-bound).
 */
FA_API const char* fa_version(void);          /* e.g. "0.1.0+652T-sm120a" */
FA_API const char* fa_status_str(fa_status_t);
FA_API const char* fa_last_cuda_error(fa_ctx_t* ctx);  /* readable after FA_ERR_CUDA */

/*
 * Dispatcher introspection — useful for autotuner / testing.
 * Returns kernel id (0..N-1) without launching. id == -1 ⇒ no kernel matches.
 */
typedef enum {
    FA_KERNEL_NONE     = -1,
    /* hd=128 family */
    FA_KERNEL_V121R    = 100,  /* peak wnd=0 */
    FA_KERNEL_V121     = 101,  /* peak wnd>0 (window champion) */
    FA_KERNEL_V118     = 102,  /* mid-grid */
    FA_KERNEL_V122     = 103,  /* bh=4 sl<=2048 wave-tail */
    FA_KERNEL_V117B    = 104,  /* bh=4 sl=8192 wnd=1024 niche */
    FA_KERNEL_V96B     = 105,  /* narrow boundary */
    /* hd=64 family */
    FA_KERNEL_V89      = 200,  /* peak hd=64 */
    FA_KERNEL_V80B     = 201   /* wave-tail hd=64 */
} fa_kernel_id_t;

FA_API fa_kernel_id_t fa_dispatch_select(int batch_heads, int seq_len, int head_dim,
                                         int causal, int window);

FA_API const char* fa_kernel_name(fa_kernel_id_t id);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FA_SM120_H_ */
