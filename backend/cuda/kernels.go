package cuda

// PTX kernels for transformer training operations.
// Target: sm_80+ (Ampere/Ada/Hopper). FP32 only for now.
// Loaded at runtime via cuModuleLoadData — no nvcc needed.
//
// All kernels use 256 threads/block.
// exp(x) is computed as 2^(x * log2(e)) using the hardware ex2 instruction.

const kernelPTX = `
.version 7.0
.target sm_80
.address_size 64

// -- Helper: compute byte offset for element index --
// offset_bytes = idx * 4 (for float32)

// ============================================================
// neg_f32: dst[i] = -src[i]
// ============================================================
.visible .entry neg_f32(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %dst, %src, %off;
    .reg .f32 %val;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %src, [p_src];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_neg_done;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %src, %off;
    ld.global.f32 %val, [%off];
    neg.f32 %val, %val;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %val;
$L_neg_done:
    ret;
}

// ============================================================
// exp_f32: dst[i] = exp(src[i])
// exp(x) = 2^(x * log2(e)),  log2(e) = 1.44269504
// ============================================================
.visible .entry exp_f32(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %dst, %src, %off;
    .reg .f32 %val, %log2e;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %src, [p_src];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_exp_done;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %src, %off;
    ld.global.f32 %val, [%off];
    mov.f32 %log2e, 0f3FB8AA3B;
    mul.f32 %val, %val, %log2e;
    ex2.approx.f32 %val, %val;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %val;
$L_exp_done:
    ret;
}

// ============================================================
// silu_f32: dst[i] = src[i] * sigmoid(src[i])
// sigmoid(x) = 1 / (1 + exp(-x))
// ============================================================
.visible .entry silu_f32(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %dst, %src, %off;
    .reg .f32 %x, %t, %sig, %one, %log2e;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %src, [p_src];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_silu_done;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %src, %off;
    ld.global.f32 %x, [%off];

    mov.f32 %log2e, 0f3FB8AA3B;
    mov.f32 %one, 0f3F800000;
    neg.f32 %t, %x;
    mul.f32 %t, %t, %log2e;
    ex2.approx.f32 %t, %t;
    add.f32 %t, %t, %one;
    rcp.approx.f32 %sig, %t;
    mul.f32 %x, %x, %sig;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %x;
$L_silu_done:
    ret;
}

// ============================================================
// add_f32: dst[i] = a[i] + b[i]
// ============================================================
.visible .entry add_f32(
    .param .u64 p_dst,
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %dst, %a, %b, %off;
    .reg .f32 %va, %vb;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %a, [p_a];
    ld.param.u64 %b, [p_b];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_add_done;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %a, %off;
    ld.global.f32 %va, [%off];
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %b, %off;
    ld.global.f32 %vb, [%off];
    add.f32 %va, %va, %vb;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %va;
$L_add_done:
    ret;
}

// ============================================================
// mul_f32: dst[i] = a[i] * b[i]
// ============================================================
.visible .entry mul_f32(
    .param .u64 p_dst,
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u32 p_n
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %dst, %a, %b, %off;
    .reg .f32 %va, %vb;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %a, [p_a];
    ld.param.u64 %b, [p_b];
    ld.param.u32 %n, [p_n];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_mul_done;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %a, %off;
    ld.global.f32 %va, [%off];
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %b, %off;
    ld.global.f32 %vb, [%off];
    mul.f32 %va, %va, %vb;
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %va;
$L_mul_done:
    ret;
}

// ============================================================
// softmax_f32: softmax along rows
// Grid: (num_rows, 1, 1), Block: (256, 1, 1)
// Thread 0 per block processes one row serially.
// TODO: warp shuffle parallel reduction for performance.
// ============================================================
.visible .entry softmax_f32(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u32 p_row_size,
    .param .u32 p_num_rows
) {
    .reg .u32 %tidx, %bidx, %row_size, %num_rows, %i, %row_off_u32;
    .reg .u64 %dst, %src, %src_base, %dst_base, %off;
    .reg .f32 %val, %max_val, %sum, %log2e, %inv_sum;
    .reg .pred %p, %ploop;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %src, [p_src];
    ld.param.u32 %row_size, [p_row_size];
    ld.param.u32 %num_rows, [p_num_rows];

    mov.u32 %bidx, %ctaid.x;
    setp.ge.u32 %p, %bidx, %num_rows;
    @%p bra $L_sm_done;

    // Only thread 0 does the work
    mov.u32 %tidx, %tid.x;
    setp.ne.u32 %p, %tidx, 0;
    @%p bra $L_sm_done;

    // src_base = src + bid * row_size * 4
    mul.lo.u32 %row_off_u32, %bidx, %row_size;
    mul.wide.u32 %src_base, %row_off_u32, 4;
    add.u64 %src_base, %src, %src_base;
    // dst_base = dst + bid * row_size * 4
    mul.wide.u32 %dst_base, %row_off_u32, 4;
    add.u64 %dst_base, %dst, %dst_base;

    // Pass 1: find max
    mov.f32 %max_val, 0fFF800000;
    mov.u32 %i, 0;
$L_sm_max:
    setp.ge.u32 %ploop, %i, %row_size;
    @%ploop bra $L_sm_max_end;
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %src_base, %off;
    ld.global.f32 %val, [%off];
    max.f32 %max_val, %max_val, %val;
    add.u32 %i, %i, 1;
    bra $L_sm_max;
$L_sm_max_end:

    // Pass 2: exp(x - max), store to dst, accumulate sum
    mov.f32 %sum, 0f00000000;
    mov.f32 %log2e, 0f3FB8AA3B;
    mov.u32 %i, 0;
$L_sm_exp:
    setp.ge.u32 %ploop, %i, %row_size;
    @%ploop bra $L_sm_exp_end;
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %src_base, %off;
    ld.global.f32 %val, [%off];
    sub.f32 %val, %val, %max_val;
    mul.f32 %val, %val, %log2e;
    ex2.approx.f32 %val, %val;
    add.f32 %sum, %sum, %val;
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %dst_base, %off;
    st.global.f32 [%off], %val;
    add.u32 %i, %i, 1;
    bra $L_sm_exp;
$L_sm_exp_end:

    // Pass 3: normalize dst[i] /= sum
    rcp.approx.f32 %inv_sum, %sum;
    mov.u32 %i, 0;
$L_sm_norm:
    setp.ge.u32 %ploop, %i, %row_size;
    @%ploop bra $L_sm_norm_end;
    mul.wide.u32 %off, %i, 4;
    add.u64 %off, %dst_base, %off;
    ld.global.f32 %val, [%off];
    mul.f32 %val, %val, %inv_sum;
    st.global.f32 [%off], %val;
    add.u32 %i, %i, 1;
    bra $L_sm_norm;
$L_sm_norm_end:

$L_sm_done:
    ret;
}

// ============================================================
// rope_f32: Rotary Positional Embedding
// Grid: (batch * heads * seq_len, 1, 1)
// Block: (half_dim, 1, 1)  (clamped to 256)
//
// For each (batch,head,pos) pair, rotate pairs:
//   dst[i]          = src[i]*cos - src[i+half]*sin
//   dst[i+half]     = src[i]*sin + src[i+half]*cos
// where angle = pos * base^(-2i/dim)
// ============================================================
.visible .entry rope_f32(
    .param .u64 p_dst,
    .param .u64 p_src,
    .param .u32 p_seq_len,
    .param .u32 p_head_dim,
    .param .u32 p_num_heads,
    .param .f32 p_base
) {
    .reg .u32 %tidx, %bidx, %half_dim, %seq_len, %head_dim, %pos;
    .reg .u32 %off0, %off1;
    .reg .u64 %dst, %src, %addr;
    .reg .f32 %base, %freq, %angle, %cos_v, %sin_v;
    .reg .f32 %x0, %x1, %r0, %r1, %t0, %t1;
    .reg .f32 %fi, %fdim, %two;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %src, [p_src];
    ld.param.u32 %seq_len, [p_seq_len];
    ld.param.u32 %head_dim, [p_head_dim];
    ld.param.f32 %base, [p_base];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;

    shr.u32 %half_dim, %head_dim, 1;
    setp.ge.u32 %p, %tidx, %half_dim;
    @%p bra $L_rope_done;

    // pos = bid % seq_len
    rem.u32 %pos, %bidx, %seq_len;

    // freq = base^(-2*tid/head_dim) = 2^(-2*tid/head_dim * log2(base))
    cvt.rn.f32.u32 %fi, %tidx;
    cvt.rn.f32.u32 %fdim, %head_dim;
    mov.f32 %two, 0f40000000;
    mul.f32 %freq, %fi, %two;
    div.approx.f32 %freq, %freq, %fdim;
    lg2.approx.f32 %t0, %base;
    mul.f32 %freq, %freq, %t0;
    neg.f32 %freq, %freq;
    ex2.approx.f32 %freq, %freq;

    // angle = pos * freq
    cvt.rn.f32.u32 %angle, %pos;
    mul.f32 %angle, %angle, %freq;

    sin.approx.f32 %sin_v, %angle;
    cos.approx.f32 %cos_v, %angle;

    // off0 = bid * head_dim + tid
    mul.lo.u32 %off0, %bidx, %head_dim;
    add.u32 %off0, %off0, %tidx;
    // off1 = off0 + half_dim
    add.u32 %off1, %off0, %half_dim;

    // Load x0, x1
    mul.wide.u32 %addr, %off0, 4;
    add.u64 %addr, %src, %addr;
    ld.global.f32 %x0, [%addr];

    mul.wide.u32 %addr, %off1, 4;
    add.u64 %addr, %src, %addr;
    ld.global.f32 %x1, [%addr];

    // r0 = x0*cos - x1*sin
    mul.f32 %t0, %x0, %cos_v;
    mul.f32 %t1, %x1, %sin_v;
    sub.f32 %r0, %t0, %t1;

    // r1 = x0*sin + x1*cos
    mul.f32 %t0, %x0, %sin_v;
    mul.f32 %t1, %x1, %cos_v;
    add.f32 %r1, %t0, %t1;

    // Store
    mul.wide.u32 %addr, %off0, 4;
    add.u64 %addr, %dst, %addr;
    st.global.f32 [%addr], %r0;

    mul.wide.u32 %addr, %off1, 4;
    add.u64 %addr, %dst, %addr;
    st.global.f32 [%addr], %r1;

$L_rope_done:
    ret;
}

// ============================================================
// adamw_f32: fused AdamW parameter update
// Grid: (ceil(N/256), 1, 1), Block: (256, 1, 1)
//
// m = beta1*m + (1-beta1)*g
// v = beta2*v + (1-beta2)*g^2
// m_hat = m / beta1_corr
// v_hat = v / beta2_corr
// p = p - lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)
// ============================================================
.visible .entry adamw_f32(
    .param .u64 p_params,
    .param .u64 p_grads,
    .param .u64 p_m,
    .param .u64 p_v,
    .param .u32 p_n,
    .param .f32 p_lr,
    .param .f32 p_beta1,
    .param .f32 p_beta2,
    .param .f32 p_eps,
    .param .f32 p_wd,
    .param .f32 p_b1corr,
    .param .f32 p_b2corr
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %params, %grads, %m, %v, %off;
    .reg .f32 %p_val, %g, %m_val, %v_val;
    .reg .f32 %lr, %beta1, %beta2, %eps, %wd, %b1c, %b2c;
    .reg .f32 %one, %t0, %t1, %mhat, %vhat, %update;
    .reg .pred %pred;

    ld.param.u64 %params, [p_params];
    ld.param.u64 %grads, [p_grads];
    ld.param.u64 %m, [p_m];
    ld.param.u64 %v, [p_v];
    ld.param.u32 %n, [p_n];
    ld.param.f32 %lr, [p_lr];
    ld.param.f32 %beta1, [p_beta1];
    ld.param.f32 %beta2, [p_beta2];
    ld.param.f32 %eps, [p_eps];
    ld.param.f32 %wd, [p_wd];
    ld.param.f32 %b1c, [p_b1corr];
    ld.param.f32 %b2c, [p_b2corr];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %pred, %idx, %n;
    @%pred bra $L_adam_done;

    mov.f32 %one, 0f3F800000;

    // Load p, g, m, v
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %params, %off;
    ld.global.f32 %p_val, [%off];

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %grads, %off;
    ld.global.f32 %g, [%off];

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %m, %off;
    ld.global.f32 %m_val, [%off];

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %v, %off;
    ld.global.f32 %v_val, [%off];

    // m = beta1*m + (1-beta1)*g
    sub.f32 %t0, %one, %beta1;
    mul.f32 %m_val, %m_val, %beta1;
    mul.f32 %t1, %g, %t0;
    add.f32 %m_val, %m_val, %t1;

    // v = beta2*v + (1-beta2)*g^2
    sub.f32 %t0, %one, %beta2;
    mul.f32 %v_val, %v_val, %beta2;
    mul.f32 %t1, %g, %g;
    mul.f32 %t1, %t1, %t0;
    add.f32 %v_val, %v_val, %t1;

    // Store m, v
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %m, %off;
    st.global.f32 [%off], %m_val;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %v, %off;
    st.global.f32 [%off], %v_val;

    // m_hat = m / b1_corr, v_hat = v / b2_corr
    div.approx.f32 %mhat, %m_val, %b1c;
    div.approx.f32 %vhat, %v_val, %b2c;

    // update = m_hat / (sqrt(v_hat) + eps)
    sqrt.approx.f32 %t0, %vhat;
    add.f32 %t0, %t0, %eps;
    div.approx.f32 %update, %mhat, %t0;

    // Weight decay: update += wd * p
    mul.f32 %t0, %wd, %p_val;
    add.f32 %update, %update, %t0;

    // p = p - lr * update
    mul.f32 %update, %lr, %update;
    sub.f32 %p_val, %p_val, %update;

    // Store p
    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %params, %off;
    st.global.f32 [%off], %p_val;

$L_adam_done:
    ret;
}

// ============================================================
// embedding_f32: dst[seq][dim] = weight[indices[seq]][dim]
// Grid: (seq_len, 1, 1), Block: (min(embed_dim, 256), 1, 1)
// ============================================================
.visible .entry embedding_f32(
    .param .u64 p_dst,
    .param .u64 p_weight,
    .param .u64 p_indices,
    .param .u32 p_embed_dim
) {
    .reg .u32 %tidx, %bidx, %dim, %idx32, %woff, %doff;
    .reg .u64 %dst, %weight, %indices, %addr;
    .reg .s64 %idx64;
    .reg .f32 %val;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u64 %weight, [p_weight];
    ld.param.u64 %indices, [p_indices];
    ld.param.u32 %dim, [p_embed_dim];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;

    setp.ge.u32 %p, %tidx, %dim;
    @%p bra $L_emb_done;

    // Load index: indices[bid] (int64)
    mul.wide.u32 %addr, %bidx, 8;
    add.u64 %addr, %indices, %addr;
    ld.global.s64 %idx64, [%addr];
    cvt.u32.s64 %idx32, %idx64;

    // Load weight[idx][tid]
    mul.lo.u32 %woff, %idx32, %dim;
    add.u32 %woff, %woff, %tidx;
    mul.wide.u32 %addr, %woff, 4;
    add.u64 %addr, %weight, %addr;
    ld.global.f32 %val, [%addr];

    // Store to dst[bid][tid]
    mul.lo.u32 %doff, %bidx, %dim;
    add.u32 %doff, %doff, %tidx;
    mul.wide.u32 %addr, %doff, 4;
    add.u64 %addr, %dst, %addr;
    st.global.f32 [%addr], %val;

$L_emb_done:
    ret;
}

// ============================================================
// fill_f32: dst[i] = value
// ============================================================
.visible .entry fill_f32(
    .param .u64 p_dst,
    .param .u32 p_n,
    .param .f32 p_value
) {
    .reg .u32 %tidx, %bidx, %idx, %n;
    .reg .u64 %dst, %off;
    .reg .f32 %val;
    .reg .pred %p;

    ld.param.u64 %dst, [p_dst];
    ld.param.u32 %n, [p_n];
    ld.param.f32 %val, [p_value];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bidx, %ctaid.x;
    mad.lo.u32 %idx, %bidx, 256, %tidx;
    setp.ge.u32 %p, %idx, %n;
    @%p bra $L_fill_done;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %off, %dst, %off;
    st.global.f32 [%off], %val;
$L_fill_done:
    ret;
}
` + "\x00" // null terminator for cuModuleLoadData

// kernelNames lists all kernels in the PTX module.
var kernelNames = []string{
	"neg_f32",
	"exp_f32",
	"silu_f32",
	"add_f32",
	"mul_f32",
	"softmax_f32",
	"rope_f32",
	"adamw_f32",
	"embedding_f32",
	"fill_f32",
}
