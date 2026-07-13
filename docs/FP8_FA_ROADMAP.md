# FP8 Flash Attention Roadmap (goml)

Honest engineering plan for adding FP8 precision to FA forward and backward
on Blackwell sm_120a. Last update 2026-06-04 (v65 complete, SMEM-bound).

## Current state (2026-06-04 evening)

| Version | TFLOPS @ sl=4096 | What |
|---------|------------------|------|
| v61 PoC | (verify only) | mma kind::f8f6f4 path proven |
| v62 full | 75 T | end-to-end correctness, baseline |
| v63 +transpose +cvt | 166 T | +121% (gather_v_b eliminated, hw FP8 cvt) |
| v64 +double-buffer | 162 T | parity (not memory-bound, no win) |
| **v65 +bigger tiles (Br=128)** | **184 T** | **+10% from amortization** |
| _v55 FP16 baseline_ | _271 T_ | we're at 68% of FP16 |

## Critical insight from v65 NCu profile

v65 is **L1/SMEM cache throughput bound (53%), NOT compute bound (26%)**.
This was unexpected — naive analysis suggested MMA dispatch was the limit.
Real bottleneck: too many scalar SMEM reads in the MMA inner loops.

This changes the optimization strategy completely:
  - stmatrix wouldn't help (it's a STORE op, not a read).
  - Double-buffering wouldn't help (already proven in v64: no gain).
  - **M_TILES=2 per warp** is the right next move — amortize each B operand
    SMEM read across 2× more MMA work.

## TL;DR forward path

| Path | Effort | Expected gain | Risk |
|------|--------|---------------|------|
| ✅ **v61** PoC | 2-3 hours | parity (75 T) | done |
| ✅ **v62** full pipeline | done | 75 T | done |
| ✅ **v63** transpose+cvt | done | **+121% → 166 T** | done |
| ✅ **v64** double-buffer | done | flat (mem not binding) | done |
| ✅ **v65** bigger tiles | done | +10% → **184 T** | done |
| **v66** M_TILES=2/warp | 2-3 hours | +20-30% → 220-240 T | medium |
| **v67** ldmatrix (if FP8-compatible swizzle works) | 4-6 hours | +30-50% → 280-330 T | high |
| **v68** block scaled mxf8f6f4 | 4-6 hours | accuracy parity | medium |
| **v69** FP4 mxf4nvf4 | 1-2 days research | +50-100% → 400+ T | very high |
| **v70** backward FP8 | 5-7 days | 127→200+ | high |

Total to "production FP8 training-ready" stack: **2-3 weeks** focused.

## Why FP8 unlocks real headroom

Verified peaks on RTX PRO 6000 Blackwell (sm_120a) by powers-of-two from the
4000 TOPS sparse FP4 marketing number:

| Precision | Dense | Sparse |
|-----------|------:|-------:|
| FP4       | 2000 T| 4000 T |
| FP8       | **1000 T**| 2000 T |
| FP16      | 500 T | 1000 T |
| FP32      | 125 T | 250 T  |

Our current best:
- FA forward v55 = **271 T at 54% of FP16 dense peak**
- FA backward v58 = **127 T at 25% of FP16 dense peak**
- fp8_gemm v23/v24 = **384 T at 38% of FP8 dense peak**

Forward is close to saturation in FP16. Backward + GEMM have headroom even
in FP16. But the biggest jump comes from switching precision: 500 T → 1000 T
peak is +100%.

## What we already have

- `libs/fp8_gemm.cu` (v23) — production FP8 GEMM with **`mma.sync m16n8k32
  .f16.e4m3.e4m3.f16`** (Ada-style FP8). 587 T on RTX 4090, 382 T on Blackwell.
- `libs/fp8_gemm_v24.cu` (v24) — same numbers via **`kind::f8f6f4`** unified
  Blackwell 5th-gen path with **FP16 accumulator** (undocumented PTX trick
  verified empirically). Drop-in replacement, same perf.
- `libs/flash_attention_v55_forward.cu` — FA forward FP16 with FP16 accum.
- `libs/flash_attention_v58_backward.cu` — FA backward FP16, partial f16 accum.
- `runs/kind_f8f6f4_f16acc_probe.cu` — proof that `kind::f8f6f4.f16.e4m3.e4m3.f16`
  is supported on sm_120a.

## Implementation plan

### Step 1 — v61 FP8 forward PoC (2-3 hours)

**Goal**: correctness, no perf focus. Prove the pipeline works.

1. Copy `fp8_gemm.cu` SMEM access pattern (scalar uint32_t reads from FP8
   swizzled tiles). NOT ldmatrix — that's FP16-oriented.
2. Adopt the FA structure from `flash_attention_v55_forward.cu` (online
   softmax, K-block iteration, Or accumulator).
3. Replace the m16n8k16 FP16 MMA with **`mma.sync.aligned.m16n8k32.row.col.
   kind::f8f6f4.f16.e4m3.e4m3.f16`** (proven to work on sm_120a, f16 accum).
4. Adjust loop bounds: head_dim=128 → K_STEPS = 4 (instead of 8 for FP16)
   because each MMA covers k=32 instead of k=16.
5. Halve SMEM allocation: FP8 = 1 byte/elem so K and V tiles fit in half the
   space. Q tile too.
6. Add quantization in the launcher: FP16 input → FP8 (e4m3) with per-tensor
   scale (or per-row for K, V).
7. Test correctness vs CPU FP16 reference. Expected: max diff ~0.01-0.05
   (FP8 precision floor).

**Output**: working FP8 forward with verified correctness. Probably ~250-350 T
(similar to FP16 since no SMEM/register optimization yet).

### Step 2 — v62 production FP8 forward (1-2 days)

**Goal**: hit 450-550 T forward.

Building on v61:

1. **Smaller SMEM → more blocks/SM**. FP8 halves Q/K/V SMEM. Aim for 3 blocks/SM
   by registers (already achievable with v55 f16 accum trick).
2. **Larger M_TILES per warp**. With smaller SMEM/block, can fit more Q rows
   per block (Br=128). Each warp owns 32 rows = 2× M_TILES.
3. **Reduce Or register pressure** via FP16 packed accum (same as v55).
4. **Profile with ncu** and tune block size (try 128 vs 192 vs 256 threads).
5. **Backward-compat check**: verify on multiple head_dim (64, 128, 256).

**Output**: 450+ T FP8 forward, correctness within 1% of FP16 reference.

### Step 3 — v63 FP8 backward (5-7 days)

**Goal**: 200+ T backward.

Backward FP8 is much harder because gradients have dynamic range that
varies dramatically across layers and tokens.

Key design decisions:
1. **Per-tile scales** for dO, dQ, dK, dV. Need to track max-abs per tile and
   re-scale on the fly.
2. **FP16 grad output for caller**: keep dQ, dK, dV in FP16 storage (caller
   expects them); only the inner MMA uses FP8 inputs from Q, K, V, dO.
3. **Two-pass structure** stays (from v55/v58 backward).
4. **Numerical safety**: dS = P*(dP - D) can be small; accumulating ε² in FP16
   may underflow. May need FP32 accumulator on dQ, dK, dV (cross-block).

**Output**: 200+ T FP8 backward, training-usable accuracy.

### Step 4 — v64 Block-scaled training accuracy (2-3 days)

**Goal**: training without accuracy degradation vs FP16.

Use **`kind::mxf8f6f4.block_scale.scale_vec::1X`** with **ue8m0** per-block
scale (32-element blocks). This is the "production training" path that
FA3 / FP8 transformers use.

1. Add per-32-element scale tracking on Q, K, V quantization.
2. Pass scales as additional kernel args.
3. Modify MMA to use `kind::mxf8f6f4.block_scale.scale_vec::1X.f16.e4m3.e4m3.f16.ue8m0`.
4. Validate against PyTorch FP8 attention reference (TransformerEngine).

### Step 5 — v65 FP4 experiment (3-5 days)

**Goal**: 2× compute over FP8 → up to 900 T forward.

Use **`kind::mxf4nvf4.block_scale.scale_vec::4X.f16.e2m1.e2m1.f16.ue4m3`**
(NVIDIA's FP4 format with FP8 scales). m16n8k64 instead of m16n8k32.

Research-tier task — accuracy may be insufficient for inference quality.

## Estimated full B execution (3-4 weeks)

| Week | Output |
|------|--------|
| Week 1 | v61 PoC + v62 production forward (~500 T) |
| Week 2 | v63 backward + accuracy validation |
| Week 3 | v64 block-scaled (production training) |
| Week 4 | Tuning, integration with goml, optional v65 FP4 |

## What this session achieved

- v55 backward → v58 backward: +9-10% via FP16 accum on S, dP MMAs
- v54 forward → v55 forward: +3% via FP16 accum on Or
- Verified `kind::f8f6f4` with f16 accum works on sm_120a
- Verified WGMMA / tcgen05 do NOT exist on sm_120a
- Established that FP16 is at ~54% of dense peak — FP8 is the next jump

The cheap-FP16-optimization phase is **done**. Next session(s) execute FP8.
