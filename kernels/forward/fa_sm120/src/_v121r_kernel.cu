// =============================================================================
// FlashAttention v69 — FP8 Forward, single-buffer V → 2 blocks/SM
// =============================================================================
// Production default for sm_120a (RTX PRO 6000 Blackwell). Replaces v68 = 220T
// peak with new ceiling 338T (+53%) on production-shape configs (≥256 blocks).
//
// Single 8 KB SMEM save vs v68 — by single-buffering V (drop smV[1]) — enables
// 2 blocks/SM (vs v68's 1). For grids ≥ 188 × 2 = 376 blocks, this halves wave
// count; for 256+ blocks (where waves go 2→1) gain is +51%; for 512+ (3→2)
// gain is +15-23%. For small grids (<188 blocks) v69 = v68 paritet (no harm).
//
// SMEM layout (48.5 KB):
//   smQ:    16 KB
//   smK[2]: 16 KB  (K stays double-buffered)
//   smV:     8 KB  (was 16 KB = double-buffered)
//   smV_T:  8.5 KB (padded stride 68 from v68, breaks 32-way write conflict)
//   smP overlaps smV after transpose_v (cur_V data extracted to smV_T)
//
// Sequencing change vs v68: V prefetch moves to END of iter (after smP read).
// K prefetch stays at MID-iter and still overlaps with compute. Cost: V load
// loses overlap, but v64 datapoint + v68 NCu (mem busy 22%) prove kernel is
// NOT memory-bound → V overlap loss ≤ 2% on all measured shapes.
//
// Why v69 > v68:
//   - Occupancy 8.33% (1 block/SM × 4 warps) → 16.67% (2 × 4 = 8 warps/SM)
//   - Directly hides MMA pipeline latency (was 1.06 cycles/inst floor in v68)
//   - Wave reduction on large grids (typical for batched LLM inference)
//
// Build: nvcc -gencode arch=compute_120a,code=sm_120a
// History: v66 baseline → v68 conflict fix (smV_T padding) → v69 single-V.
//          v67 TMA rejected. v69_singleV experiment kept as reference.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <algorithm>

#define FA_BR 128
#define FA_BC 64
#define FA_THREADS 128   // 4 warps × M_TILES=2 × 16 rows = Br=128
#define FA_STRIDE 128
#define M_TILES 2        // each warp owns 2 m16 sub-tiles in M direction
#define SMV_T_STRIDE 68  // padded smV_T row stride to break 32-way conflict
                          // (64 + 4: gcd(17, 32) = 1 → 32 lanes hit 32 banks)

__device__ __forceinline__ void cpa16(void *s, const void *g, int n)
{
    uint32_t sa = __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" ::"r"(sa), "l"(g), "r"(n));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

__device__ __forceinline__ void mma_fp8_f16(
    uint32_t &d0, uint32_t &d1,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t c0, uint32_t c1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e4m3.e4m3.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(c0), "r"(c1));
}

__device__ __forceinline__ int swz_byte(int row, int col_bytes)
{
    int chunk = col_bytes >> 4;
    int within = col_bytes & 15;
    return row * FA_STRIDE + ((chunk ^ (row & 7)) << 4) + within;
}

// Swizzle for smP (stride = FA_BC = 64 bytes per row).
// Still used by P quantize writes and P read in P·V loop.
__device__ __forceinline__ int swz_byte_bc(int row, int col_bytes)
{
    int chunk = col_bytes >> 4;
    int within = col_bytes & 15;
    return row * FA_BC + ((chunk ^ (row & 3)) << 4) + within;
}

// v68 padded swizzle for smV_T (stride = SMV_T_STRIDE = 68).
// Breaks 32-way write conflict in transpose_v: with stride 64 the
// lane-to-bank period was 128 B → all 32 lanes hit bank 0.
// With stride 68: lane_stride / 4 = 17, gcd(17, 32) = 1 → 32 distinct banks.
__device__ __forceinline__ int swz_byte_smvt(int row, int col_bytes)
{
    int chunk = col_bytes >> 4;
    int within = col_bytes & 15;
    return row * SMV_T_STRIDE + ((chunk ^ (row & 3)) << 4) + within;
}

__device__ __forceinline__ void load_tile_fp8(
    uint8_t *dst, const uint8_t *src, int start, int rows,
    int seq_len, int head_dim)
{
    constexpr int CHUNK = 16;
    int chunks_per_row = head_dim / CHUNK;
    int total = rows * chunks_per_row;
#pragma unroll 4
    for (int c = threadIdx.x; c < total; c += FA_THREADS)
    {
        int row = c / chunks_per_row;
        int col_bytes = (c % chunks_per_row) * CHUNK;
        int gr = start + row;
        int dst_off = swz_byte(row, col_bytes);
        cpa16(&dst[dst_off], &src[gr * head_dim + col_bytes], (gr < seq_len) ? 16 : 0);
    }
}

// Hardware FP16x2 → FP8x2 conversion. cvt.rn.satfinite.e4m3x2.f16x2 is
// available on sm_89+ as a single PTX instruction.
__device__ __forceinline__ uint16_t fp16x2_to_e4m3x2(uint32_t h2)
{
    uint16_t out;
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;"
                 : "=h"(out) : "r"(h2));
    return out;
}

// Transpose smV [seq_k=Bc, head_dim] → smV_T [head_dim, seq_k=Bc].
// Once per K-block iter. After this, V B-operand reads use the same fast
// scalar uint32_t pattern as K (no byte-gather).
//
// Each thread copies a 4×4 tile: 4 consecutive k-rows × 4 consecutive n-cols.
// Reads as uint32_t (4 contiguous n-bytes per k-row), reassembles via
// bytewise shuffle, writes as uint32_t into the transposed layout.
__device__ __forceinline__ void transpose_v(
    uint8_t *smV_T, const uint8_t *smV, int head_dim)
{
    // Layout: smV[k_row * FA_STRIDE + n_col] (swizzled by swz_byte)
    //         smV_T[n_row * FA_STRIDE + k_col] (swizzled by swz_byte)
    // We do 16 4x4 transposes per thread (covers 64 rows × 64 cols = 4096 elems).
    constexpr int TILE = 4;
    int tiles_k = FA_BC / TILE;       // 16
    int tiles_n = head_dim / TILE;     // 32 for hd=128
    int total = tiles_k * tiles_n;     // 512
    for (int t = threadIdx.x; t < total; t += FA_THREADS)
    {
        int tk = t / tiles_n;          // 0..15
        int tn = t % tiles_n;          // 0..31
        int k0 = tk * TILE;
        int n0 = tn * TILE;
        // Load 4 uint32_t (one per k-row) = 4x4 fp8 block
        uint32_t r0 = *(uint32_t *)&smV[swz_byte(k0 + 0, n0)];
        uint32_t r1 = *(uint32_t *)&smV[swz_byte(k0 + 1, n0)];
        uint32_t r2 = *(uint32_t *)&smV[swz_byte(k0 + 2, n0)];
        uint32_t r3 = *(uint32_t *)&smV[swz_byte(k0 + 3, n0)];
        // Transpose 4x4: write 4 uint32_t (one per n-col, 4 consecutive k bytes)
        uint32_t c0 = ((r0 >>  0) & 0xff)
                    | ((r1 <<  8) & 0xff00)
                    | ((r2 << 16) & 0xff0000)
                    | ((r3 << 24) & 0xff000000);
        uint32_t c1 = ((r0 >>  8) & 0xff)
                    | ((r1 <<  0) & 0xff00)
                    | ((r2 <<  8) & 0xff0000)
                    | ((r3 << 16) & 0xff000000);
        uint32_t c2 = ((r0 >> 16) & 0xff)
                    | ((r1 >>  8) & 0xff00)
                    | ((r2 <<  0) & 0xff0000)
                    | ((r3 <<  8) & 0xff000000);
        uint32_t c3 = ((r0 >> 24) & 0xff)
                    | ((r1 >> 16) & 0xff00)
                    | ((r2 >>  8) & 0xff0000)
                    | ((r3 <<  0) & 0xff000000);
        *(uint32_t *)&smV_T[swz_byte_smvt(n0 + 0, k0)] = c0;
        *(uint32_t *)&smV_T[swz_byte_smvt(n0 + 1, k0)] = c1;
        *(uint32_t *)&smV_T[swz_byte_smvt(n0 + 2, k0)] = c2;
        *(uint32_t *)&smV_T[swz_byte_smvt(n0 + 3, k0)] = c3;
    }
}

__global__ void __launch_bounds__(FA_THREADS, 2)  // R-cost measurement at 2 blocks/SM
    fa96b_kernel(
        const uint8_t *__restrict__ Q,
        const uint8_t *__restrict__ K,
        const uint8_t *__restrict__ V,
        __half *__restrict__ O,
        int seq_len, int head_dim, int causal, float scale,
        float qk_descale, float v_descale,
        int window)  // 0 = no window (full attention or causal). >0 = sliding window.
{
    int nqt = (seq_len + FA_BR - 1) / FA_BR;
    int bh = blockIdx.x / nqt, qt = blockIdx.x % nqt, qs = qt * FA_BR;
    if (qs >= seq_len) return;
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    int gid = lane / 4, tid = lane % 4;
    int mrb = wid * 32;  // M_TILES=2 → each warp owns 32 rows

    // v121: kv-loop swizzle invariants — gid&7 для FA_STRIDE-swizzle (cur_K, smQ),
    // gid&3 для smV_T (stride 68) и smP (stride 64). gid_lane_base = gid*128,
    // gid_smvt_base = gid*68 — лейн-смещения, переиспользуемые в каждом батче.
    const int gid7 = gid & 7;
    const int gid3 = gid & 3;
    const int gid_lane_base = gid * FA_STRIDE;
    const int gid_smvt_base = gid * SMV_T_STRIDE;

    extern __shared__ uint8_t raw[];
    uint8_t *smQ = raw;
    // v96b LOCAL-MEM FIX: replace `uint8_t *smK[2]` (16-byte stack frame + LDL.64
    // on dynamic index in hot loop) with base+arithmetic-stride.
    constexpr int K_STRIDE_BYTES = FA_BC * FA_STRIDE;
    uint8_t *smK_base = smQ + FA_BR * FA_STRIDE;
    // smK[s] is replaced by (smK_base + s * K_STRIDE_BYTES) inline at call sites.
    // v69_singleV: single V buffer (was double). smP overlaps smV after
    // transpose. V prefetch goes to END of iter (cannot overlap with smP use).
    uint8_t *smV = smK_base + 2 * K_STRIDE_BYTES;   // was smK[1] + FA_BC * FA_STRIDE
    uint8_t *smV_T = smV + FA_BC * FA_STRIDE;

    int hs = seq_len * head_dim;
    const uint8_t *Qh = Q + bh * hs;
    const uint8_t *Kh = K + bh * hs;
    const uint8_t *Vh = V + bh * hs;
    __half *Oh = O + bh * hs;

    load_tile_fp8(smQ, Qh, qs, FA_BR, seq_len, head_dim);
    cpa_commit();
    cpa_wait<0>();
    __syncthreads();

    // Qr[ks][mi][r] — 4 k-steps × 2 M-tiles × 4 uint32 (m16k32 A operand)
    uint32_t Qr[4][M_TILES][4];
#pragma unroll
    for (int ks = 0; ks < 4; ks++)
    {
        int k_off = ks * 32;
        int cl = k_off + tid * 4;
        int ch = cl + 16;
#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++)
        {
            int mr = mrb + mi * 16;
            int g0 = mr + gid, g8 = g0 + 8;
            Qr[ks][mi][0] = *(uint32_t *)&smQ[swz_byte(g0, cl)];
            Qr[ks][mi][1] = *(uint32_t *)&smQ[swz_byte(g8, cl)];
            Qr[ks][mi][2] = *(uint32_t *)&smQ[swz_byte(g0, ch)];
            Qr[ks][mi][3] = *(uint32_t *)&smQ[swz_byte(g8, ch)];
        }
    }

    // Or_p[nt][mi][r] — 16 N-tiles × 2 M-tiles × 2 packed uint32 (m16n8 D)
    uint32_t Or_p[16][M_TILES][2];
#pragma unroll
    for (int t = 0; t < 16; t++)
#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++)
            Or_p[t][mi][0] = Or_p[t][mi][1] = 0u;

    // Per-row state: [mi][side] where side=0 is gid row, side=1 is gid+8 row
    float rmax[M_TILES][2] = {{-1e30f, -1e30f}, {-1e30f, -1e30f}};
    float rsexp[M_TILES][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    int nkv = (seq_len + FA_BC - 1) / FA_BC;
    int kv_max_blocks = causal ? ((qs + FA_BR - 1) / FA_BC + 1) : nkv;
    if (kv_max_blocks > nkv) kv_max_blocks = nkv;
    // v69+window: sliding-window lower bound. For causal sliding window, Q-row q
    // attends to K in [max(0, q - window + 1), q]. Q-block covers qs..qs+Br-1;
    // earliest K needed = max(0, qs - window + 1). Skip K-blocks below that.
    int kv_min_blocks = 0;
    if (window > 0 && qs + 1 > window) {
        kv_min_blocks = (qs - window + 1) / FA_BC;  // floor
    }

    // Pre-load iter kv_min_blocks: K + V. Iter k prefetches iter k+1 → other bank.
    load_tile_fp8(smK_base + (kv_min_blocks & 1) * K_STRIDE_BYTES,
                  Kh, kv_min_blocks * FA_BC, FA_BC, seq_len, head_dim);
    load_tile_fp8(smV, Vh, kv_min_blocks * FA_BC, FA_BC, seq_len, head_dim);
    cpa_commit();

    // v78: V buffer alternates location. Iter kv_min reads V from smV (pre-loop).
    // Iter kv_min+N (N≥1) reads V from smK[buf_prev] (loaded by prev iter's mid-iter cp.async).
    uint8_t *prev_V_slot = smV;

    for (int kv = kv_min_blocks; kv < kv_max_blocks; kv++)
    {
        int kvs = kv * FA_BC;
        int buf = kv & 1;
        // v96b: smK[buf] / smK[buf^1] → arithmetic. buf uniform per-iter → IMAD.
        uint8_t *cur_K = smK_base + buf * K_STRIDE_BYTES;
        uint8_t *nxt_K = smK_base + (buf ^ 1) * K_STRIDE_BYTES;

        // Wait until current iter's K AND V land (V may be in smK[buf_prev] = nxt_K).
        cpa_wait<0>();
        __syncthreads();

        // v78: read V from wherever prev iter's mid-iter cp.async put it.
        // NOTE on Lever 1 (K-before-transpose) — IT IS UNSAFE in this scheme.
        // prev_V_slot == nxt_K in iter ≥ 1 (V[N+1] and K[N+2] alias the same smK slot).
        // K cp.async before transpose would write the slot WHILE transpose reads it.
        transpose_v(smV_T, prev_V_slot, head_dim);
        __syncthreads();

        uint8_t *smP = smV;  // smV stays the smP scratchpad (only iter kv_min stored real V here)

        // v79 lever 3: branch-free row count — no `if` guard, ternary clamps for last iter.
        // load_tile_fp8's inner loop runs 0 iters when rows_p=0 → no cp.async issued.
        int kv_p = kv + 1;
        int rows_p = (kv_p < kv_max_blocks) ? FA_BC : 0;
        load_tile_fp8(nxt_K, Kh, kv_p * FA_BC, rows_p, seq_len, head_dim);
        cpa_commit();

        // S = Q · Kᵀ — K B-operand loaded once per (nt, ks), reused across mi.
        uint32_t Sr_p[8][M_TILES][2];
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
                Sr_p[nt][mi][0] = Sr_p[nt][mi][1] = 0u;
        // v96: Option C ks-batching ported from v87 hd=64.
        // hd=128 has 4 ks-steps in QK (head_dim/32). Explicit batches replace
        // `for ks` outer loop — each batch is a complete (nt, mi) sweep with
        // fixed ks. Scheduler sees explicit phase boundaries.
        // v121 step1a: QK B-операнды hoist swz_byte инвариантов.
        // br=nt*8 → (br+gid)&7 = gid&7. row*128 = 1024*nt + 128*gid.
        // base_b{0,1} = 128*gid + ((col>>4)^gid7)<<4 + (col&15) — const по nt.
        // === QK ks=0 batch ===
        {
            int cl = tid * 4, ch = cl + 16;
            const int base_b0 = gid_lane_base + (((cl >> 4) ^ gid7) << 4) + (cl & 15);
            const int base_b1 = gid_lane_base + (((ch >> 4) ^ gid7) << 4) + (ch & 15);
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                uint32_t b0 = *(uint32_t *)&cur_K[base_b0 + 1024 * nt];
                uint32_t b1 = *(uint32_t *)&cur_K[base_b1 + 1024 * nt];
#pragma unroll
                for (int mi = 0; mi < M_TILES; mi++)
                {
                    mma_fp8_f16(Sr_p[nt][mi][0], Sr_p[nt][mi][1],
                                Qr[0][mi][0], Qr[0][mi][1],
                                Qr[0][mi][2], Qr[0][mi][3],
                                b0, b1, Sr_p[nt][mi][0], Sr_p[nt][mi][1]);
                }
            }
        }
        // === QK ks=1 batch ===
        {
            int cl = 32 + tid * 4, ch = cl + 16;
            const int base_b0 = gid_lane_base + (((cl >> 4) ^ gid7) << 4) + (cl & 15);
            const int base_b1 = gid_lane_base + (((ch >> 4) ^ gid7) << 4) + (ch & 15);
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                uint32_t b0 = *(uint32_t *)&cur_K[base_b0 + 1024 * nt];
                uint32_t b1 = *(uint32_t *)&cur_K[base_b1 + 1024 * nt];
#pragma unroll
                for (int mi = 0; mi < M_TILES; mi++)
                {
                    mma_fp8_f16(Sr_p[nt][mi][0], Sr_p[nt][mi][1],
                                Qr[1][mi][0], Qr[1][mi][1],
                                Qr[1][mi][2], Qr[1][mi][3],
                                b0, b1, Sr_p[nt][mi][0], Sr_p[nt][mi][1]);
                }
            }
        }
        // === QK ks=2 batch ===
        {
            int cl = 64 + tid * 4, ch = cl + 16;
            const int base_b0 = gid_lane_base + (((cl >> 4) ^ gid7) << 4) + (cl & 15);
            const int base_b1 = gid_lane_base + (((ch >> 4) ^ gid7) << 4) + (ch & 15);
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                uint32_t b0 = *(uint32_t *)&cur_K[base_b0 + 1024 * nt];
                uint32_t b1 = *(uint32_t *)&cur_K[base_b1 + 1024 * nt];
#pragma unroll
                for (int mi = 0; mi < M_TILES; mi++)
                {
                    mma_fp8_f16(Sr_p[nt][mi][0], Sr_p[nt][mi][1],
                                Qr[2][mi][0], Qr[2][mi][1],
                                Qr[2][mi][2], Qr[2][mi][3],
                                b0, b1, Sr_p[nt][mi][0], Sr_p[nt][mi][1]);
                }
            }
        }
        // === QK ks=3 batch ===
        {
            int cl = 96 + tid * 4, ch = cl + 16;
            const int base_b0 = gid_lane_base + (((cl >> 4) ^ gid7) << 4) + (cl & 15);
            const int base_b1 = gid_lane_base + (((ch >> 4) ^ gid7) << 4) + (ch & 15);
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                uint32_t b0 = *(uint32_t *)&cur_K[base_b0 + 1024 * nt];
                uint32_t b1 = *(uint32_t *)&cur_K[base_b1 + 1024 * nt];
#pragma unroll
                for (int mi = 0; mi < M_TILES; mi++)
                {
                    mma_fp8_f16(Sr_p[nt][mi][0], Sr_p[nt][mi][1],
                                Qr[3][mi][0], Qr[3][mi][1],
                                Qr[3][mi][2], Qr[3][mi][3],
                                b0, b1, Sr_p[nt][mi][0], Sr_p[nt][mi][1]);
                }
            }
        }

        // v78: V[kv+1] cp.async → smK[buf] (dead after QK MMA finished). Overlaps
        // with softmax + smP STS + PV MMA (~60% of iter time).
        // v79 lever 3: branch-free. rows_v=0 on last iter → load_tile_fp8 inner loop is no-op.
        // prev_V_slot is unconditionally set to smK[buf]; on the last iter it'll never be
        // read (we exit the loop), so no harm.
        int rows_v = (kv + 1 < kv_max_blocks) ? FA_BC : 0;
        // v96b: smK[buf] → arithmetic
        uint8_t *v_dst = smK_base + buf * K_STRIDE_BYTES;
        load_tile_fp8(v_dst, Vh, (kv + 1) * FA_BC, rows_v, seq_len, head_dim);
        prev_V_slot = v_dst;
        cpa_commit();

        // v121r R2: Sr float[8][2][4] → __half2[8][2][2]. f16x2 native ops:
        //   Sr[nt][mi][0] = (Sr0, Sr1) top, Sr[nt][mi][1] = (Sr2, Sr3) bot.
        // rmax/nm/rsexp/rsc/ns остаются float (стабильность running sums).
        __half2 Sr[8][M_TILES][2];
        const float fs = scale * qk_descale * 1.4426950408889634f;
        const __half2 fs_h2 = __float2half2_rn(fs);
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                __half2 v0 = *reinterpret_cast<__half2 *>(&Sr_p[nt][mi][0]);
                __half2 v1 = *reinterpret_cast<__half2 *>(&Sr_p[nt][mi][1]);
                Sr[nt][mi][0] = __hmul2(v0, fs_h2);
                Sr[nt][mi][1] = __hmul2(v1, fs_h2);
            }
        }

        const __half NEG_INF_H = __float2half(-65504.0f);
        const __half2 NEG_INF_H2 = __halves2half2(NEG_INF_H, NEG_INF_H);
        if (causal)
        {
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                int gq0 = qs + mrb + mi * 16 + gid, gq8 = gq0 + 8;
                int kmin0 = (window > 0 && gq0 + 1 > window) ? (gq0 - window + 1) : 0;
                int kmin8 = (window > 0 && gq8 + 1 > window) ? (gq8 - window + 1) : 0;
#pragma unroll
                for (int nt = 0; nt < 8; nt++)
                {
                    int gk0 = kvs + nt * 8 + tid * 2, gk1 = gk0 + 1;
                    bool m_top_lo = (gk0 > gq0) || (gk0 < kmin0) || (gq0 >= seq_len) || (gk0 >= seq_len);
                    bool m_top_hi = (gk1 > gq0) || (gk1 < kmin0) || (gq0 >= seq_len) || (gk1 >= seq_len);
                    bool m_bot_lo = (gk0 > gq8) || (gk0 < kmin8) || (gq8 >= seq_len) || (gk0 >= seq_len);
                    bool m_bot_hi = (gk1 > gq8) || (gk1 < kmin8) || (gq8 >= seq_len) || (gk1 >= seq_len);
                    if (m_top_lo || m_top_hi) {
                        __half lo = m_top_lo ? NEG_INF_H : __low2half(Sr[nt][mi][0]);
                        __half hi = m_top_hi ? NEG_INF_H : __high2half(Sr[nt][mi][0]);
                        Sr[nt][mi][0] = __halves2half2(lo, hi);
                    }
                    if (m_bot_lo || m_bot_hi) {
                        __half lo = m_bot_lo ? NEG_INF_H : __low2half(Sr[nt][mi][1]);
                        __half hi = m_bot_hi ? NEG_INF_H : __high2half(Sr[nt][mi][1]);
                        Sr[nt][mi][1] = __halves2half2(lo, hi);
                    }
                }
            }
        }

        // Per-tile softmax: max, rescale Or, exp+sum.
        float nm[M_TILES][2];
        float rsc[M_TILES][2];
#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++)
        {
            __half2 nm_h2_top = NEG_INF_H2, nm_h2_bot = NEG_INF_H2;
#pragma unroll
            for (int nt = 0; nt < 8; nt++)
            {
                nm_h2_top = __hmax2(nm_h2_top, Sr[nt][mi][0]);
                nm_h2_bot = __hmax2(nm_h2_bot, Sr[nt][mi][1]);
            }
            nm[mi][0] = fmaxf(__half2float(__low2half(nm_h2_top)),
                              __half2float(__high2half(nm_h2_top)));
            nm[mi][1] = fmaxf(__half2float(__low2half(nm_h2_bot)),
                              __half2float(__high2half(nm_h2_bot)));
            nm[mi][0] = fmaxf(nm[mi][0], __shfl_xor_sync(0xffffffff, nm[mi][0], 1));
            nm[mi][0] = fmaxf(nm[mi][0], __shfl_xor_sync(0xffffffff, nm[mi][0], 2));
            nm[mi][1] = fmaxf(nm[mi][1], __shfl_xor_sync(0xffffffff, nm[mi][1], 1));
            nm[mi][1] = fmaxf(nm[mi][1], __shfl_xor_sync(0xffffffff, nm[mi][1], 2));
            nm[mi][0] = fmaxf(nm[mi][0], rmax[mi][0]);
            nm[mi][1] = fmaxf(nm[mi][1], rmax[mi][1]);
            // v79 lever 4: log2-space → exp2f instead of __expf.
            rsc[mi][0] = exp2f(rmax[mi][0] - nm[mi][0]);
            rsc[mi][1] = exp2f(rmax[mi][1] - nm[mi][1]);
        }

        // Rescale Or by per-(mi,side) factor.
#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++)
        {
            __half2 h2_rsc0 = __float2half2_rn(rsc[mi][0]);
            __half2 h2_rsc1 = __float2half2_rn(rsc[mi][1]);
#pragma unroll
            for (int t = 0; t < 16; t++)
            {
                __half2 v0 = *reinterpret_cast<__half2 *>(&Or_p[t][mi][0]);
                __half2 v1 = *reinterpret_cast<__half2 *>(&Or_p[t][mi][1]);
                v0 = __hmul2(v0, h2_rsc0);
                v1 = __hmul2(v1, h2_rsc1);
                Or_p[t][mi][0] = *reinterpret_cast<uint32_t *>(&v0);
                Or_p[t][mi][1] = *reinterpret_cast<uint32_t *>(&v1);
            }
        }
#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++) {
            rmax[mi][0] = nm[mi][0];
            rmax[mi][1] = nm[mi][1];
        }

        // v79b: P stored as __half2 (16 b32 regs total vs v79's 64 FP32 = save ~48 regs).
        // ns sum stays FP32 to avoid f16 accumulator drift on 32-element row sums.
        // v121r R1: P-fusion. Убираем P_top/P_bot[8][2] (=32 regs). Sync переезжает ВВЕРХ
        // до softmax-loop: гейт smP-региона свободен после PV прошлой iter. STS smP
        // делается в той же nt-итерации, что и вычисление P. P_top/P_bot живут только
        // в локальном scope одной (nt, mi).
        __syncthreads();  // гейт smP пуст (PV прошлой iter done)

        float ns[M_TILES][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
        const int smP_g0_base_w = (mrb + gid) * FA_BC;
        const int smP_g8_base_w = smP_g0_base_w + 8 * FA_BC;
#pragma unroll
        for (int nt = 0; nt < 8; nt++)
        {
            int col0 = nt * 8 + tid * 2;
            const int col0_xor = (((col0 >> 4) ^ gid3) << 4) + (col0 & 15);
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                // v121r R2: Sr уже __half2. rmax float → broadcast в half2 → h2sub.
                __half2 rmax_top_h2 = __float2half2_rn(rmax[mi][0]);
                __half2 rmax_bot_h2 = __float2half2_rn(rmax[mi][1]);
                __half2 d_top = __hsub2(Sr[nt][mi][0], rmax_top_h2);
                __half2 d_bot = __hsub2(Sr[nt][mi][1], rmax_bot_h2);
                uint32_t p_top_u, p_bot_u;
                asm("ex2.approx.f16x2 %0, %1;"
                    : "=r"(p_top_u) : "r"(*reinterpret_cast<uint32_t *>(&d_top)));
                asm("ex2.approx.f16x2 %0, %1;"
                    : "=r"(p_bot_u) : "r"(*reinterpret_cast<uint32_t *>(&d_bot)));
                __half2 P_top_loc = *reinterpret_cast<__half2 *>(&p_top_u);
                __half2 P_bot_loc = *reinterpret_cast<__half2 *>(&p_bot_u);
                ns[mi][0] += __low2float(P_top_loc) + __high2float(P_top_loc);
                ns[mi][1] += __low2float(P_bot_loc) + __high2float(P_bot_loc);
                // STS немедленно: P_top_loc/P_bot_loc мертвы после этой строки.
                const int mi_off = mi * 1024;
                uint16_t fp8x2_top = fp16x2_to_e4m3x2(p_top_u);
                uint16_t fp8x2_bot = fp16x2_to_e4m3x2(p_bot_u);
                *(uint16_t *)&smP[smP_g0_base_w + mi_off + col0_xor] = fp8x2_top;
                *(uint16_t *)&smP[smP_g8_base_w + mi_off + col0_xor] = fp8x2_bot;
            }
        }
#pragma unroll
        for (int mi = 0; mi < M_TILES; mi++)
        {
            ns[mi][0] += __shfl_xor_sync(0xffffffff, ns[mi][0], 1);
            ns[mi][0] += __shfl_xor_sync(0xffffffff, ns[mi][0], 2);
            ns[mi][1] += __shfl_xor_sync(0xffffffff, ns[mi][1], 1);
            ns[mi][1] += __shfl_xor_sync(0xffffffff, ns[mi][1], 2);
            rsexp[mi][0] = rsexp[mi][0] * rsc[mi][0] + ns[mi][0];
            rsexp[mi][1] = rsexp[mi][1] * rsc[mi][1] + ns[mi][1];
        }
        __syncthreads();  // гейт smP видим всем варпам

        // v121 step1b: PV hoisting:
        //   smV_T (swz_byte_smvt, stride 68): row=nt*8+gid → row&3 = gid3 const.
        //     row*68 = 544*nt + gid_smvt_base. base = gid_smvt_base + xor_part.
        //   smP LDS (swz_byte_bc, stride 64): row=mrb+mi*16+gid → row&3 = gid3 const.
        //     row*64 = (mrb+gid)*64 + 1024*mi. base_g0 = (mrb+gid)*64 + xor_part.
        const int smP_row_g0_base = (mrb + gid) * FA_BC;
        const int smP_row_g8_base = smP_row_g0_base + 8 * FA_BC;
        // === PV ks=0 batch ===
        {
            int cl = tid * 4, ch = cl + 16;
            const int base_b0 = gid_smvt_base + (((cl >> 4) ^ gid3) << 4) + (cl & 15);
            const int base_b1 = gid_smvt_base + (((ch >> 4) ^ gid3) << 4) + (ch & 15);
            const int smP_xor_cl = (((cl >> 4) ^ gid3) << 4) + (cl & 15);
            const int smP_xor_ch = (((ch >> 4) ^ gid3) << 4) + (ch & 15);
            uint32_t Pr0[M_TILES][4];
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                int mi_off = mi * 1024;  // 16 row * 64 stride
                Pr0[mi][0] = *(uint32_t *)&smP[smP_row_g0_base + mi_off + smP_xor_cl];
                Pr0[mi][1] = *(uint32_t *)&smP[smP_row_g8_base + mi_off + smP_xor_cl];
                Pr0[mi][2] = *(uint32_t *)&smP[smP_row_g0_base + mi_off + smP_xor_ch];
                Pr0[mi][3] = *(uint32_t *)&smP[smP_row_g8_base + mi_off + smP_xor_ch];
            }
#pragma unroll
            for (int nt = 0; nt < 16; nt++)
            {
                uint32_t b0 = *(uint32_t *)&smV_T[base_b0 + 544 * nt];
                uint32_t b1 = *(uint32_t *)&smV_T[base_b1 + 544 * nt];
#pragma unroll
                for (int mi = 0; mi < M_TILES; mi++)
                {
                    mma_fp8_f16(Or_p[nt][mi][0], Or_p[nt][mi][1],
                                Pr0[mi][0], Pr0[mi][1], Pr0[mi][2], Pr0[mi][3],
                                b0, b1, Or_p[nt][mi][0], Or_p[nt][mi][1]);
                }
            }
        }
        // === PV ks=1 batch ===
        {
            int cl = 32 + tid * 4, ch = cl + 16;
            const int base_b0 = gid_smvt_base + (((cl >> 4) ^ gid3) << 4) + (cl & 15);
            const int base_b1 = gid_smvt_base + (((ch >> 4) ^ gid3) << 4) + (ch & 15);
            const int smP_xor_cl = (((cl >> 4) ^ gid3) << 4) + (cl & 15);
            const int smP_xor_ch = (((ch >> 4) ^ gid3) << 4) + (ch & 15);
            uint32_t Pr1[M_TILES][4];
#pragma unroll
            for (int mi = 0; mi < M_TILES; mi++)
            {
                int mi_off = mi * 1024;
                Pr1[mi][0] = *(uint32_t *)&smP[smP_row_g0_base + mi_off + smP_xor_cl];
                Pr1[mi][1] = *(uint32_t *)&smP[smP_row_g8_base + mi_off + smP_xor_cl];
                Pr1[mi][2] = *(uint32_t *)&smP[smP_row_g0_base + mi_off + smP_xor_ch];
                Pr1[mi][3] = *(uint32_t *)&smP[smP_row_g8_base + mi_off + smP_xor_ch];
            }
#pragma unroll
            for (int nt = 0; nt < 16; nt++)
            {
                uint32_t b0 = *(uint32_t *)&smV_T[base_b0 + 544 * nt];
                uint32_t b1 = *(uint32_t *)&smV_T[base_b1 + 544 * nt];
#pragma unroll
                for (int mi = 0; mi < M_TILES; mi++)
                {
                    mma_fp8_f16(Or_p[nt][mi][0], Or_p[nt][mi][1],
                                Pr1[mi][0], Pr1[mi][1], Pr1[mi][2], Pr1[mi][3],
                                b0, b1, Or_p[nt][mi][0], Or_p[nt][mi][1]);
                }
            }
        }
        // v79 lever 2: end-of-iter __syncthreads removed. Was needed in v69 to gate the
        // end-of-iter V cp.async vs PV's smP reads, but v78 moved V cp.async to mid-iter
        // → no SMEM writes after PV in this iter. Next iter's cpa_wait + sync at line 278-279
        // synchronizes all warps before transpose_v writes smV_T.
    }

#pragma unroll
    for (int mi = 0; mi < M_TILES; mi++)
    {
        float li0 = (rsexp[mi][0] > 0) ? v_descale / rsexp[mi][0] : 0.0f;
        float li1 = (rsexp[mi][1] > 0) ? v_descale / rsexp[mi][1] : 0.0f;
        int mr = mrb + mi * 16;
        int gr0 = qs + mr + gid, gr8 = gr0 + 8;
#pragma unroll
        for (int nt = 0; nt < 16; nt++)
        {
            int c0 = nt * 8 + tid * 2, c1 = c0 + 1;
            __half2 v0 = *reinterpret_cast<__half2 *>(&Or_p[nt][mi][0]);
            __half2 v1 = *reinterpret_cast<__half2 *>(&Or_p[nt][mi][1]);
            float O0 = __half2float(__low2half(v0)) * li0;
            float O1 = __half2float(__high2half(v0)) * li0;
            float O2 = __half2float(__low2half(v1)) * li1;
            float O3 = __half2float(__high2half(v1)) * li1;
            if (gr0 < seq_len && c0 < head_dim) Oh[gr0 * head_dim + c0] = __float2half(O0);
            if (gr0 < seq_len && c1 < head_dim) Oh[gr0 * head_dim + c1] = __float2half(O1);
            if (gr8 < seq_len && c0 < head_dim) Oh[gr8 * head_dim + c0] = __float2half(O2);
            if (gr8 < seq_len && c1 < head_dim) Oh[gr8 * head_dim + c1] = __float2half(O3);
        }
    }
}

#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }} while(0)

static inline uint8_t float_to_e4m3(float f)
{
    if (f != f) return 0x7Fu;
    int sign = (f < 0.0f) ? 1 : 0;
    float af = fabsf(f);
    if (af > 448.0f) return sign ? 0xFEu : 0x7Eu;
    if (af < 1.953125e-3f) return sign ? 0x80u : 0x00u;
    int eu = (int)floorf(log2f(af));
    float mf = af / ldexpf(1.0f, eu) - 1.0f;
    int m3 = (int)(mf * 8.0f + 0.5f);
    if (m3 >= 8) { m3 = 0; eu++; }
    int eb = eu + 7;
    if (eb < 1) {
        int ms = (int)(af / ldexpf(1.0f, -9) + 0.5f);
        if (ms > 7) ms = 7;
        return (uint8_t)((sign << 7) | (ms & 7));
    }
    if (eb > 15) eb = 15;
    return (uint8_t)((sign << 7) | (eb << 3) | (m3 & 7));
}
static inline float e4m3_to_float(uint8_t v)
{
    int s = (v >> 7) & 1, e = (v >> 3) & 0xF, m = v & 7;
    if (e == 0xF && m == 7) return nanf("");
    float r = (e == 0) ? ldexpf((float)m, -9) : ldexpf(1.0f + m / 8.0f, e - 7);
    return s ? -r : r;
}
static inline float fp16f(uint16_t h)
{
    __half hv; memcpy(&hv, &h, 2); return __half2float(hv);
}

void cpu_attention_fp8(
    const uint8_t *Q, const uint8_t *K, const uint8_t *V,
    float *O_out, int bh, int sl, int hd, int causal, int window = 0)
{
    float scale = 1.0f / sqrtf((float)hd);
    int hs = sl * hd;
    for (int h = 0; h < bh; h++)
    {
        const uint8_t *Qh = Q + h * hs;
        const uint8_t *Kh = K + h * hs;
        const uint8_t *Vh = V + h * hs;
        float *Oh = O_out + h * hs;
        for (int q = 0; q < sl; q++)
        {
            int kv_max = causal ? (q + 1) : sl;
            // Sliding window: K range = [max(0, q - window + 1), q] for causal.
            int kv_min = (window > 0 && q + 1 > window) ? (q - window + 1) : 0;
            float *P = (float *)malloc(sizeof(float) * sl);
            float rmax = -1e30f;
            for (int k = kv_min; k < kv_max; k++)
            {
                float s = 0;
                for (int d = 0; d < hd; d++)
                    s += e4m3_to_float(Qh[q * hd + d]) * e4m3_to_float(Kh[k * hd + d]);
                P[k] = s * scale;
                if (P[k] > rmax) rmax = P[k];
            }
            float rsum = 0;
            for (int k = kv_min; k < kv_max; k++)
            {
                P[k] = expf(P[k] - rmax);
                rsum += P[k];
            }
            for (int k = kv_min; k < kv_max; k++) P[k] /= rsum;
            for (int d = 0; d < hd; d++)
            {
                float o = 0;
                for (int k = kv_min; k < kv_max; k++)
                    o += P[k] * e4m3_to_float(Vh[k * hd + d]);
                Oh[q * hd + d] = o;
            }
            free(P);
        }
    }
}
