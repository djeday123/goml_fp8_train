// R1c: kernel_dq_new — dS_nat consumer for gradient dQ.
//   dQ[i][d] = sum_j dS[i,j] * K[j,d]
//   MMA m16n8k32 row.col (FP8×FP8→FP16 acc):
//     A [M=i][K=j] row-major = dS_nat  (Br rows × Bc cols)
//     B [K=j][N=d] col-major = K_T[d][j] row-major, stride KT_STRIDE=68 (mirror sealed AA1)
//   → K loaded row-major, transposed to smK_T (mirror sealed AA1 phase 1.5)
//   → dS_nat loaded from global with ABI-padded row stride (stride_ds = (sl+15)&~15).
//
//   Grid: bh × n_qt (block owns Q-tile).
//   Loop kt: load K + dS tile → transpose K → smK_T → MMA-C.
//   Epilogue: dQ_acc * scale → global (unpack fp16-packed → fp32).
//
//   BIT-EXACT invariants vs sealed AA1 (fa_bwd_dq.cu):
//     1. dQ_acc = FP16x2 packed [NI_DQ=16][2] (same pack as sealed AA1 line 220-222).
//     2. MMA fires kb=0..1 outer, ni=0..15 inner (same order as sealed lines 505-529).
//        fp16-acc non-associative → SAME order = SAME bits.
//     3. K load swizzle XOR: (j_local & 7) << 4 (same as sealed line 256).
//     4. K_T read uses XOR k_xor = l_div4 << 4 (same as sealed line 485).
//     5. K_T write natural stride KT_STRIDE=68 (same as sealed line 495-496).
//     6. Epilogue unpack dQ_acc[ni][0..1] → fp16 halves → scale → store (same as sealed line 548-565).
//   SMDS_STRIDE=80 (vs sealed 68): value at (i,j) is same fp8 byte; MMA reads same 4-byte
//   chunk → same computation. Stride affects only SMEM layout, NOT MMA arithmetic → bit-exact preserved.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fa_bwd_common.cuh"

#define FA_DQN_BC          64
#define FA_DQN_BR          64
#define FA_DQN_HD          128
#define FA_DQN_THREADS     128
#define FA_DQN_KT_STRIDE   68     // mirror sealed AA1 KT stride (natural K_T write layout)
#define FA_DQN_SMDS_STRIDE 80     // 16-aligned for cp.async 16-byte dest + bank-conflict-free MMA read

namespace fa_bwd_dq_new {

__device__ __forceinline__ void mma_m16n8k32_e4m3_f16(
    uint32_t &d0, uint32_t &d1,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t c0, uint32_t c1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1));
}

__global__ void kernel_dq_new(
    const uint8_t * __restrict__ K,        // [bh, sl, hd] FP8
    const uint8_t * __restrict__ dS_nat,   // [bh, sl_i, sl_j (stride_ds)] FP8, from ds_gen
    float         * __restrict__ dQ,       // [bh, sl, hd] FP32
    int bh, int sl, int hd,
    int causal, int window,
    float scale)
{
    constexpr int Bc          = FA_DQN_BC;
    constexpr int Br          = FA_DQN_BR;
    constexpr int Hd          = FA_DQN_HD;
    constexpr int KT_STRIDE   = FA_DQN_KT_STRIDE;
    constexpr int SMDS_STRIDE = FA_DQN_SMDS_STRIDE;
    constexpr int NI_QK       = Bc / 8;          // 8 (K_T transpose ni-range, mirror sealed)
    constexpr int KS_QK       = Hd / 32;         // 4 (K_T transpose ks-range)
    constexpr int NI_DQ       = Hd / 8;          // 16 (MMA-C N-tiles)
    constexpr int KB_DQ       = Bc / 32;         // 2 (MMA-C K-batches)

    const int tid    = threadIdx.x;
    const int wid    = tid >> 5;
    const int lane   = tid & 31;
    const int l_div4 = lane >> 2;
    const int l_mod4 = lane & 3;

    // Swizzle XOR mask for smK reads (lane-constant, mirror sealed AA1 line 149).
    const int k_xor  = l_div4 << 4;

    const int n_qt = (sl + Br - 1) / Br;
    const int b    = blockIdx.x / n_qt;
    const int qt   = blockIdx.x % n_qt;
    if (b >= bh) return;
    const int qt_base = qt * Br;

    // ABI-padded row stride for global dS_nat access (matches ds_gen writer).
    const int stride_ds = (sl + 15) & ~15;

    // SMEM layout: smK_area (K↔K_T aliased) + smdS.
    //   smK_area  = max(Bc*Hd, Hd*KT_STRIDE) = max(8192, 8704) = 8704 B
    //   smdS      = Br * SMDS_STRIDE = 64 * 80 = 5120 B
    //   Total     = 13824 B/block. floor(102400 / (13824+1024)) = 6 blocks/SM (SMEM-limited).
    constexpr int SMK_AREA_BYTES = (Bc * Hd > Hd * KT_STRIDE) ? Bc * Hd : Hd * KT_STRIDE;  // 8704
    extern __shared__ uint8_t smem_raw[];
    uint8_t *smK_area = smem_raw;
    uint8_t *smdS     = smem_raw + SMK_AREA_BYTES;

    // dQ_acc packed FP16x2 [16][2] (mirror sealed AA1).
    uint32_t dQ_acc[NI_DQ][2];
    #pragma unroll
    for (int ni = 0; ni < NI_DQ; ++ni) { dQ_acc[ni][0] = 0u; dQ_acc[ni][1] = 0u; }

    const int n_kt = (sl + Bc - 1) / Bc;
    // Causal-aware KV-skip (mirror sealed AA1 line 238): skip kt > qt.
    const int kt_end = causal ? (qt + 1) : n_kt;
    for (int kt = 0; kt < kt_end; ++kt) {
        const int kt_base = kt * Bc;

        // D5-lite: split cp.async K and dS into two commit_groups для overlap
        //   group_0 = K (larger, 8 KB), issued first
        //   group_1 = dS (smaller, 4 KB), issued second
        //   wait<1> after group_0 → Phase 1.5 overlaps with dS tail-load
        //   wait<0> before MMA-C для полного dS
        // Step A1: cp.async K → smK_area (XOR-swizzled, mirror sealed AA1 line 246-266).
        {
            const uint8_t *Kb = K + (size_t)b * sl * Hd;
            constexpr int CHUNK = 16;
            constexpr int chunks_per_row = Hd / CHUNK;      // 8
            constexpr int total = Bc * chunks_per_row;      // 512
            for (int c = tid; c < total; c += FA_DQN_THREADS) {
                int j_local = c / chunks_per_row;
                int col_byte = (c % chunks_per_row) * CHUNK;
                int j_g = kt_base + j_local;
                int k_xor_row = (j_local & 7) << 4;
                cpa16(&smK_area[j_local * Hd + (col_byte ^ k_xor_row)],
                      &Kb[(size_t)j_g * Hd + col_byte],
                      (j_g < sl) ? CHUNK : 0);
            }
        }
        cpa_commit();      // group_0 = K
        // Step A2: cp.async dS_nat → smdS (stride SMDS_STRIDE=80, 16-aligned).
        //   Global source stride = stride_ds (padded, 16-aligned per 005a ABI).
        //   Per-chunk: 16-byte dest offset = i_local * 80 + col_byte, all 16-aligned.
        //   In-bounds (full chunk): cp.async 16.  Full OOB (i or j full-OOB): cp.async bytes=0 (zeros dest).
        //   Partial (0 < sl - j_g_base < 16): rare (only CANARY-class); STS-zero + per-byte STS.
        {
            const uint8_t *dSb = dS_nat + (size_t)b * sl * stride_ds;
            constexpr int CHUNK = 16;
            constexpr int cpr   = Bc / CHUNK;    // 4
            constexpr int total = Br * cpr;      // 256
            for (int c = tid; c < total; c += FA_DQN_THREADS) {
                int i_local = c / cpr;
                int col_byte = (c % cpr) * CHUNK;
                int i_g = qt_base + i_local;
                int j_g_base = kt_base + col_byte;
                bool i_ok = (i_g < sl);
                int j_avail = i_ok ? (sl - j_g_base) : 0;
                if (j_avail < 0) j_avail = 0;

                if (i_ok && j_avail >= CHUNK) {
                    // Full in-bounds: fast cp.async 16.
                    cpa16(&smdS[i_local * SMDS_STRIDE + col_byte],
                          &dSb[(size_t)i_g * stride_ds + j_g_base], CHUNK);
                } else if (!i_ok || j_avail <= 0) {
                    // Full OOB: cp.async bytes=0 (zeros dest 16 bytes; source uses base dSb — always aligned).
                    cpa16(&smdS[i_local * SMDS_STRIDE + col_byte], dSb, 0);
                } else {
                    // Partial (0 < j_avail < 16, CANARY-only): STS-zero + per-byte load.
                    uint32_t *sp = reinterpret_cast<uint32_t*>(&smdS[i_local * SMDS_STRIDE + col_byte]);
                    sp[0] = 0u; sp[1] = 0u; sp[2] = 0u; sp[3] = 0u;
                    #pragma unroll 1
                    for (int bb = 0; bb < 16; ++bb) {
                        if (bb < j_avail) {
                            smdS[i_local * SMDS_STRIDE + col_byte + bb] =
                                dSb[(size_t)i_g * stride_ds + j_g_base + bb];
                        }
                    }
                }
            }
        }
        cpa_commit();      // group_1 = dS
        cpa_wait<1>();     // wait until 1 group remains: K done, dS in flight
        __syncthreads();   // BARRIER #1: K ready in smK_area

        // Phase 1.5: K → K_T transpose (mirror sealed AA1 lines 474-499).
        //   ks = wid (per-warp), read swizzled smK into kr_lo/kr_hi, then write natural K_T (aliased overwrite).
        {
            const int ks = wid;
            const int k_lo = ks * 32 + l_mod4 * 4 + 0;
            const int k_hi = ks * 32 + l_mod4 * 4 + 16;
            uint32_t kr_lo[NI_QK], kr_hi[NI_QK];
            // Read phase (swizzle-aware LDS on K natural).
            #pragma unroll
            for (int ni = 0; ni < NI_QK; ++ni) {
                int n_K = ni * 8 + l_div4;
                kr_lo[ni] = *reinterpret_cast<uint32_t*>(&smK_area[n_K * Hd + (k_lo ^ k_xor)]);
                kr_hi[ni] = *reinterpret_cast<uint32_t*>(&smK_area[n_K * Hd + (k_hi ^ k_xor)]);
            }
            __syncthreads();   // BARRIER #2: all reads done before aliased overwrite writes
            // Write phase — PACK K_T (dq exchange net, unit-tested 8192/8192, ТЗ 025-b):
            //   64 STS.U8 → 12 SHFL + 16 STS.32 per qt/lane
            //   Group: {c + 4*p' + 16*h}, обмен по l_div4, slot bits (half, ni_hi)
            //   ЗАПРЕТ V[] array — V0-V3 named регистры (LDL/STL детектор в гейтах)
            {
                const int c = l_mod4;
                const int p = l_div4 & 3;
                const int h = l_div4 >> 2;
                const int ks_local = wid;
                #pragma unroll
                for (int slot = 0; slot < 4; ++slot) {
                    const int slot_half  = (slot >> 1) & 1;   // 0=lo, 1=hi
                    const int slot_ni_hi = slot & 1;          // 0=[0..3], 1=[4..7]
                    const int ni_base    = slot_ni_hi * 4;
                    uint32_t W0, W1, W2, W3;
                    if (slot_half == 0) {
                        W0 = kr_lo[ni_base + 0]; W1 = kr_lo[ni_base + 1];
                        W2 = kr_lo[ni_base + 2]; W3 = kr_lo[ni_base + 3];
                    } else {
                        W0 = kr_hi[ni_base + 0]; W1 = kr_hi[ni_base + 1];
                        W2 = kr_hi[ni_base + 2]; W3 = kr_hi[ni_base + 3];
                    }
                    // Phase A — byte-position gather (8 PRMT)
                    uint32_t t01_lo, t01_hi, t23_lo, t23_hi;
                    asm volatile("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(t01_lo) : "r"(W0), "r"(W1));
                    asm volatile("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(t01_hi) : "r"(W0), "r"(W1));
                    asm volatile("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(t23_lo) : "r"(W2), "r"(W3));
                    asm volatile("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(t23_hi) : "r"(W2), "r"(W3));
                    uint32_t G0, G1, G2, G3;
                    asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(G0) : "r"(t01_lo), "r"(t23_lo));
                    asm volatile("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(G1) : "r"(t01_lo), "r"(t23_lo));
                    asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(G2) : "r"(t01_hi), "r"(t23_hi));
                    asm volatile("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(G3) : "r"(t01_hi), "r"(t23_hi));
                    // Phase B — 3 SHFL exchange
                    uint32_t V0 = G0, V1 = G1, V2 = G2, V3 = G3;
                    #pragma unroll
                    for (int r = 1; r <= 3; ++r) {
                        int src_p = (p - r) & 3;
                        int src_lane = c + 4 * src_p + 16 * h;
                        int idx = (p + r) & 3;
                        uint32_t lo_g = (idx & 1) ? G1 : G0;
                        uint32_t hi_g = (idx & 1) ? G3 : G2;
                        uint32_t expose_val = (idx & 2) ? hi_g : lo_g;
                        uint32_t val = __shfl_sync(0xFFFFFFFF, expose_val, src_lane);
                        V0 = (src_p == 0) ? val : V0;
                        V1 = (src_p == 1) ? val : V1;
                        V2 = (src_p == 2) ? val : V2;
                        V3 = (src_p == 3) ? val : V3;
                    }
                    // Phase C — receive PRMT (8)
                    uint32_t u01_lo, u01_hi, u23_lo, u23_hi;
                    asm volatile("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(u01_lo) : "r"(V0), "r"(V1));
                    asm volatile("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(u01_hi) : "r"(V0), "r"(V1));
                    asm volatile("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(u23_lo) : "r"(V2), "r"(V3));
                    asm volatile("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(u23_hi) : "r"(V2), "r"(V3));
                    uint32_t OUT0, OUT1, OUT2, OUT3;
                    asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(OUT0) : "r"(u01_lo), "r"(u23_lo));
                    asm volatile("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(OUT1) : "r"(u01_lo), "r"(u23_lo));
                    asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(OUT2) : "r"(u01_hi), "r"(u23_hi));
                    asm volatile("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(OUT3) : "r"(u01_hi), "r"(u23_hi));
                    // Phase D — 4 STS.32 (dq K_T mapping) + π_V bank-conflict fix
                    //   PI_V(r) = ((r&7)<<2) | (((r>>3)&1)<<1) | ((r>>4)&1) | (r & 0x60)
                    //   CPU-судья 027 pt.2b: 4/4 ASSERT PASS, ST/LD bijection ✓
                    const int base_row = ks_local * 32 + 4 * c + p + 16 * slot_half;
                    #define PI_V(r) ((((r) & 7) << 2) | ((((r) >> 3) & 1) << 1) | (((r) >> 4) & 1) | ((r) & 0x60))
                    const int row_pi = PI_V(base_row);
                    #undef PI_V
                    *reinterpret_cast<uint32_t*>(&smK_area[row_pi * KT_STRIDE + (ni_base + 0) * 8 + 4 * h]) = OUT0;
                    *reinterpret_cast<uint32_t*>(&smK_area[row_pi * KT_STRIDE + (ni_base + 1) * 8 + 4 * h]) = OUT1;
                    *reinterpret_cast<uint32_t*>(&smK_area[row_pi * KT_STRIDE + (ni_base + 2) * 8 + 4 * h]) = OUT2;
                    *reinterpret_cast<uint32_t*>(&smK_area[row_pi * KT_STRIDE + (ni_base + 3) * 8 + 4 * h]) = OUT3;
                }
            }
        }
        cpa_wait<0>();     // D5-lite: wait for dS group_1 (may be already done if hidden by Phase 1.5)
        __syncthreads();   // BARRIER #3: K_T + dS both ready for MMA-C

        // MMA-C: dQ_acc += dS · K_T (mirror sealed AA1 lines 505-530).
        //   A = smdS NATURAL [M=i=Br][K=j=Bc] stride SMDS_STRIDE
        //   B = smK_T stored [N=d][K=j] stride KT_STRIDE (== col-major B[K=j][N=d])
        //   Order: kb outer, ni inner — SAME as sealed AA1, fp16-acc non-associative.
        #pragma unroll
        for (int kb = 0; kb < KB_DQ; ++kb) {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k_j_lo = kb * 32 + l_mod4 * 4 + 0;
            int k_j_hi = kb * 32 + l_mod4 * 4 + 16;

            uint32_t A0 = *reinterpret_cast<uint32_t*>(&smdS[m_lo * SMDS_STRIDE + k_j_lo]);
            uint32_t A1 = *reinterpret_cast<uint32_t*>(&smdS[m_hi * SMDS_STRIDE + k_j_lo]);
            uint32_t A2 = *reinterpret_cast<uint32_t*>(&smdS[m_lo * SMDS_STRIDE + k_j_hi]);
            uint32_t A3 = *reinterpret_cast<uint32_t*>(&smdS[m_hi * SMDS_STRIDE + k_j_hi]);

            #pragma unroll
            for (int ni = 0; ni < NI_DQ; ++ni) {
                int n_d = ni * 8 + l_div4;
                // π_V B-load mirror of Phase D π_V write
                #define PI_V(r) ((((r) & 7) << 2) | ((((r) >> 3) & 1) << 1) | (((r) >> 4) & 1) | ((r) & 0x60))
                int n_d_pi = PI_V(n_d);
                #undef PI_V
                uint32_t B0 = *reinterpret_cast<uint32_t*>(&smK_area[n_d_pi * KT_STRIDE + k_j_lo]);
                uint32_t B1 = *reinterpret_cast<uint32_t*>(&smK_area[n_d_pi * KT_STRIDE + k_j_hi]);

                mma_m16n8k32_e4m3_f16(
                    dQ_acc[ni][0], dQ_acc[ni][1],
                    A0, A1, A2, A3, B0, B1,
                    dQ_acc[ni][0], dQ_acc[ni][1]);
            }
        }
        __syncthreads();   // BARRIER #4: end of kt (smK_area free for next cp.async)
    }

    // Epilogue: unpack packed fp16 dQ_acc → fp32 → *scale → gmem (mirror sealed AA1 lines 536-566).
    {
        int i_local_lo = wid * 16 + l_div4 + 0;
        int i_local_hi = wid * 16 + l_div4 + 8;
        int i_g_lo_out = qt_base + i_local_lo;
        int i_g_hi_out = qt_base + i_local_hi;
        bool i_lo_ok = (i_g_lo_out < sl);
        bool i_hi_ok = (i_g_hi_out < sl);
        float *dQb = dQ + (size_t)b * sl * Hd;

        #pragma unroll
        for (int ni = 0; ni < NI_DQ; ++ni) {
            int d_lo = ni * 8 + l_mod4 * 2 + 0;
            int d_hi = ni * 8 + l_mod4 * 2 + 1;
            __half2 lo_h2 = *reinterpret_cast<__half2*>(&dQ_acc[ni][0]);
            __half2 hi_h2 = *reinterpret_cast<__half2*>(&dQ_acc[ni][1]);
            float lo_d_lo = (float)__low2half (lo_h2);
            float lo_d_hi = (float)__high2half(lo_h2);
            float hi_d_lo = (float)__low2half (hi_h2);
            float hi_d_hi = (float)__high2half(hi_h2);
            if (i_lo_ok) {
                dQb[(size_t)i_g_lo_out * Hd + d_lo] = lo_d_lo * scale;
                dQb[(size_t)i_g_lo_out * Hd + d_hi] = lo_d_hi * scale;
            }
            if (i_hi_ok) {
                dQb[(size_t)i_g_hi_out * Hd + d_lo] = hi_d_lo * scale;
                dQb[(size_t)i_g_hi_out * Hd + d_hi] = hi_d_hi * scale;
            }
        }
    }
}

void launch_dq_new(
    const uint8_t *K, const uint8_t *dS_nat,
    float *dQ,
    int bh, int sl, int hd,
    int causal, int window,
    float scale, cudaStream_t stream)
{
    if (hd != FA_DQN_HD) {
        fprintf(stderr, "fa_bwd_dq_new: hd=%d, expected %d\n", hd, FA_DQN_HD);
        exit(1);
    }
    const int Bc = FA_DQN_BC;
    const int Br = FA_DQN_BR;
    const int n_qt = (sl + Br - 1) / Br;
    const int grid = bh * n_qt;
    constexpr int SMK_AREA = (FA_DQN_BC * FA_DQN_HD > FA_DQN_HD * FA_DQN_KT_STRIDE)
                             ? FA_DQN_BC * FA_DQN_HD
                             : FA_DQN_HD * FA_DQN_KT_STRIDE;
    const int smem_bytes = SMK_AREA + FA_DQN_BR * FA_DQN_SMDS_STRIDE;   // 8704 + 5120 = 13824
    cudaFuncSetAttribute(kernel_dq_new,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    kernel_dq_new<<<grid, FA_DQN_THREADS, smem_bytes, stream>>>(
        K, dS_nat, dQ, bh, sl, hd, causal, window, scale);
}

} // namespace fa_bwd_dq_new
