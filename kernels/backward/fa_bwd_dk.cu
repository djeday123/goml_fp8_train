// =====================================================================
//  fa_bwd_dk.cu — B3.2 final body: dK kernel + D-precompute kernel.
//
//  Дизайн (locked per Vugar's reviews):
//  - Split от dV (P1 cert intact).
//  - MMA #1: m16n8k16 row.col.f32.f16.f16.f32 (FP16 dO × FP16 V → FP32 acc).
//    V cast e4m3 → FP16 на лету через cvt.rn.f16x2.e4m3x2.
//  - MMA #2: m16n8k32 row.col.f32.e4m3.e4m3.f32 (FP8 dS_q × FP8 Q → FP32 acc).
//    dS quantize FP16 → e4m3 в регистрах перед smdST write.
//  - Geometry: Bc=64, Br=64, Hd=128, 128 threads, 4 warps. M_TILES_per_warp=1.
//  - SMEM 45 KB → 2 blocks/SM.
//  - Aliasing smQ ↔ smQ_T (8.5 KB region, stride-68 padding for 0-conflict
//    write/read per probe analysis).
//  - Transpose pass между step B и step G (per-qt-load, не per-inner-iter).
//  - cp.async staging (Q, dO, K, V) — DMA, не STS.
//  - D vector input от separate precompute kernel (β, ~1-3 ms overhead).
//
//  Barriers per qt: 4 (after A, mid-T, end-T, end-of-qt). Per KV-tile sl=8192:
//  4 × 128 = 512. vs dV P1 384 = +128 from transpose pass. "Сотни, не тысячи."
//
//  Tensor pipe idle во время transpose pass — accepted cost (NCu замерит).
//  Double-buffer Q (P3-style) для overlap НЕ делаем (P3 lesson: +8 KB = 1 block).
// =====================================================================

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fa_bwd_common.cuh"

#define FA_DK_BC        64
#define FA_DK_BR        64
#define FA_DK_HD        128
#define FA_DK_THREADS   128
#define FA_DK_QT_STRIDE 68   // padding for 0-conflict smQ_T (forward smV_T proven)
#define FA_DK_SMDST_STRIDE 68  // padding for step F scatter STS: 16-way → 4-way bank conflict

namespace fa_bwd_dk {

// =====================================================================
// PTX MMA wrappers + e4m3↔f16 cvt
// =====================================================================
__device__ __forceinline__ void mma_m16n8k16_f32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void mma_m16n8k32_e4m3_f32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint16_t fp8x2) {
    uint32_t r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(fp8x2));
    return r;
}

// =====================================================================
// dK kernel
//   Inputs: Q, K, V (FP8 e4m3), dO (FP16), L, D (FP32 vectors), scale.
//   D = sum_d O[i,d]*dO[i,d] — precomputed by D-kernel (separate pass).
//   Output: dK (FP32) [bh, sl, hd].
// =====================================================================
__global__ void kernel_dk(
    const uint8_t * __restrict__ Q,
    const uint8_t * __restrict__ K,
    const uint8_t * __restrict__ V,
    const __half  * __restrict__ dO_g,
    const float   * __restrict__ L,
    const float   * __restrict__ D,
    float         * __restrict__ dK,
    int bh, int sl, int hd,
    int causal, int window,
    float scale)
{
    constexpr int Bc        = FA_DK_BC;
    constexpr int Br        = FA_DK_BR;
    constexpr int Hd        = FA_DK_HD;
    constexpr int QT_STRIDE = FA_DK_QT_STRIDE;
    constexpr int NI_QK     = Bc / 8;          // 8 N-tiles for Q·K^T
    constexpr int NI_DP     = Bc / 8;          // 8 N-tiles for dP MMA #1
    constexpr int NI_DK     = Hd / 8;          // 16 N-tiles for dK MMA #2
    constexpr int KS_QK     = Hd / 32;         // 4 ks-batches FP8 m16n8k32
    constexpr int KS_DP     = Hd / 16;         // 8 ks-batches FP16 m16n8k16
    constexpr int KB_DK     = Br / 32;         // 2 k-batches FP8 m16n8k32

    const int tid    = threadIdx.x;
    const int wid    = tid >> 5;
    const int lane   = tid & 31;
    const int l_div4 = lane >> 2;
    const int l_mod4 = lane & 3;

    const int n_kt = (sl + Bc - 1) / Bc;
    const int b    = blockIdx.x / n_kt;
    const int kt   = blockIdx.x % n_kt;
    if (b >= bh) return;

    // ---- SMEM layout (45 KB, 2 blocks/SM) ----
    //   smK 8K + smV 8K + smQ_aliased 8.5K + smdO 16K + smdST 4K + smL+smD 0.5K
    constexpr int SMQ_AREA_BYTES = Hd * QT_STRIDE;   // 128 * 68 = 8704 (max of Q row 8K and Q_T padded 8.5K)
    extern __shared__ uint8_t smem_raw[];
    uint8_t *smK       = smem_raw;
    uint8_t *smV       = smK    + Bc * Hd;                                       // 8K
    uint8_t *smQ_area  = smV    + Bc * Hd;                                       // 8704 bytes (aliased)
    __half  *smdO      = reinterpret_cast<__half*>(smQ_area + SMQ_AREA_BYTES);   // 16K
    uint8_t *smdST     = reinterpret_cast<uint8_t*>(smdO + Br * Hd);             // 4.25K (stride 68)
    float   *smL       = reinterpret_cast<float*>(smdST + Bc * FA_DK_SMDST_STRIDE); // 256B
    float   *smD       = smL + Br;                                                // 256B

    // ---- Warmup K + V via cp.async ----
    {
        const uint8_t *Kb = K + b * sl * Hd;
        constexpr int CHUNK = 16;
        constexpr int chunks_per_row = Hd / CHUNK;       // 8
        constexpr int total = Bc * chunks_per_row;       // 512
        for (int c = tid; c < total; c += FA_DK_THREADS) {
            int j_local  = c / chunks_per_row;
            int col_byte = (c % chunks_per_row) * CHUNK;
            int j_g      = kt * Bc + j_local;
            cpa16(&smK[swz_byte(j_local, col_byte)],
                  &Kb[j_g * Hd + col_byte],
                  (j_g < sl) ? CHUNK : 0);
        }
        const uint8_t *Vb = V + b * sl * Hd;
        for (int c = tid; c < total; c += FA_DK_THREADS) {
            int j_local  = c / chunks_per_row;
            int col_byte = (c % chunks_per_row) * CHUNK;
            int j_g      = kt * Bc + j_local;
            // smV swizzle: same XOR ((row & 7) << 4) formula as dQ/dV smV/smK swizzle
            int V_xor = (j_local & 7) << 4;
            cpa16(&smV[j_local * Hd + (col_byte ^ V_xor)],
                  &Vb[j_g * Hd + col_byte],
                  (j_g < sl) ? CHUNK : 0);
        }
        cpa_commit();
        cpa_wait<0>();
    }
    __syncthreads();

    // ---- dK_acc init (FP32 in registers) ----
    float dK_acc[NI_DK][4];
    #pragma unroll
    for (int ni = 0; ni < NI_DK; ++ni)
        #pragma unroll
        for (int s = 0; s < 4; ++s) dK_acc[ni][s] = 0.0f;

    const int n_qt = (sl + Br - 1) / Br;
    // Causal-aware KV-skip: skip qt where ALL pairs (i, j) are masked (kt > qt).
    // Tile fully masked iff min(j_tile)=kt*Bc > max(i_tile)=qt*Br + Br - 1, i.e., kt > qt for Br=Bc.
    // Diagonal (kt==qt) goes through with mask. Non-causal: qt_start=0, no change.
    const int qt_start = causal ? kt : 0;
    for (int qt = qt_start; qt < n_qt; ++qt) {
        const int qt_base = qt * Br;

        // ===== step A: cp.async Q, dO, L, D =====
        {
            const uint8_t *Qb = Q    + b * sl * Hd;
            const __half  *dB = dO_g + b * sl * Hd;
            // Q row-major into smQ_area (will be transposed later)
            constexpr int CHUNK = 16;
            constexpr int Q_cpr   = Hd / CHUNK;
            constexpr int Q_total = Br * Q_cpr;
            for (int c = tid; c < Q_total; c += FA_DK_THREADS) {
                int i_local  = c / Q_cpr;
                int col_byte = (c % Q_cpr) * CHUNK;
                int i_g      = qt_base + i_local;
                // smQ_area swizzle: XOR ((row & 7) << 4) — same formula as smK/smV
                int Q_xor = (i_local & 7) << 4;
                cpa16(&smQ_area[i_local * Hd + (col_byte ^ Q_xor)],
                      &Qb[i_g * Hd + col_byte],
                      (i_g < sl) ? CHUNK : 0);
            }
            // dO FP16 [Br, hd] row-major
            constexpr int dO_bpr  = Hd * 2;
            constexpr int dO_cpr  = dO_bpr / CHUNK;
            constexpr int dO_total = Br * dO_cpr;
            uint8_t *smdO_b = reinterpret_cast<uint8_t*>(smdO);
            const uint8_t *dB_b = reinterpret_cast<const uint8_t*>(dB);
            for (int c = tid; c < dO_total; c += FA_DK_THREADS) {
                int i_local  = c / dO_cpr;
                int col_byte = (c % dO_cpr) * CHUNK;
                int i_g      = qt_base + i_local;
                int dO_xor   = (i_local & 7) << 4;    // smdO-swizzle (byte space, stride 256B)
                cpa16(smdO_b + i_local * dO_bpr + (col_byte ^ dO_xor),
                      dB_b   + i_g * dO_bpr + col_byte,
                      (i_g < sl) ? CHUNK : 0);
            }
            // L + D: sync small loads
            if (tid < Br) {
                int i_g = qt_base + tid;
                smL[tid] = (i_g < sl) ? L[b * sl + i_g] : 0.0f;
                smD[tid] = (i_g < sl) ? D[b * sl + i_g] : 0.0f;
            }
            cpa_commit();
            cpa_wait<0>();
        }
        __syncthreads();    // BARRIER #1

        // ===== step B: Q·K^T MMA → Sr (FP8 e4m3 m16n8k32 → f16 acc) =====
        uint32_t Qr[KS_QK][4];
        {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            #pragma unroll
            for (int ks = 0; ks < KS_QK; ++ks) {
                int k_lo = ks * 32 + l_mod4 * 4 + 0;
                int k_hi = ks * 32 + l_mod4 * 4 + 16;
                // smQ-swizzle read: (m_lo&7) = (m_hi&7) = l_div4 → XOR = l_div4 << 4
                const int Q_xor_rd = l_div4 << 4;
                Qr[ks][0] = *reinterpret_cast<uint32_t*>(&smQ_area[m_lo * Hd + (k_lo ^ Q_xor_rd)]);
                Qr[ks][1] = *reinterpret_cast<uint32_t*>(&smQ_area[m_hi * Hd + (k_lo ^ Q_xor_rd)]);
                Qr[ks][2] = *reinterpret_cast<uint32_t*>(&smQ_area[m_lo * Hd + (k_hi ^ Q_xor_rd)]);
                Qr[ks][3] = *reinterpret_cast<uint32_t*>(&smQ_area[m_hi * Hd + (k_hi ^ Q_xor_rd)]);
            }
        }
        uint32_t Sr[NI_QK][2];
        #pragma unroll
        for (int ni = 0; ni < NI_QK; ++ni) { Sr[ni][0] = 0u; Sr[ni][1] = 0u; }
        #pragma unroll
        for (int ks = 0; ks < KS_QK; ++ks) {
            int k_lo = ks * 32 + l_mod4 * 4 + 0;
            int k_hi = ks * 32 + l_mod4 * 4 + 16;
            #pragma unroll
            for (int ni = 0; ni < NI_QK; ++ni) {
                int n_K = ni * 8 + l_div4;
                uint32_t Kr0 = *reinterpret_cast<uint32_t*>(&smK[swz_byte(n_K, k_lo)]);
                uint32_t Kr1 = *reinterpret_cast<uint32_t*>(&smK[swz_byte(n_K, k_hi)]);
                mma_fp8_f16(Sr[ni][0], Sr[ni][1],
                            Qr[ks][0], Qr[ks][1], Qr[ks][2], Qr[ks][3],
                            Kr0, Kr1,
                            Sr[ni][0], Sr[ni][1]);
            }
        }

        // ===== step C: softmax → Pr (FP16 in regs, no smP write) =====
        const float L_lo = smL[wid * 16 + l_div4 + 0];
        const float L_hi = smL[wid * 16 + l_div4 + 8];
        const float D_lo = smD[wid * 16 + l_div4 + 0];
        const float D_hi = smD[wid * 16 + l_div4 + 8];
        const int i_g_lo = qt_base + wid * 16 + l_div4 + 0;
        const int i_g_hi = qt_base + wid * 16 + l_div4 + 8;
        const bool i_lo_oob = (i_g_lo >= sl);
        const bool i_hi_oob = (i_g_hi >= sl);

        uint32_t Pr[NI_QK][2];
        #pragma unroll
        for (int ni = 0; ni < NI_QK; ++ni) {
            __half2 s_lo_h2 = *reinterpret_cast<__half2*>(&Sr[ni][0]);
            __half2 s_hi_h2 = *reinterpret_cast<__half2*>(&Sr[ni][1]);
            float s_mlo_nlo = (float)__low2half (s_lo_h2);
            float s_mlo_nhi = (float)__high2half(s_lo_h2);
            float s_mhi_nlo = (float)__low2half (s_hi_h2);
            float s_mhi_nhi = (float)__high2half(s_hi_h2);

            int j_local_lo = ni * 8 + l_mod4 * 2 + 0;
            int j_local_hi = ni * 8 + l_mod4 * 2 + 1;
            int j_g_lo = kt * Bc + j_local_lo;
            int j_g_hi = kt * Bc + j_local_hi;
            bool j_lo_oob = (j_g_lo >= sl);
            bool j_hi_oob = (j_g_hi >= sl);

            auto mask = [&](int i_g, bool i_o, int j_g, bool j_o) -> bool {
                if (i_o || j_o)                           return true;
                if (causal && j_g > i_g)                  return true;
                if (window > 0 && j_g < i_g + 1 - window) return true;
                return false;
            };

            float p00 = mask(i_g_lo, i_lo_oob, j_g_lo, j_lo_oob) ? 0.0f
                       : __expf(scale * s_mlo_nlo - L_lo);
            float p01 = mask(i_g_lo, i_lo_oob, j_g_hi, j_hi_oob) ? 0.0f
                       : __expf(scale * s_mlo_nhi - L_lo);
            float p10 = mask(i_g_hi, i_hi_oob, j_g_lo, j_lo_oob) ? 0.0f
                       : __expf(scale * s_mhi_nlo - L_hi);
            float p11 = mask(i_g_hi, i_hi_oob, j_g_hi, j_hi_oob) ? 0.0f
                       : __expf(scale * s_mhi_nhi - L_hi);

            __half2 p_lo = __halves2half2(__float2half(p00), __float2half(p01));
            __half2 p_hi = __halves2half2(__float2half(p10), __float2half(p11));
            Pr[ni][0] = *reinterpret_cast<uint32_t*>(&p_lo);
            Pr[ni][1] = *reinterpret_cast<uint32_t*>(&p_hi);
        }

        // ===== step D: dP MMA #1 (dO·V^T) → dPr F32 acc =====
        float dPr[NI_DP][4];
        #pragma unroll
        for (int ni = 0; ni < NI_DP; ++ni)
            #pragma unroll
            for (int s = 0; s < 4; ++s) dPr[ni][s] = 0.0f;

        #pragma unroll
        for (int ks = 0; ks < KS_DP; ++ks) {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k_lo = ks * 16 + l_mod4 * 2 + 0;
            int k_hi = ks * 16 + l_mod4 * 2 + 8;

            // smdO-swizzle read: __half* indexed, element-space XOR = l_div4 << 3
            const int dO_xor_el = l_div4 << 3;
            uint32_t A0 = *reinterpret_cast<uint32_t*>(&smdO[m_lo * Hd + (k_lo ^ dO_xor_el)]);
            uint32_t A1 = *reinterpret_cast<uint32_t*>(&smdO[m_hi * Hd + (k_lo ^ dO_xor_el)]);
            uint32_t A2 = *reinterpret_cast<uint32_t*>(&smdO[m_lo * Hd + (k_hi ^ dO_xor_el)]);
            uint32_t A3 = *reinterpret_cast<uint32_t*>(&smdO[m_hi * Hd + (k_hi ^ dO_xor_el)]);

            #pragma unroll
            for (int ni = 0; ni < NI_DP; ++ni) {
                int n = ni * 8 + l_div4;
                // smV-swizzle read: (n & 7) = l_div4 → byte XOR = l_div4 << 4
                const int V_xor_rd = l_div4 << 4;
                uint16_t v0_u16 = *reinterpret_cast<uint16_t*>(&smV[n * Hd + (k_lo ^ V_xor_rd)]);
                uint16_t v1_u16 = *reinterpret_cast<uint16_t*>(&smV[n * Hd + (k_hi ^ V_xor_rd)]);
                uint32_t B0 = e4m3x2_to_f16x2(v0_u16);
                uint32_t B1 = e4m3x2_to_f16x2(v1_u16);

                mma_m16n8k16_f32(
                    dPr[ni][0], dPr[ni][1], dPr[ni][2], dPr[ni][3],
                    A0, A1, A2, A3, B0, B1,
                    dPr[ni][0], dPr[ni][1], dPr[ni][2], dPr[ni][3]);
            }
        }

        // ===== step E+F: dS = P·(dP - D), quantize → smdST transposed =====
        #pragma unroll
        for (int ni = 0; ni < NI_DP; ++ni) {
            __half2 p_lo_h2 = *reinterpret_cast<__half2*>(&Pr[ni][0]);
            __half2 p_hi_h2 = *reinterpret_cast<__half2*>(&Pr[ni][1]);
            float p_mlo_nlo = (float)__low2half (p_lo_h2);
            float p_mlo_nhi = (float)__high2half(p_lo_h2);
            float p_mhi_nlo = (float)__low2half (p_hi_h2);
            float p_mhi_nhi = (float)__high2half(p_hi_h2);

            // dS = P · (dP - D_i)
            float dS_mlo_nlo = p_mlo_nlo * (dPr[ni][0] - D_lo);
            float dS_mlo_nhi = p_mlo_nhi * (dPr[ni][1] - D_lo);
            float dS_mhi_nlo = p_mhi_nlo * (dPr[ni][2] - D_hi);
            float dS_mhi_nhi = p_mhi_nhi * (dPr[ni][3] - D_hi);

            __half2 ds_lo = __halves2half2(
                __float2half(dS_mlo_nlo), __float2half(dS_mlo_nhi));
            __half2 ds_hi = __halves2half2(
                __float2half(dS_mhi_nlo), __float2half(dS_mhi_nhi));
            uint32_t ds_lo_u32 = *reinterpret_cast<uint32_t*>(&ds_lo);
            uint32_t ds_hi_u32 = *reinterpret_cast<uint32_t*>(&ds_hi);
            uint16_t ds_lo_fp8 = fp16x2_to_e4m3x2(ds_lo_u32);
            uint16_t ds_hi_fp8 = fp16x2_to_e4m3x2(ds_hi_u32);

            uint8_t b00 = ds_lo_fp8 & 0xFF;
            uint8_t b01 = (ds_lo_fp8 >> 8) & 0xFF;
            uint8_t b10 = ds_hi_fp8 & 0xFF;
            uint8_t b11 = (ds_hi_fp8 >> 8) & 0xFF;

            int i_local_lo = wid * 16 + l_div4 + 0;
            int i_local_hi = wid * 16 + l_div4 + 8;
            int j_local_lo = ni * 8 + l_mod4 * 2 + 0;
            int j_local_hi = ni * 8 + l_mod4 * 2 + 1;

            smdST[j_local_lo * FA_DK_SMDST_STRIDE + i_local_lo] = b00;
            smdST[j_local_hi * FA_DK_SMDST_STRIDE + i_local_lo] = b01;
            smdST[j_local_lo * FA_DK_SMDST_STRIDE + i_local_hi] = b10;
            smdST[j_local_hi * FA_DK_SMDST_STRIDE + i_local_hi] = b11;
        }

        // ===== Phase 1.5 (Qr-keep-alive): transpose smQ → smQ_T from regs =====
        //   Lever (4): Qr loaded в step B остаётся живым, source для STS напрямую.
        //   Save: 16 LDS.U32 per thread per qt (LSU traffic ~halved on phase 1.5).
        //   Cost: +16 regs to live set (Qr kept across softmax/dP/dS phases).
        //
        //   Per-lane Qr→smQ_T mapping (matches MMA #1 A operand layout):
        //     Qr[ks][0] = 4 fp8 at (m_lo, k_lo_base+0..3)
        //     Qr[ks][1] = 4 fp8 at (m_hi, k_lo_base+0..3)
        //     Qr[ks][2] = 4 fp8 at (m_lo, k_hi_base+0..3)
        //     Qr[ks][3] = 4 fp8 at (m_hi, k_hi_base+0..3)
        //   Per lane: 4 ks × 4 byte × 4 writes = 64 STS.U8 (same count as prior).
        //
        //   Cross-warp sync REQUIRED before STS (smQ aliased smQ_T — all warps
        //   must complete step B's Qr loads before any STS overwrites smQ).
        __syncthreads();    // BARRIER #2': cross-warp sync, all warps past step B
        #pragma unroll
        for (int ks = 0; ks < KS_QK; ++ks) {
            int k_lo_base = ks * 32 + l_mod4 * 4 + 0;
            int k_hi_base = ks * 32 + l_mod4 * 4 + 16;
            int m_lo_q    = wid * 16 + l_div4 + 0;
            int m_hi_q    = wid * 16 + l_div4 + 8;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                uint8_t v_mlo_klo = (Qr[ks][0] >> (b * 8)) & 0xFF;
                uint8_t v_mhi_klo = (Qr[ks][1] >> (b * 8)) & 0xFF;
                uint8_t v_mlo_khi = (Qr[ks][2] >> (b * 8)) & 0xFF;
                uint8_t v_mhi_khi = (Qr[ks][3] >> (b * 8)) & 0xFF;
                smQ_area[(k_lo_base + b) * QT_STRIDE + m_lo_q] = v_mlo_klo;
                smQ_area[(k_lo_base + b) * QT_STRIDE + m_hi_q] = v_mhi_klo;
                smQ_area[(k_hi_base + b) * QT_STRIDE + m_lo_q] = v_mlo_khi;
                smQ_area[(k_hi_base + b) * QT_STRIDE + m_hi_q] = v_mhi_khi;
            }
        }
        __syncthreads();    // BARRIER #3: smQ_T ready (+ smdST visibility for step G)

        // ===== step G: dK MMA #2 (dS^T · Q → dK_acc F32 acc) =====
        #pragma unroll
        for (int kb = 0; kb < KB_DK; ++kb) {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k_q_lo = kb * 32 + l_mod4 * 4 + 0;
            int k_q_hi = kb * 32 + l_mod4 * 4 + 16;

            // A=dS^T from smdST row-major (stride FA_DK_SMDST_STRIDE=68): single LDS.U32
            uint32_t A0 = *reinterpret_cast<uint32_t*>(&smdST[m_lo * FA_DK_SMDST_STRIDE + k_q_lo]);
            uint32_t A1 = *reinterpret_cast<uint32_t*>(&smdST[m_hi * FA_DK_SMDST_STRIDE + k_q_lo]);
            uint32_t A2 = *reinterpret_cast<uint32_t*>(&smdST[m_lo * FA_DK_SMDST_STRIDE + k_q_hi]);
            uint32_t A3 = *reinterpret_cast<uint32_t*>(&smdST[m_hi * FA_DK_SMDST_STRIDE + k_q_hi]);

            #pragma unroll
            for (int ni = 0; ni < NI_DK; ++ni) {
                int n_d = ni * 8 + l_div4;

                // B=Q_T col-major from smQ_T row-major (stride QT_STRIDE)
                // Single LDS.U32 captures 4 adjacent fp8 at fixed n_d
                uint32_t B0 = *reinterpret_cast<uint32_t*>(
                    &smQ_area[n_d * QT_STRIDE + k_q_lo]);
                uint32_t B1 = *reinterpret_cast<uint32_t*>(
                    &smQ_area[n_d * QT_STRIDE + k_q_hi]);

                mma_m16n8k32_e4m3_f32(
                    dK_acc[ni][0], dK_acc[ni][1], dK_acc[ni][2], dK_acc[ni][3],
                    A0, A1, A2, A3, B0, B1,
                    dK_acc[ni][0], dK_acc[ni][1], dK_acc[ni][2], dK_acc[ni][3]);
            }
        }

        __syncthreads();    // BARRIER #4: end of qt
    }

    // ---- Final write dK_acc * scale → gmem ----
    {
        int j_local_lo = wid * 16 + l_div4 + 0;
        int j_local_hi = wid * 16 + l_div4 + 8;
        int j_g_lo = kt * Bc + j_local_lo;
        int j_g_hi = kt * Bc + j_local_hi;
        bool j_lo_ok = (j_g_lo < sl);
        bool j_hi_ok = (j_g_hi < sl);

        float *dKb = dK + b * sl * Hd;
        #pragma unroll
        for (int ni = 0; ni < NI_DK; ++ni) {
            int d_lo = ni * 8 + l_mod4 * 2 + 0;
            int d_hi = ni * 8 + l_mod4 * 2 + 1;
            if (j_lo_ok) {
                dKb[j_g_lo * Hd + d_lo] = dK_acc[ni][0] * scale;
                dKb[j_g_lo * Hd + d_hi] = dK_acc[ni][1] * scale;
            }
            if (j_hi_ok) {
                dKb[j_g_hi * Hd + d_lo] = dK_acc[ni][2] * scale;
                dKb[j_g_hi * Hd + d_hi] = dK_acc[ni][3] * scale;
            }
        }
    }
}

// =====================================================================
// D-precompute kernel: D[i] = sum_d O[i,d] * dO[i,d]
//   Inputs: O (FP16), dO (FP16) [bh, sl, hd]
//   Output: D (FP32) [bh, sl]
//   One-shot pass, memory-bound (~1-3 ms at sl=8192 bh=128).
// =====================================================================
__global__ void kernel_d_precompute(
    const __half * __restrict__ O,
    const __half * __restrict__ dO,
    float        * __restrict__ D,
    int bh, int sl, int hd)
{
    const int b = blockIdx.y;
    const int i = blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= bh || i >= sl) return;

    // Per row: sum over hd, parallelism across threadIdx.x (32 threads = 1 warp)
    const __half *Oi  = O  + (b * sl + i) * hd;
    const __half *dOi = dO + (b * sl + i) * hd;

    const int tid_x = threadIdx.x;
    constexpr int WARP = 32;

    float acc = 0.0f;
    for (int d = tid_x; d < hd; d += WARP) {
        acc += (float)Oi[d] * (float)dOi[d];
    }
    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset, WARP);
    }
    if (tid_x == 0) {
        D[b * sl + i] = acc;
    }
}

// =====================================================================
// Host launchers.
// =====================================================================
void launch_d_precompute(
    const __half *O, const __half *dO, float *D,
    int bh, int sl, int hd, cudaStream_t stream)
{
    constexpr int ROWS_PER_BLOCK = 4;
    dim3 block(32, ROWS_PER_BLOCK);
    dim3 grid((sl + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, bh);
    kernel_d_precompute<<<grid, block, 0, stream>>>(O, dO, D, bh, sl, hd);
}

void launch_dk(
    const uint8_t *Q, const uint8_t *K, const uint8_t *V,
    const __half *dO_g, const float *L, const float *D,
    float *dK,
    int bh, int sl, int hd,
    int causal, int window,
    float scale, cudaStream_t stream)
{
    if (hd != FA_DK_HD) {
        fprintf(stderr, "fa_bwd_dk: hd=%d, expected %d\n", hd, FA_DK_HD);
        exit(1);
    }
    const int Bc   = FA_DK_BC;
    const int Br   = FA_DK_BR;
    const int n_kt = (sl + Bc - 1) / Bc;
    const int grid = bh * n_kt;
    constexpr int SMQ_AREA = FA_DK_HD * FA_DK_QT_STRIDE;   // 128*68 = 8704
    const int smem_bytes =
        Bc * hd * sizeof(uint8_t)               // smK
      + Bc * hd * sizeof(uint8_t)               // smV
      + SMQ_AREA                                 // smQ_aliased
      + Br * hd * sizeof(__half)                // smdO
      + Bc * FA_DK_SMDST_STRIDE * sizeof(uint8_t)  // smdST (stride 68 padding for 4-way conflict)
      + 2 * Br * sizeof(float);                 // smL + smD

    cudaFuncSetAttribute(kernel_dk,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    kernel_dk<<<grid, FA_DK_THREADS, smem_bytes, stream>>>(
        Q, K, V, dO_g, L, D, dK, bh, sl, hd, causal, window, scale);
}

} // namespace fa_bwd_dk
