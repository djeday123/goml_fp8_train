// R2C: kernel_merged_v1 — fused [ds_gen + dV_p1]. KV-owned (bh × n_kt grid).
//   Per block: K/V resident. Loop qt: stream Q/dO/L/D, compute S, P, dP, dS,
//   materialize dS_nat/dS_T (006-I/II staging), accumulate dV_acc via P^T·dO.
//
// BIT-EXACT invariants:
//   1. dV vs sealed dV_p1: qt-loop order == sealed (causal: qt=kt..n_qt).
//      MMA_dV (kb outer, ni inner) accumulation order == sealed.
//      fp32 dV_acc non-assoc → identical order == identical bytes.
//      smP_T layout+XOR == sealed dV_p1 lines 279-282.
//   2. dS_nat/dS_T vs R1a ds_gen: dS = P·(dP-D) tile-local (L precomputed forward,
//      no cross-tile state) → bytes identical regardless of visit order.
//      Uses ds_gen's fp16x2_to_e4m3x2 quant path unchanged.
//
// SMEM ~46592 B, aliased plan (smQ / smdS_stage / smP_T union, sequential lifetimes):
//   smK (8192) + smV (8192) + smQ_region (8192 union) + smdO (16384) + smL+smD (512) + smdS_T_stage (5120)
//   Occupancy 2 blocks/SM SMEM-limited (floor(101376 / 47616) = 2).
//
// Barriers per qt: 6 (t3 post-loads / t_new1 post-smQ-reads pre-scatter /
//                    t9 pre-drain / t_new2 post-drain pre-P_T /
//                    t11 pre-MMA_dV / t13 end qt).

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fa_bwd_common.cuh"

#define FA_M_BC       64
#define FA_M_BR       64
#define FA_M_HD       128
#define FA_M_THREADS  128

namespace fa_bwd_merged_v1 {

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

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint16_t fp8x2) {
    uint32_t r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(fp8x2));
    return r;
}

__global__ void kernel_merged_v1(
    const uint8_t * __restrict__ Q,
    const uint8_t * __restrict__ K,
    const uint8_t * __restrict__ V,
    const __half  * __restrict__ dO_g,
    const float   * __restrict__ L,
    const float   * __restrict__ D,
    uint8_t       * __restrict__ dS_nat_out,
    uint8_t       * __restrict__ dS_T_out,
    float         * __restrict__ dV,
    int bh, int sl, int hd,
    int causal, int window,
    float scale)
{
    constexpr int Bc                  = FA_M_BC;
    constexpr int Br                  = FA_M_BR;
    constexpr int Hd                  = FA_M_HD;
    constexpr int NI_QK               = Bc / 8;      // 8
    constexpr int KS_QK               = Hd / 32;     // 4
    constexpr int NI_DP               = Bc / 8;      // 8
    constexpr int KS_DP               = Hd / 16;     // 8
    constexpr int NI_DV               = Hd / 8;      // 16
    constexpr int KB_DV               = Br / 16;     // 4
    constexpr int SMDS_STAGE_STRIDE   = 80;          // 006-I
    constexpr int SMDS_T_STAGE_STRIDE = 80;          // 006-II

    const int tid    = threadIdx.x;
    const int wid    = tid >> 5;
    const int lane   = tid & 31;
    const int l_div4 = lane >> 2;
    const int l_mod4 = lane & 3;

    // Swizzle XOR masks (lane-constant across kernel)
    const int k_xor     = l_div4 << 4;               // byte-space smK/smV MMA reads
    const int dO_xor_el = l_div4 << 3;               // element-space smdO MMA-B reads (matches ds_gen)

    const int n_kt = (sl + Bc - 1) / Bc;
    const int n_qt = (sl + Br - 1) / Br;
    const int b    = blockIdx.x / n_kt;
    const int kt   = blockIdx.x % n_kt;
    if (b >= bh) return;
    const int kt_base = kt * Bc;

    const int stride_ds = (sl + 15) & ~15;

    // SMEM layout (aliased over smQ region):
    //   smK (8192) + smV (8192) + smQ_region (8192) + smdO (16384) + smL (256) + smD (256) + smdS_T_stage (5120)
    extern __shared__ uint8_t smem_raw[];
    uint8_t *smK          = smem_raw;
    uint8_t *smV          = smK + Bc * Hd;                          // 8192
    uint8_t *smQ_region   = smV + Bc * Hd;                          // 8192 union
    uint8_t *smQ          = smQ_region;                             // phase A: smQ
    uint8_t *smdS_stage   = smQ_region;                             // phase C: smdS_stage
    __half  *smP_T        = reinterpret_cast<__half*>(smQ_region);  // phase D: smP_T (2048 fp16)
    __half  *smdO         = reinterpret_cast<__half*>(smQ_region + Br * Hd);  // +8192 = 16384
    float   *smL          = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smdO) + Br * Hd * 2);
    float   *smD          = smL + Br;
    uint8_t *smdS_T_stage = reinterpret_cast<uint8_t*>(smD + Br);   // 5120

    // ==== Warmup: cp.async K + V → smK + smV (resident) ====
    {
        const uint8_t *Kb = K + (size_t)b * sl * Hd;
        const uint8_t *Vb = V + (size_t)b * sl * Hd;
        constexpr int CHUNK = 16;
        constexpr int chunks_per_row = Hd / CHUNK;     // 8
        constexpr int total = Bc * chunks_per_row;     // 512
        for (int c = tid; c < total; c += FA_M_THREADS) {
            int j_local  = c / chunks_per_row;
            int col_byte = (c % chunks_per_row) * CHUNK;
            int j_g      = kt_base + j_local;
            cpa16(&smK[swz_byte(j_local, col_byte)],
                  &Kb[(size_t)j_g * Hd + col_byte],
                  (j_g < sl) ? CHUNK : 0);
            cpa16(&smV[swz_byte(j_local, col_byte)],
                  &Vb[(size_t)j_g * Hd + col_byte],
                  (j_g < sl) ? CHUNK : 0);
        }
        cpa_commit();
        cpa_wait<0>();
    }
    __syncthreads();

    // ==== dV_acc[NI_DV=16][4] fp32 = 0 (persist across qt loop) ====
    float dV_acc[NI_DV][4];
    #pragma unroll
    for (int ni = 0; ni < NI_DV; ++ni)
        #pragma unroll
        for (int s = 0; s < 4; ++s) dV_acc[ni][s] = 0.0f;

    const int qt_start = causal ? kt : 0;
    for (int qt = qt_start; qt < n_qt; ++qt) {
        const int qt_base = qt * Br;

        // ==== Step A: cp.async Q → smQ, dO → smdO, sync L + D ====
        {
            const uint8_t *Qb  = Q     + (size_t)b * sl * Hd;
            const __half  *dOb = dO_g  + (size_t)b * sl * Hd;
            constexpr int CHUNK = 16;
            // Q FP8
            constexpr int Q_cpr = Hd / CHUNK;             // 8
            constexpr int Q_total = Br * Q_cpr;           // 512
            for (int c = tid; c < Q_total; c += FA_M_THREADS) {
                int i_local  = c / Q_cpr;
                int col_byte = (c % Q_cpr) * CHUNK;
                int i_g      = qt_base + i_local;
                cpa16(&smQ[swz_byte(i_local, col_byte)],
                      &Qb[(size_t)i_g * Hd + col_byte],
                      (i_g < sl) ? CHUNK : 0);
            }
            // dO FP16 (256 B/row, 16 chunks/row)
            constexpr int dO_bpr = Hd * 2;
            constexpr int dO_cpr = dO_bpr / CHUNK;        // 16
            constexpr int dO_total = Br * dO_cpr;         // 1024
            uint8_t *smdO_b       = reinterpret_cast<uint8_t*>(smdO);
            const uint8_t *dOb_b  = reinterpret_cast<const uint8_t*>(dOb);
            for (int c = tid; c < dO_total; c += FA_M_THREADS) {
                int i_local  = c / dO_cpr;
                int col_byte = (c % dO_cpr) * CHUNK;
                int i_g      = qt_base + i_local;
                int dO_xor   = (i_local & 7) << 4;
                cpa16(smdO_b + i_local * dO_bpr + (col_byte ^ dO_xor),
                      dOb_b  + (size_t)i_g * dO_bpr + col_byte,
                      (i_g < sl) ? CHUNK : 0);
            }
            // L, D FP32 [Br]
            if (tid < Br) {
                int i_g  = qt_base + tid;
                smL[tid] = (i_g < sl) ? L[(size_t)b * sl + i_g] : 0.0f;
                smD[tid] = (i_g < sl) ? D[(size_t)b * sl + i_g] : 0.0f;
            }
            cpa_commit();
            cpa_wait<0>();
        }
        __syncthreads();                              // BARRIER t3: post-loads

        // ==== Step B: MMA-A Q·K^T → Sr fp16-acc (sealed dV_p1 pattern) ====
        uint32_t Qr[KS_QK][4];
        {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k0   = l_mod4 * 4;
            #pragma unroll
            for (int ks = 0; ks < KS_QK; ++ks) {
                int k_lo = ks * 32 + k0 + 0;
                int k_hi = ks * 32 + k0 + 16;
                Qr[ks][0] = *reinterpret_cast<uint32_t*>(&smQ[swz_byte(m_lo, k_lo)]);
                Qr[ks][1] = *reinterpret_cast<uint32_t*>(&smQ[swz_byte(m_hi, k_lo)]);
                Qr[ks][2] = *reinterpret_cast<uint32_t*>(&smQ[swz_byte(m_lo, k_hi)]);
                Qr[ks][3] = *reinterpret_cast<uint32_t*>(&smQ[swz_byte(m_hi, k_hi)]);
            }
        }

        uint32_t Sr[NI_QK][2];
        #pragma unroll
        for (int ni = 0; ni < NI_QK; ++ni) { Sr[ni][0] = 0u; Sr[ni][1] = 0u; }

        #pragma unroll
        for (int ks = 0; ks < KS_QK; ++ks) {
            int k0_b = ks * 32 + l_mod4 * 4;
            int k_lo = k0_b + 0;
            int k_hi = k0_b + 16;
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

        // ==== Step C: softmax → Pr fp16 packed (halves persist to STS smP_T later) ====
        const float L_lo = smL[wid * 16 + l_div4 + 0];
        const float L_hi = smL[wid * 16 + l_div4 + 8];
        const float D_lo = smD[wid * 16 + l_div4 + 0];
        const float D_hi = smD[wid * 16 + l_div4 + 8];
        const int   i_g_lo = qt_base + wid * 16 + l_div4 + 0;
        const int   i_g_hi = qt_base + wid * 16 + l_div4 + 8;
        const bool  i_lo_oob = (i_g_lo >= sl);
        const bool  i_hi_oob = (i_g_hi >= sl);

        auto mask_chk = [&](int i_g, bool i_oob, int j_g, bool j_oob) -> bool {
            if (i_oob || j_oob)                       return true;
            if (causal && j_g > i_g)                  return true;
            if (window > 0 && j_g < i_g + 1 - window) return true;
            return false;
        };

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
            int j_g_lo = kt_base + j_local_lo;
            int j_g_hi = kt_base + j_local_hi;
            bool j_lo_oob = (j_g_lo >= sl);
            bool j_hi_oob = (j_g_hi >= sl);

            float p00 = mask_chk(i_g_lo, i_lo_oob, j_g_lo, j_lo_oob) ? 0.0f
                       : __expf(scale * s_mlo_nlo - L_lo);
            float p01 = mask_chk(i_g_lo, i_lo_oob, j_g_hi, j_hi_oob) ? 0.0f
                       : __expf(scale * s_mlo_nhi - L_lo);
            float p10 = mask_chk(i_g_hi, i_hi_oob, j_g_lo, j_lo_oob) ? 0.0f
                       : __expf(scale * s_mhi_nlo - L_hi);
            float p11 = mask_chk(i_g_hi, i_hi_oob, j_g_hi, j_hi_oob) ? 0.0f
                       : __expf(scale * s_mhi_nhi - L_hi);

            __half2 p_lo = __halves2half2(__float2half(p00), __float2half(p01));
            __half2 p_hi = __halves2half2(__float2half(p10), __float2half(p11));
            Pr[ni][0] = *reinterpret_cast<uint32_t*>(&p_lo);
            Pr[ni][1] = *reinterpret_cast<uint32_t*>(&p_hi);
        }

        // ==== Step D: MMA-B dO·V^T → dPr fp32-acc (ds_gen path) ====
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

            uint32_t A0 = *reinterpret_cast<uint32_t*>(&smdO[m_lo * Hd + (k_lo ^ dO_xor_el)]);
            uint32_t A1 = *reinterpret_cast<uint32_t*>(&smdO[m_hi * Hd + (k_lo ^ dO_xor_el)]);
            uint32_t A2 = *reinterpret_cast<uint32_t*>(&smdO[m_lo * Hd + (k_hi ^ dO_xor_el)]);
            uint32_t A3 = *reinterpret_cast<uint32_t*>(&smdO[m_hi * Hd + (k_hi ^ dO_xor_el)]);

            #pragma unroll
            for (int ni = 0; ni < NI_DP; ++ni) {
                int n = ni * 8 + l_div4;
                uint16_t v0_u16 = *reinterpret_cast<uint16_t*>(&smV[n * Hd + (k_lo ^ k_xor)]);
                uint16_t v1_u16 = *reinterpret_cast<uint16_t*>(&smV[n * Hd + (k_hi ^ k_xor)]);
                uint32_t B0 = e4m3x2_to_f16x2(v0_u16);
                uint32_t B1 = e4m3x2_to_f16x2(v1_u16);

                mma_m16n8k16_f32(
                    dPr[ni][0], dPr[ni][1], dPr[ni][2], dPr[ni][3],
                    A0, A1, A2, A3, B0, B1,
                    dPr[ni][0], dPr[ni][1], dPr[ni][2], dPr[ni][3]);
            }
        }

        // ==== Step E: dS = P·(dP - D) quantize + STS to smdS_stage / smdS_T_stage (006 path) ====
        // Pre-scatter BARRIER t_new1: post smQ MMA-A reads → alias overlay safe.
        __syncthreads();                              // BARRIER t_new1

        #pragma unroll
        for (int np = 0; np < NI_DP / 2; ++np) {
            const int ni_a = 2 * np;
            const int ni_b = 2 * np + 1;

            __half2 pa_lo_h2 = *reinterpret_cast<__half2*>(&Pr[ni_a][0]);
            __half2 pa_hi_h2 = *reinterpret_cast<__half2*>(&Pr[ni_a][1]);
            __half2 pb_lo_h2 = *reinterpret_cast<__half2*>(&Pr[ni_b][0]);
            __half2 pb_hi_h2 = *reinterpret_cast<__half2*>(&Pr[ni_b][1]);

            float pa_mlo_nlo = (float)__low2half (pa_lo_h2);
            float pb_mlo_nlo = (float)__low2half (pb_lo_h2);
            float pa_mlo_nhi = (float)__high2half(pa_lo_h2);
            float pb_mlo_nhi = (float)__high2half(pb_lo_h2);
            float pa_mhi_nlo = (float)__low2half (pa_hi_h2);
            float pb_mhi_nlo = (float)__low2half (pb_hi_h2);
            float pa_mhi_nhi = (float)__high2half(pa_hi_h2);
            float pb_mhi_nhi = (float)__high2half(pb_hi_h2);

            float dSa_mlo_nlo = pa_mlo_nlo * (dPr[ni_a][0] - D_lo);
            float dSb_mlo_nlo = pb_mlo_nlo * (dPr[ni_b][0] - D_lo);
            float dSa_mlo_nhi = pa_mlo_nhi * (dPr[ni_a][1] - D_lo);
            float dSb_mlo_nhi = pb_mlo_nhi * (dPr[ni_b][1] - D_lo);
            float dSa_mhi_nlo = pa_mhi_nlo * (dPr[ni_a][2] - D_hi);
            float dSb_mhi_nlo = pb_mhi_nlo * (dPr[ni_b][2] - D_hi);
            float dSa_mhi_nhi = pa_mhi_nhi * (dPr[ni_a][3] - D_hi);
            float dSb_mhi_nhi = pb_mhi_nhi * (dPr[ni_b][3] - D_hi);

            __half2 dsa_lo = __halves2half2(__float2half(dSa_mlo_nlo), __float2half(dSa_mlo_nhi));
            __half2 dsb_lo = __halves2half2(__float2half(dSb_mlo_nlo), __float2half(dSb_mlo_nhi));
            __half2 dsa_hi = __halves2half2(__float2half(dSa_mhi_nlo), __float2half(dSa_mhi_nhi));
            __half2 dsb_hi = __halves2half2(__float2half(dSb_mhi_nlo), __float2half(dSb_mhi_nhi));

            uint32_t dsa_lo_u32 = *reinterpret_cast<uint32_t*>(&dsa_lo);
            uint32_t dsb_lo_u32 = *reinterpret_cast<uint32_t*>(&dsb_lo);
            uint32_t dsa_hi_u32 = *reinterpret_cast<uint32_t*>(&dsa_hi);
            uint32_t dsb_hi_u32 = *reinterpret_cast<uint32_t*>(&dsb_hi);

            uint16_t dsa_lo_fp8 = fp16x2_to_e4m3x2(dsa_lo_u32);
            uint16_t dsb_lo_fp8 = fp16x2_to_e4m3x2(dsb_lo_u32);
            uint16_t dsa_hi_fp8 = fp16x2_to_e4m3x2(dsa_hi_u32);
            uint16_t dsb_hi_fp8 = fp16x2_to_e4m3x2(dsb_hi_u32);

            int i_local_lo = wid * 16 + l_div4 + 0;
            int i_local_hi = wid * 16 + l_div4 + 8;
            int ja_lo = ni_a * 8 + l_mod4 * 2 + 0;
            int jb_lo = ni_b * 8 + l_mod4 * 2 + 0;
            int ja_hi = ni_a * 8 + l_mod4 * 2 + 1;
            int jb_hi = ni_b * 8 + l_mod4 * 2 + 1;

            // dS_nat path — STS.b16 to smdS_stage (006-I)
            *reinterpret_cast<uint16_t*>(&smdS_stage[i_local_lo * SMDS_STAGE_STRIDE + ja_lo]) = dsa_lo_fp8;
            *reinterpret_cast<uint16_t*>(&smdS_stage[i_local_lo * SMDS_STAGE_STRIDE + jb_lo]) = dsb_lo_fp8;
            *reinterpret_cast<uint16_t*>(&smdS_stage[i_local_hi * SMDS_STAGE_STRIDE + ja_lo]) = dsa_hi_fp8;
            *reinterpret_cast<uint16_t*>(&smdS_stage[i_local_hi * SMDS_STAGE_STRIDE + jb_lo]) = dsb_hi_fp8;

            // 033-c: dS_T path устранён (dk_new читает dS_nat + транспонирует on-the-fly, W2 A1)
            //        smdS_T_stage больше не пишется; STG.128 drain dS_T также убран (Step F).
        }

        __syncthreads();                              // BARRIER t9: pre-drain

        // ==== Step F: STG.128 drain dS_nat + dS_T (006 patterns) ====
        {
            uint8_t *dS_nat_b = dS_nat_out + (size_t)b * sl * stride_ds;
            constexpr int CHUNK = 16;
            constexpr int cpr   = Bc / CHUNK;    // 4
            constexpr int total = Br * cpr;      // 256
            for (int c = tid; c < total; c += FA_M_THREADS) {
                int r = c / cpr;
                int col_byte = (c % cpr) * CHUNK;
                int i_g = qt_base + r;
                int j_start = kt_base + col_byte;
                if (i_g < sl && j_start < stride_ds) {
                    uint4 chunk = *reinterpret_cast<uint4*>(&smdS_stage[r * SMDS_STAGE_STRIDE + col_byte]);
                    *reinterpret_cast<uint4*>(&dS_nat_b[(size_t)i_g * stride_ds + j_start]) = chunk;
                }
            }
        }
        // 033-c: dS_T drain устранён; dS_T буфер не пишется в DRAM.
        //        BARRIER t_new2 сохранён — нужен для dS_nat drain sync и Step G STS Pr → smP_T.

        __syncthreads();                              // BARRIER t_new2: post-drain

        // ==== Step G: STS Pr → smP_T (sealed dV_p1 layout, fp16 halves with XORs) ====
        #pragma unroll
        for (int ni = 0; ni < NI_QK; ++ni) {
            __half2 pa_lo_h2 = *reinterpret_cast<__half2*>(&Pr[ni][0]);
            __half2 pa_hi_h2 = *reinterpret_cast<__half2*>(&Pr[ni][1]);
            __half h_p00 = __low2half (pa_lo_h2);
            __half h_p01 = __high2half(pa_lo_h2);
            __half h_p10 = __low2half (pa_hi_h2);
            __half h_p11 = __high2half(pa_hi_h2);

            int i_local_lo = wid * 16 + l_div4 + 0;
            int i_local_hi = wid * 16 + l_div4 + 8;
            int j_local_lo = ni * 8 + l_mod4 * 2 + 0;
            int j_local_hi = ni * 8 + l_mod4 * 2 + 1;
            const int PT_xor_even_wr = l_mod4 << 4;
            const int PT_xor_odd_wr  = PT_xor_even_wr + 8;
            smP_T[j_local_lo * Br + (i_local_lo ^ PT_xor_even_wr)] = h_p00;
            smP_T[j_local_hi * Br + (i_local_lo ^ PT_xor_odd_wr)]  = h_p01;
            smP_T[j_local_lo * Br + (i_local_hi ^ PT_xor_even_wr)] = h_p10;
            smP_T[j_local_hi * Br + (i_local_hi ^ PT_xor_odd_wr)]  = h_p11;
        }

        __syncthreads();                              // BARRIER t11: pre-MMA_dV

        // ==== Step H: MMA_dV P^T · dO → dV_acc (sealed dV_p1 path, kb outer, ni inner) ====
        #pragma unroll
        for (int kb = 0; kb < KB_DV; ++kb) {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k_lo = kb * 16 + l_mod4 * 2 + 0;
            int k_hi = kb * 16 + l_mod4 * 2 + 8;
            const int PT_xor_rd = l_div4 << 3;
            uint32_t Ar0 = *reinterpret_cast<uint32_t*>(&smP_T[m_lo * Br + (k_lo ^ PT_xor_rd)]);
            uint32_t Ar1 = *reinterpret_cast<uint32_t*>(&smP_T[m_hi * Br + (k_lo ^ PT_xor_rd)]);
            uint32_t Ar2 = *reinterpret_cast<uint32_t*>(&smP_T[m_lo * Br + (k_hi ^ PT_xor_rd)]);
            uint32_t Ar3 = *reinterpret_cast<uint32_t*>(&smP_T[m_hi * Br + (k_hi ^ PT_xor_rd)]);

            // 040: LDSM.x4.trans.b16 читает 2 MMA-B (ni-adjacent при same kb) за инструкцию.
            //      Row-ptr layout (32 lanes -> 4 tiles × 8 rows):
            //        Tile 0 (l 0..7):   k=kb*16+0..7,   n=ni_a*8
            //        Tile 1 (l 8..15):  k=kb*16+8..15,  n=ni_a*8
            //        Tile 2 (l 16..23): k=kb*16+0..7,   n=ni_b*8
            //        Tile 3 (l 24..31): k=kb*16+8..15,  n=ni_b*8
            //      Формула row-ptr (element-space): k_row*Hd + (n_col ^ ((k_row & 7) << 3)).
            //      MMA-call order (kb outer, ni inner) сохранён -> bit-exact ✓.
            const int tile_id     = lane >> 3;
            const int row_in_tile = lane & 7;
            const int k_row       = kb * 16 + row_in_tile + ((tile_id & 1) ? 8 : 0);
            const int k_row_xor   = (k_row & 7) << 3;
            #pragma unroll
            for (int p = 0; p < NI_DV / 2; ++p) {
                const int ni_a       = 2 * p;
                const int ni_b       = 2 * p + 1;
                const int ni_choose  = (tile_id & 2) ? ni_b : ni_a;
                const int n_col_elem = ni_choose * 8;
                const int elem_addr  = k_row * Hd + (n_col_elem ^ k_row_xor);
                const uint32_t sm_addr = __cvta_generic_to_shared(&smdO[elem_addr]);

                uint32_t R0, R1, R2, R3;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)
                    : "r"(sm_addr)
                );

                // MMA for ni_a (R0=Br0, R1=Br1)
                mma_m16n8k16_f32(
                    dV_acc[ni_a][0], dV_acc[ni_a][1], dV_acc[ni_a][2], dV_acc[ni_a][3],
                    Ar0, Ar1, Ar2, Ar3,
                    R0, R1,
                    dV_acc[ni_a][0], dV_acc[ni_a][1], dV_acc[ni_a][2], dV_acc[ni_a][3]);

                // MMA for ni_b (R2=Br0, R3=Br1)
                mma_m16n8k16_f32(
                    dV_acc[ni_b][0], dV_acc[ni_b][1], dV_acc[ni_b][2], dV_acc[ni_b][3],
                    Ar0, Ar1, Ar2, Ar3,
                    R2, R3,
                    dV_acc[ni_b][0], dV_acc[ni_b][1], dV_acc[ni_b][2], dV_acc[ni_b][3]);
            }
        }

        __syncthreads();                              // BARRIER t13: end qt (before next cp.async Q/dO)
    }

    // ==== Epilogue: dV_acc → global dV[b][j_g][d] (sealed dV_p1, no scale) ====
    {
        int j_local_lo = wid * 16 + l_div4 + 0;
        int j_local_hi = wid * 16 + l_div4 + 8;
        int j_g_lo = kt_base + j_local_lo;
        int j_g_hi = kt_base + j_local_hi;
        bool j_lo_ok = (j_g_lo < sl);
        bool j_hi_ok = (j_g_hi < sl);
        float *dVb = dV + (size_t)b * sl * Hd;

        #pragma unroll
        for (int ni = 0; ni < NI_DV; ++ni) {
            int d_lo = ni * 8 + l_mod4 * 2 + 0;
            int d_hi = ni * 8 + l_mod4 * 2 + 1;
            if (j_lo_ok) {
                dVb[(size_t)j_g_lo * Hd + d_lo] = dV_acc[ni][0];
                dVb[(size_t)j_g_lo * Hd + d_hi] = dV_acc[ni][1];
            }
            if (j_hi_ok) {
                dVb[(size_t)j_g_hi * Hd + d_lo] = dV_acc[ni][2];
                dVb[(size_t)j_g_hi * Hd + d_hi] = dV_acc[ni][3];
            }
        }
    }
}

void launch_merged(
    const uint8_t *Q, const uint8_t *K, const uint8_t *V,
    const __half *dO_g, const float *L, const float *D,
    uint8_t *dS_nat, uint8_t *dS_T, float *dV,
    int bh, int sl, int hd,
    int causal, int window,
    float scale, cudaStream_t stream)
{
    if (hd != FA_M_HD) {
        fprintf(stderr, "fa_bwd_merged_v1: hd=%d, expected %d\n", hd, FA_M_HD);
        exit(1);
    }
    const int Bc = FA_M_BC;
    const int Br = FA_M_BR;
    const int n_kt = (sl + Bc - 1) / Bc;
    const int grid = bh * n_kt;
    // 038-E: dead-alloc smdS_T_stage (5120 B) снят из smem_bytes.
    //        Указатель smdS_T_stage в kernel остаётся (не используется), но SMEM не резервирует.
    const int smem_bytes =
          Bc * hd                              // smK  8192
        + Bc * hd                              // smV  8192
        + Br * hd                              // smQ_region (union: smQ/smdS_stage/smP_T) 8192
        + Br * hd * sizeof(__half)             // smdO 16384
        + 2 * Br * sizeof(float);              // smL + smD 512
    // total = 41472 B (was 46592; −5120 dead)
    cudaFuncSetAttribute(kernel_merged_v1,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    kernel_merged_v1<<<grid, FA_M_THREADS, smem_bytes, stream>>>(
        Q, K, V, dO_g, L, D, dS_nat, dS_T, dV, bh, sl, hd, causal, window, scale);
}

} // namespace fa_bwd_merged_v1
