// S2v4: kernel_dk_new через свизл писателя smQ + LDSM.x2.trans.b8-читатель.
//   dK[j][d] = sum_i dS^T[j,i] * Q[i,d]
//   MMA m16n8k32.e4m3.e4m3.f32:
//     A [M=j][K=i] row-major = smdS_T (Bc × Br, stride Br) — без изменений
//     B [K=i][N=d] = fp8 из смQ через LDSM.x2.trans.b8 с свизлом swz_byte
//   Q_T pack (feeder 16 LDS + 12 SHFL + 16 STS + π_V + smQ_T 8704B + барьер line 310) — УДАЛЁН.
//   Свизл писателя: cp.async(smQ_row_ptr[swz_byte(i_local, col_byte)]).
//   Ридер: LDSM.x2.trans.b8 с row_ptr 049-B (lane-shift, in-bounds) + свизл-поправка.
//   Мост 060 (row 100% 16384/16384 + 64/64 rows) + 061 (col 100% 128/128 cols + MMA-B fragment order).
//
//   SMEM 12288 B (smQ 8192 + smdS_T 4096), было 20992. Blocks/SM ≥ 4 (SMEM headroom).
//   Барьеры: t1a nat-data ready + t1b T-layout ready + t2 end-qt (было 4).

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fa_bwd_common.cuh"

#define FA_DKN_BC        64
#define FA_DKN_BR        64
#define FA_DKN_HD        128
#define FA_DKN_THREADS   128
#define FA_DKN_QT_STRIDE 68     // stride for smQ_T bank-conflict-free (mirror sealed dK)

namespace fa_bwd_dk_new {

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

__global__ void kernel_dk_new(
    const uint8_t * __restrict__ Q,       // [bh, sl, hd] FP8
    const uint8_t * __restrict__ dS_T,    // [bh, sl_j, sl_i] FP8
    float         * __restrict__ dK,      // [bh, sl, hd] FP32
    int bh, int sl, int hd,
    int causal, int window,
    float scale)
{
    constexpr int Bc         = FA_DKN_BC;
    constexpr int Br         = FA_DKN_BR;
    constexpr int Hd         = FA_DKN_HD;
    constexpr int QT_STRIDE  = FA_DKN_QT_STRIDE;
    constexpr int NI_DK      = Hd / 8;         // 16
    constexpr int KB_DK      = Br / 32;        // 2
    constexpr int KS_QK      = Hd / 32;        // 4  (used for Q-fragment load pattern)

    const int tid    = threadIdx.x;
    const int wid    = tid >> 5;
    const int lane   = tid & 31;
    const int l_div4 = lane >> 2;
    const int l_mod4 = lane & 3;

    const int n_kt = (sl + Bc - 1) / Bc;
    const int b    = blockIdx.x / n_kt;
    const int kt   = blockIdx.x % n_kt;
    if (b >= bh) return;
    const int j_base = kt * Bc;

    // S2v4 SMEM: smQ (Br*Hd=8192 свизлован) + smdS_T (Bc*Br=4096) = 12288 B.
    //   smQ_T (8704B) УДАЛЁН — LDSM.x2.trans.b8 читает свизлованный smQ напрямую.
    extern __shared__ uint8_t smem_raw[];
    uint8_t *smQ    = smem_raw;                             // 8192 (свизлован)
    uint8_t *smdS_T = smem_raw + Br * Hd;                   // 4096

    // dK_acc FP32 [16][4] = 64 regs
    float dK_acc[NI_DK][4];
    #pragma unroll
    for (int ni = 0; ni < NI_DK; ++ni)
        #pragma unroll
        for (int s = 0; s < 4; ++s) dK_acc[ni][s] = 0.0f;

    const int n_qt = (sl + Br - 1) / Br;
    const int qt_start = causal ? kt : 0;

    for (int qt = qt_start; qt < n_qt; ++qt) {
        const int i_base = qt * Br;

        // S2v4: cp.async Q rows → smQ СВИЗЛОВАННЫЙ (swz_byte, дословно из моста 060).
        {
            const uint8_t *Qb = Q + b * sl * Hd;
            constexpr int CHUNK = 16;
            constexpr int cpr   = Hd / CHUNK;    // 8
            constexpr int total = Br * cpr;      // 512
            for (int c = tid; c < total; c += FA_DKN_THREADS) {
                int i_local = c / cpr;
                int col_byte = (c % cpr) * CHUNK;
                int i_g = i_base + i_local;
                bool in_bounds = (i_g < sl);
                int dst_off = swz_byte(i_local, col_byte);   // ← СВИЗЛ (было i_local * Hd + col_byte)
                if (in_bounds) {
                    cpa16(&smQ[dst_off], &Qb[i_g * Hd + col_byte], CHUNK);
                } else {
                    uint32_t *sp = reinterpret_cast<uint32_t*>(&smQ[dst_off]);
                    sp[0] = 0u; sp[1] = 0u; sp[2] = 0u; sp[3] = 0u;
                }
            }
        }
        // 033-b A1/W2: cp.async dS_nat tile → smdS_T area (natural layout, then in-place transpose).
        //   Note: dS_T pointer parameter now semantically = dS_nat (see 033-b ABI-дельта).
        //   Nat layout: dS_nat[b][i][j], stride_ds along i-axis (i=row, j=col).
        //   Loaded as natural [i_local][j_local] in smdS_T area; transposed below.
        {
            const int stride_ds = (sl + 15) & ~15;
            const uint8_t *dSb = dS_T + (size_t)b * sl * stride_ds;   // dS_T параметр = dS_nat указатель
            constexpr int CHUNK = 16;
            constexpr int cpr   = Bc / CHUNK;    // 4 (j-dim = Bc = 64)
            constexpr int total = Br * cpr;      // 256 (i-dim = Br = 64)
            for (int c = tid; c < total; c += FA_DKN_THREADS) {
                int i_local = c / cpr;
                int col_byte = (c % cpr) * CHUNK;
                int i_g = i_base + i_local;
                int j_g_base = j_base + col_byte;
                bool i_ok = (i_g < sl);
                int j_avail = i_ok ? (sl - j_g_base) : 0;
                if (j_avail < 0) j_avail = 0;
                if (i_ok && j_avail >= CHUNK) {
                    cpa16(&smdS_T[i_local * Bc + col_byte],
                          &dSb[(size_t)i_g * stride_ds + j_g_base], CHUNK);
                } else if (!i_ok || j_avail <= 0) {
                    cpa16(&smdS_T[i_local * Bc + col_byte], dSb, 0);
                } else {
                    uint32_t *sp = reinterpret_cast<uint32_t*>(&smdS_T[i_local * Bc + col_byte]);
                    sp[0] = 0u; sp[1] = 0u; sp[2] = 0u; sp[3] = 0u;
                    #pragma unroll 1
                    for (int bb = 0; bb < 16; ++bb) {
                        if (bb < j_avail) {
                            smdS_T[i_local * Bc + col_byte + bb] =
                                dSb[(size_t)i_g * stride_ds + j_g_base + bb];
                        }
                    }
                }
            }
        }
        cpa_commit();
        cpa_wait<0>();
        __syncthreads();   // BARRIER #1a: nat data ready

        // Phase 1.5-dS: transpose smdS_T (nat) → smdS_T (T layout) via SHFL exchange.
        //   Реализация из unit-test transpose_ds_unit_test.cu (4096/4096 verified).
        //   Feeder + Phase D выведены ИЗ ЧИТАТЕЛЯ (MMA-A ниже).
        {
            const int c_ds = l_mod4;
            const int p_ds = l_div4 & 3;
            const int h_ds = l_div4 >> 2;
            uint32_t W_all[8];
            #pragma unroll
            for (int slot = 0; slot < 2; ++slot) {
                int kb = slot;
                int i0 = kb * 32 + c_ds * 4 + p_ds;
                int i1 = i0 + 16;
                int j0 = wid * 16 + 4 * h_ds;
                int j1 = j0 + 8;
                W_all[slot * 4 + 0] = *reinterpret_cast<uint32_t*>(&smdS_T[i0 * Bc + j0]);
                W_all[slot * 4 + 1] = *reinterpret_cast<uint32_t*>(&smdS_T[i0 * Bc + j1]);
                W_all[slot * 4 + 2] = *reinterpret_cast<uint32_t*>(&smdS_T[i1 * Bc + j0]);
                W_all[slot * 4 + 3] = *reinterpret_cast<uint32_t*>(&smdS_T[i1 * Bc + j1]);
            }
            __syncthreads();   // BARRIER #NEW (W2): all reads done before aliased overwrite
            #pragma unroll
            for (int slot = 0; slot < 2; ++slot) {
                uint32_t W0 = W_all[slot*4+0], W1 = W_all[slot*4+1], W2 = W_all[slot*4+2], W3 = W_all[slot*4+3];
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
                uint32_t V0 = G0, V1 = G1, V2 = G2, V3 = G3;
                #pragma unroll
                for (int r = 1; r <= 3; ++r) {
                    int src_p = (p_ds - r) & 3;
                    int src_lane = c_ds + 4 * src_p + 16 * h_ds;
                    int idx = (p_ds + r) & 3;
                    uint32_t lo_g = (idx & 1) ? G1 : G0;
                    uint32_t hi_g = (idx & 1) ? G3 : G2;
                    uint32_t expose_val = (idx & 2) ? hi_g : lo_g;
                    uint32_t val = __shfl_sync(0xFFFFFFFF, expose_val, src_lane);
                    V0 = (src_p == 0) ? val : V0;
                    V1 = (src_p == 1) ? val : V1;
                    V2 = (src_p == 2) ? val : V2;
                    V3 = (src_p == 3) ? val : V3;
                }
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
                int kb = slot;
                int m_lo = wid * 16 + 4 * h_ds + p_ds;
                int m_hi = m_lo + 8;
                int k_i_lo = kb * 32 + c_ds * 4;
                int k_i_hi = k_i_lo + 16;
                *reinterpret_cast<uint32_t*>(&smdS_T[m_lo * Br + k_i_lo]) = OUT0;
                *reinterpret_cast<uint32_t*>(&smdS_T[m_hi * Br + k_i_lo]) = OUT1;
                *reinterpret_cast<uint32_t*>(&smdS_T[m_lo * Br + k_i_hi]) = OUT2;
                *reinterpret_cast<uint32_t*>(&smdS_T[m_hi * Br + k_i_hi]) = OUT3;
            }
        }
        __syncthreads();   // BARRIER #1b: T layout ready

        // S2v4 MMA dS_T · Q → dK_acc через LDSM.x2.trans.b8 (свизлованный smQ).
        //   A = smdS_T (dS transposed already) — MMA-A читатель unchanged
        //   B = fp8 из смQ через LDSM lo (b0) + LDSM hi (b1); мост 060+061 100%
        //   Мост дифф: row_ptr = swz_byte(kb*32+lane, np*16) lo / swz_byte(kb*32+(lane&15)+16, np*16) hi
        //   LDSM output layout: R0 = b0 for ni_a, R1 = b0 for ni_b; R2=R0 dup, R3=R1 dup (ISA-квирк 045)
        //   MMA calls: ni_a using (R0_lo, R0_hi); ni_b using (R1_lo, R1_hi)
        #pragma unroll
        for (int kb = 0; kb < KB_DK; ++kb) {
            int m_lo = wid * 16 + l_div4 + 0;
            int m_hi = wid * 16 + l_div4 + 8;
            int k_i_lo = kb * 32 + l_mod4 * 4 + 0;
            int k_i_hi = kb * 32 + l_mod4 * 4 + 16;

            uint32_t A0 = *reinterpret_cast<uint32_t*>(&smdS_T[m_lo * Br + k_i_lo]);
            uint32_t A1 = *reinterpret_cast<uint32_t*>(&smdS_T[m_hi * Br + k_i_lo]);
            uint32_t A2 = *reinterpret_cast<uint32_t*>(&smdS_T[m_lo * Br + k_i_hi]);
            uint32_t A3 = *reinterpret_cast<uint32_t*>(&smdS_T[m_hi * Br + k_i_hi]);

            #pragma unroll
            for (int np = 0; np < NI_DK / 2; ++np) {
                const int ni_a = 2 * np;
                const int ni_b = 2 * np + 1;

                // LDSM lo — b0 fragments (k=lo_range)
                int row_lo = kb * 32 + lane;
                int addr_lo = swz_byte(row_lo, np * 16);
                uint32_t sm_addr_lo = __cvta_generic_to_shared(&smQ[addr_lo]);
                uint32_t B0a_lo, B0b_lo, Dlo0, Dlo1;
                asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];\n"
                    : "=r"(B0a_lo), "=r"(B0b_lo), "=r"(Dlo0), "=r"(Dlo1) : "r"(sm_addr_lo));

                // LDSM hi — b1 fragments (k=hi_range)
                int row_hi = kb * 32 + (lane & 15) + 16;
                int addr_hi = swz_byte(row_hi, np * 16);
                uint32_t sm_addr_hi = __cvta_generic_to_shared(&smQ[addr_hi]);
                uint32_t B0a_hi, B0b_hi, Dhi0, Dhi1;
                asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3},[%4];\n"
                    : "=r"(B0a_hi), "=r"(B0b_hi), "=r"(Dhi0), "=r"(Dhi1) : "r"(sm_addr_hi));

                // MMA for ni_a: B = (B0a_lo=b0, B0a_hi=b1)
                mma_m16n8k32_e4m3_f32(
                    dK_acc[ni_a][0], dK_acc[ni_a][1], dK_acc[ni_a][2], dK_acc[ni_a][3],
                    A0, A1, A2, A3, B0a_lo, B0a_hi,
                    dK_acc[ni_a][0], dK_acc[ni_a][1], dK_acc[ni_a][2], dK_acc[ni_a][3]);

                // MMA for ni_b: B = (B0b_lo=b0, B0b_hi=b1)
                mma_m16n8k32_e4m3_f32(
                    dK_acc[ni_b][0], dK_acc[ni_b][1], dK_acc[ni_b][2], dK_acc[ni_b][3],
                    A0, A1, A2, A3, B0b_lo, B0b_hi,
                    dK_acc[ni_b][0], dK_acc[ni_b][1], dK_acc[ni_b][2], dK_acc[ni_b][3]);
            }
        }
        __syncthreads();
    }

    // Epilogue: dK_acc * scale → global dK
    {
        int j_local_lo = wid * 16 + l_div4 + 0;
        int j_local_hi = wid * 16 + l_div4 + 8;
        int j_g_lo = j_base + j_local_lo;
        int j_g_hi = j_base + j_local_hi;
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

void launch_dk_new(
    const uint8_t *Q, const uint8_t *dS_T,
    float *dK,
    int bh, int sl, int hd,
    int causal, int window,
    float scale, cudaStream_t stream)
{
    if (hd != FA_DKN_HD) {
        fprintf(stderr, "fa_bwd_dk_new: hd=%d, expected %d\n", hd, FA_DKN_HD);
        exit(1);
    }
    const int Bc = FA_DKN_BC;
    const int Br = FA_DKN_BR;
    const int n_kt = (sl + Bc - 1) / Bc;
    const int grid = bh * n_kt;
    const int smem_bytes = Br * hd + Bc * Br;  // 8192 + 4096 = 12288 (S2v4: smQ_T removed)
    cudaFuncSetAttribute(kernel_dk_new,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    kernel_dk_new<<<grid, FA_DKN_THREADS, smem_bytes, stream>>>(
        Q, dS_T, dK, bh, sl, hd, causal, window, scale);
}

} // namespace fa_bwd_dk_new
