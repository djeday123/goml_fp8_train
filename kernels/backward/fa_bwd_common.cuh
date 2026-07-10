// =====================================================================
//  fa_bwd_common.cuh — кирпичи из v121r forward FA для backward.
//
//  ЗАКОН КРИТИЧЕСКОГО ПУТИ:
//    Каждая инструкция в hot-loop должна быть либо MMA (полезная),
//    либо неустранимо нужна (transpose-tax, sync). Любая «случайная»
//    арифметика на критическом пути SASS = регрессия. Перед добавлением
//    кода в hot-loop спрашивать: уйдёт ли это в неблокирующий префикс
//    через hoisting (v121 stage1)? Если нет — это уже не оптимизация.
//
//  ТРИ КАЛИБРОВКИ (sm_120a/RTX PRO 6000 Blackwell):
//    [1] У потолка (regs 244 на v121, 247 на v96b) каждый регистр стоит
//        ~0.5–1% wall. Δregs ≤ +2 правило при любых правках hot-loop.
//    [2] Tensor util ≈ M_TILES / (M_TILES + 0.6) — линейная функция числа
//        m-tiles на варп. M_TILES=2→1 → util × 0.625 → −37% peak (v122).
//        НЕ уменьшать M_TILES.
//    [3] SMEM-доступ vs L2: LDS ≈ 30 cycles, L2 ≈ 250 cycles. Любой
//        LDL/STL spill в hot-loop = ×8 хуже LDS — катастрофа scheduling
//        (см. v102 spill 404 B → math_pipe + short_scb стали top stalls).
//
//  ПРОВЕРЕНО В ЭКСПЕРИМЕНТАХ — каждый кирпич сопровождается ссылкой.
// =====================================================================

#pragma once
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef FA_BWD_BR
#define FA_BWD_BR 128     // строк Q-тайла на блок (= forward Br)
#endif
#ifndef FA_BWD_BC
#define FA_BWD_BC 64      // строк K/V-тайла на блок (= forward Bc)
#endif
#ifndef FA_BWD_THREADS
#define FA_BWD_THREADS 128
#endif
#define FA_BWD_STRIDE 128  // hd=128 stride (как у forward v121)
#define SMV_T_STRIDE 68    // паддинг для smV_T (v68: 64 + 4 → gcd(17,32)=1 → 32 банка)

// CK error-check (как в forward)
#define CK(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }} while(0)

// =====================================================================
//  Кирпич 1: cp.async обёртки.
//  Проверено: v121 forward, использует во всех K/V cp.async load_tile.
// =====================================================================
__device__ __forceinline__ void cpa16(void *s, const void *g, int n) {
    uint32_t sa = __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"
                 ::"r"(sa), "l"(g), "r"(n));
}
__device__ __forceinline__ void cpa_commit() {
    asm volatile("cp.async.commit_group;");
}
template <int N>
__device__ __forceinline__ void cpa_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// =====================================================================
//  Кирпич 2: swizzle-функции с XOR-маппингом для bank-conflict-free SMEM.
//  Проверено: v68 (stride 68 для smV_T), v121 (hoisting gid_lane_base/gid3/gid7).
//  ЗАКОН HOISTING: (br+gid)&7 = gid&7 при br=nt*8 (mod 8 = 0) — кратный 8
//  loop index делает XOR-часть инвариантом по nt, выносится в префикс батча.
//  Применимо к ВСЕМ swizzle_byte семейство для PV/QK тайлов backward тоже.
// =====================================================================
__device__ __forceinline__ int swz_byte(int row, int col_bytes) {
    int chunk = col_bytes >> 4;
    int within = col_bytes & 15;
    return row * FA_BWD_STRIDE + ((chunk ^ (row & 7)) << 4) + within;
}
__device__ __forceinline__ int swz_byte_bc(int row, int col_bytes) {
    int chunk = col_bytes >> 4;
    int within = col_bytes & 15;
    return row * FA_BWD_BC + ((chunk ^ (row & 3)) << 4) + within;
}
__device__ __forceinline__ int swz_byte_smvt(int row, int col_bytes) {
    int chunk = col_bytes >> 4;
    int within = col_bytes & 15;
    return row * SMV_T_STRIDE + ((chunk ^ (row & 3)) << 4) + within;
}

// =====================================================================
//  Кирпич 3: load_tile_fp8 — копия forward'а v121.
//  Проверено: v121 forward, корректность 8/8 PASS.
//  Используется и для Q, и для K, и для V в обоих passах backward.
// =====================================================================
__device__ __forceinline__ void load_tile_fp8(
    uint8_t *dst, const uint8_t *src, int start, int rows,
    int seq_len, int head_dim)
{
    constexpr int CHUNK = 16;
    int chunks_per_row = head_dim / CHUNK;
    int total = rows * chunks_per_row;
    int tid = threadIdx.x;
#pragma unroll 1
    for (int c = tid; c < total; c += FA_BWD_THREADS) {
        int row = c / chunks_per_row;
        int col_bytes = (c % chunks_per_row) * CHUNK;
        int gr = start + row;
        int dst_off = swz_byte(row, col_bytes);
        cpa16(&dst[dst_off], &src[gr * head_dim + col_bytes],
              (gr < seq_len) ? 16 : 0);
    }
}

// =====================================================================
//  Кирпич 4: transpose_v — V[Bc][hd] → V_T[hd][Bc].
//  Проверено: v68 → v121 forward.
//  В backward потребуется ДВАЖДЫ: smV_T для dP=dOVᵀ и smK_T для QKᵀ.
//  Цена транспонирования теперь возвращается; на бумаге B1.4 расчёт цены.
// =====================================================================
__device__ __forceinline__ void transpose_v(
    uint8_t *smV_T, const uint8_t *smV, int head_dim)
{
    int tid = threadIdx.x;
    // 4×4 элементов FP8 на чанк (smV row-major → smV_T col-major)
    constexpr int TILE_K = 4;
    int chunks_per_row = head_dim / TILE_K;
    int total = FA_BWD_BC / TILE_K * chunks_per_row;
#pragma unroll 1
    for (int t = tid; t < total; t += FA_BWD_THREADS) {
        int kg = t / chunks_per_row;
        int ng = t % chunks_per_row;
        int k0 = kg * TILE_K, n0 = ng * TILE_K;
        uint32_t r0 = *(uint32_t *)&smV[swz_byte(k0 + 0, n0)];
        uint32_t r1 = *(uint32_t *)&smV[swz_byte(k0 + 1, n0)];
        uint32_t r2 = *(uint32_t *)&smV[swz_byte(k0 + 2, n0)];
        uint32_t r3 = *(uint32_t *)&smV[swz_byte(k0 + 3, n0)];
        // Бит-перестановка 4 байтов внутри 4×4 чанка через PRMT
        uint32_t c0, c1, c2, c3;
        asm("prmt.b32 %0, %1, %2, 0x4040;" : "=r"(c0) : "r"(r0), "r"(r1));
        asm("prmt.b32 %0, %1, %2, 0x4040;" : "=r"(c1) : "r"(r2), "r"(r3));
        asm("prmt.b32 %0, %1, %2, 0x5151;" : "=r"(c2) : "r"(r0), "r"(r1));
        asm("prmt.b32 %0, %1, %2, 0x5151;" : "=r"(c3) : "r"(r2), "r"(r3));
        *(uint32_t *)&smV_T[swz_byte_smvt(n0 + 0, k0)] = c0;
        *(uint32_t *)&smV_T[swz_byte_smvt(n0 + 1, k0)] = c1;
        *(uint32_t *)&smV_T[swz_byte_smvt(n0 + 2, k0)] = c2;
        *(uint32_t *)&smV_T[swz_byte_smvt(n0 + 3, k0)] = c3;
    }
}

// =====================================================================
//  Кирпич 5: QMMA m16n8k32 row.col FP8 e4m3 → f16 accum.
//  Проверено: v121 forward (4 ks QK batches + 2 ks PV batches).
//  Для backward — 4 MMA в Pass 2 и 3 MMA в Pass 1.
// =====================================================================
__device__ __forceinline__ void mma_fp8_f16(
    uint32_t &d0, uint32_t &d1,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t c0, uint32_t c1)
{
    asm("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(c0), "r"(c1));
}

// =====================================================================
//  Кирпич 6: fp16x2 ↔ e4m3 конверсии.
//  Проверено: v121 forward (fp16x2_to_e4m3x2 в smP STS quantize).
// =====================================================================
__device__ __forceinline__ uint16_t fp16x2_to_e4m3x2(uint32_t h2) {
    uint16_t result;
    asm("{ .reg .b16 lo;\n"
        "  cvt.rn.satfinite.e4m3x2.f16x2 lo, %1;\n"
        "  mov.b16 %0, lo;\n"
        "}" : "=h"(result) : "r"(h2));
    return result;
}

// =====================================================================
//  Кирпич 7: f16x2 softmax-паттерны (для recompute P в Pass 2 и Pass 1).
//    ex2.approx.f16x2 ≈ 2× throughput над ex2.f32 (v79 lever 4).
//    Семантика: P_ij = exp(s_ij − L_i) = ex2(log2_e × s_ij − L_i_log2).
//    В нашей log2-space L_i = m_i + log2(l_i) — exp-substract сразу f16x2.
// =====================================================================
__device__ __forceinline__ uint32_t ex2_approx_f16x2(uint32_t h2_log2) {
    uint32_t r;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(h2_log2));
    return r;
}

// =====================================================================
//  Кирпич 8: mbarrier helpers (PTX raw, sm_120a parity convention).
//  Проверено: M7/M8 (sm120-mbarrier-parity-convention).
//  ВАЖНО: на sm_120a SYNCS.PHASECHK реализует "pass if current != arg".
//  Свежий barrier parity=0, первый wait с arg=0. expected_phase = {0,0}.
//  Использование canonical квалификаторов .release.cta/.acquire.cta — no-op
//  на sm_120a, но оставляем для совместимости (см. sm120-mbarrier-qualifiers).
// =====================================================================
__device__ __forceinline__ void mbar_init(uint64_t *bar, uint32_t count) {
    uint32_t sa = __cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t *bar) {
    uint32_t sa = __cvta_generic_to_shared(bar);
    uint64_t token;
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1];"
                 : "=l"(token) : "r"(sa) : "memory");
    (void)token;
}
__device__ __forceinline__ bool mbar_test_wait(uint64_t *bar, uint32_t phase) {
    uint32_t sa = __cvta_generic_to_shared(bar);
    uint32_t result;
    asm volatile(
        "{ .reg .pred P1;\n"
        "  mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P1, [%1], %2;\n"
        "  selp.b32 %0, 1, 0, P1;\n"
        "}" : "=r"(result) : "r"(sa), "r"(phase)
    );
    return result != 0;
}
__device__ __forceinline__ void mbar_wait(uint64_t *bar, uint32_t phase) {
    while (!mbar_test_wait(bar, phase)) { }
}

// =====================================================================
//  Кирпич 9: float ↔ e4m3 host roundtrip (для CPU reference).
//  Проверено: v121 forward CPU reference + fa_bwd_cpu_reference.
// =====================================================================
static inline uint8_t float_to_e4m3_host(float f) {
    if (f != f) return 0x7Fu;
    int sign = (f < 0.0f) ? 1 : 0;
    f = fabsf(f);
    if (f == 0.0f) return (uint8_t)(sign << 7);
    if (f >= 448.0f) return (uint8_t)((sign << 7) | 0x7E);
    int e_bits;
    uint32_t m_bits;
    if (f < (1.0f / 64.0f)) {
        e_bits = 0;
        float ms = f * 1024.0f;
        int m = (int)(ms + 0.5f);
        if (m >= 8) { e_bits = 1; m = 0; }
        m_bits = (uint32_t)m;
    } else {
        int e = (int)floorf(log2f(f));
        e_bits = e + 7;
        float scale = ldexpf(1.0f, e);
        float ms = (f / scale - 1.0f) * 8.0f;
        int m = (int)(ms + 0.5f);
        if (m >= 8) { m = 0; e_bits++; }
        if (e_bits > 15) return (uint8_t)((sign << 7) | 0x7E);
        m_bits = (uint32_t)m;
    }
    return (uint8_t)((sign << 7) | (e_bits << 3) | m_bits);
}
static inline float e4m3_to_float_host(uint8_t b) {
    int sign = (b >> 7) & 1;
    int e_bits = (b >> 3) & 0xF;
    int m_bits = b & 0x7;
    float val;
    if (e_bits == 0) {
        val = (float)m_bits / 1024.0f;
    } else {
        float scale = ldexpf(1.0f, e_bits - 7);
        val = scale * (1.0f + (float)m_bits / 8.0f);
    }
    return sign ? -val : val;
}

// =====================================================================
//  Кирпич 10: kv/q iterator abstraction (заготовка для sparse).
//  На этапе B1: тривиальный range. Sparse будет передавать другой объект
//  с тем же интерфейсом — бесплатное расширение.
// =====================================================================
struct RangeIter {
    int beg, end;
    __device__ __forceinline__ int begin() const { return beg; }
    __device__ __forceinline__ int last() const { return end; }
    __device__ __forceinline__ int count() const { return end - beg; }
};
