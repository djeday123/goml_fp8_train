/* Verbatim README C example, wrapped with minimal device-memory setup. */
#include "../include/fa_sm120.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

static void fill_safe_e4m3(void* dev_buf, size_t nbytes) {
    /* Random bytes in 0..0x3F = NaN-safe and softmax-safe (max magnitude ≈ 1.75). */
    unsigned char* host = (unsigned char*)malloc(nbytes);
    for (size_t i = 0; i < nbytes; i++) host[i] = (unsigned char)(rand() & 0x3F);
    cudaMemcpy(dev_buf, host, nbytes, cudaMemcpyHostToDevice);
    free(host);
}

static int output_has_finite(void* dev_o, size_t n_halves) {
    unsigned short* host = (unsigned short*)malloc(n_halves * sizeof(unsigned short));
    cudaMemcpy(host, dev_o, n_halves * sizeof(unsigned short), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n_halves; i++) {
        unsigned short h = host[i];
        unsigned short exp = (h >> 10) & 0x1F;
        unsigned short mant = h & 0x3FF;
        if (exp == 0x1F) { /* Inf or NaN */
            free(host);
            fprintf(stderr, "non-finite at index %zu: 0x%04x\n", i, h);
            return 0;
        }
    }
    free(host);
    return 1;
}

int main(void) {
    const int BH = 64, SL = 8192, HD = 128;
    const size_t n_qkv = (size_t)BH * SL * HD;
    void *Q, *K, *V, *O;
    cudaMalloc(&Q, n_qkv);
    cudaMalloc(&K, n_qkv);
    cudaMalloc(&V, n_qkv);
    cudaMalloc(&O, n_qkv * sizeof(unsigned short));
    fill_safe_e4m3(Q, n_qkv);
    fill_safe_e4m3(K, n_qkv);
    fill_safe_e4m3(V, n_qkv);

    /* ===== begin verbatim README C block ===== */
    fa_ctx_t* ctx = NULL;
    fa_status_t s = fa_create(&ctx);
    if (s != FA_OK) {
        fprintf(stderr, "fa_create: %s — %s\n",
                fa_status_str(s), fa_last_cuda_error(ctx));
        return 1;
    }

    /* Q, K, V — uint8* device pointers (FP8 e4m3 bytes).
     *          Bytes 0x7F and 0xFF encode NaN. Full e4m3 range (±448) also
     *          overflows the FP16 softmax accumulator — for synthetic data
     *          stay in 0..0x3F (max magnitude ≈ 1.75) or pre-scale real
     *          fp16/fp32 tensors before encoding via .to(torch.float8_e4m3fn).
     * O       — __half* device pointer (FP16 output).
     * Layout: [batch_heads, seq_len, head_dim] row-major.
     */
    fa_status_t r = fa_forward(
        ctx, Q, K, V, O,
        /* bh     */ 64,
        /* sl     */ 8192,
        /* hd     */ 128,
        /* causal */ 0,
        /* window */ 0,
        /* scale  */ 1.0f / sqrtf(128.0f),
        /* stream */ 0);
    if (r != FA_OK) {
        fprintf(stderr, "fa_forward: %s — %s\n",
                fa_status_str(r), fa_last_cuda_error(ctx));
    }

    fa_destroy(ctx);
    /* ===== end verbatim README C block ===== */

    int ok = (r == FA_OK) && output_has_finite(O, n_qkv);
    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O);
    printf(ok ? "OK\n" : "FAIL\n");
    return ok ? 0 : 1;
}
