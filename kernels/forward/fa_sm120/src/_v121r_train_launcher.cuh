/* fa_sm120 v121r-TRAIN launcher: forward + LSE для backward.
   Concatenated into _v121r_train_kernel.cu by Makefile (analogous to v121r). */
namespace fa_sm120_v121r_train {
void launch(
    const uint8_t* Q, const uint8_t* K, const uint8_t* V, __half* O,
    float* L_out,  // [bh, sl] log-sum-exp per Q-row. nullptr → don't write L.
    int bh, int sl, int hd, int causal, int window,
    float scale, cudaStream_t stream)
{
    int smem = FA_BR * FA_STRIDE + 2 * FA_BC * FA_STRIDE + FA_BC * FA_STRIDE
             + hd * SMV_T_STRIDE;
    int nqt = (sl + FA_BR - 1) / FA_BR;
    int grid = bh * nqt;
    cudaFuncSetAttribute(fa96b_train_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fa96b_train_kernel<<<grid, FA_THREADS, smem, stream>>>(
        Q, K, V, O, L_out, sl, hd, causal, scale, 1.0f, 1.0f, window);
}
} /* namespace fa_sm120_v121r_train */
