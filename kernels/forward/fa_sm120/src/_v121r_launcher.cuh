/* fa_sm120 v121r launcher wrapper (concatenated into _v121r_kernel.cu by Makefile). */
namespace fa_sm120_v121r {
void launch(
    const uint8_t* Q, const uint8_t* K, const uint8_t* V, __half* O,
    int bh, int sl, int hd, int causal, int window,
    float scale, cudaStream_t stream)
{
    int smem = FA_BR * FA_STRIDE + 2 * FA_BC * FA_STRIDE + FA_BC * FA_STRIDE
             + hd * SMV_T_STRIDE;
    int nqt = (sl + FA_BR - 1) / FA_BR;
    int grid = bh * nqt;
    cudaFuncSetAttribute(fa96b_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    fa96b_kernel<<<grid, FA_THREADS, smem, stream>>>(
        Q, K, V, O, sl, hd, causal, scale, 1.0f, 1.0f, window);
}
} /* namespace fa_sm120_v121r */
