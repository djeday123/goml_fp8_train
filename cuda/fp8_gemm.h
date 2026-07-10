#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// fp8_gemm performs C = A * B with FP8 inputs and FP32 output.
//
// Parameters
// ----------
// a_data   : FP8 row-major matrix A (M x K bytes)
// a_scale  : per-tensor dequantisation scale for A
// a_dtype  : 0 = E4M3FN, 1 = E5M2
// b_data   : FP8 row-major matrix B (K x N bytes)
// b_scale  : per-tensor dequantisation scale for B
// b_dtype  : 0 = E4M3FN, 1 = E5M2
// c_out    : FP32 row-major output matrix C (M x N floats)
// M, N, K  : matrix dimensions
void fp8_gemm(
    const uint8_t* a_data, float a_scale, int a_dtype,
    const uint8_t* b_data, float b_scale, int b_dtype,
    float*         c_out,
    int M, int N, int K
);

// fp8_gemm_backward_dW computes the weight gradient:
//   dW (K×N, fp32) = A^T (K×M, fp8-e4m3) * dY (M×N, fp8-e5m2)
// Used in the FP8 backward pass.
void fp8_gemm_backward_dW(
    const uint8_t* a_data,  float a_scale,
    const uint8_t* dy_data, float dy_scale,
    float*         dw_out,
    int M, int N, int K
);

// fp8_gemm_backward_dX computes the input gradient:
//   dX (M×K, fp32) = dY (M×N, fp8-e5m2) * W^T (N×K, fp8-e4m3)
void fp8_gemm_backward_dX(
    const uint8_t* dy_data, float dy_scale,
    const uint8_t* w_data,  float w_scale,
    float*         dx_out,
    int M, int N, int K
);

#ifdef __cplusplus
}
#endif
