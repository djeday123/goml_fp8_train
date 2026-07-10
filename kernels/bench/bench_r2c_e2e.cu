// R2C E2E chain: D → merged → dk_new → dq_new.
//   Fingerprint x4 (D, merged, dk_new, dq_new) + BIT-EXACT chain 11 forms × 3 gradients
//   vs sealed references (dV_p1, sealed dK, sealed AA1 dQ).
//   Sequential wall canonical 5-run vs R1 59.31 and sealed 61.6.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "fa_bwd_common.cuh"

#define CKR(c) do { cudaError_t e = (c); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(e)); std::exit(1); }} while (0)

namespace fa_bwd_dk {
void launch_d_precompute(const __half *, const __half *, float *,
                          int, int, int, cudaStream_t);
void launch_dk(const uint8_t *, const uint8_t *, const uint8_t *,
               const __half *, const float *, const float *, float *,
               int, int, int, int, int, float, cudaStream_t);
__global__ void kernel_d_precompute(const __half *, const __half *, float *,
                                    int, int, int);
__global__ void kernel_dk(const uint8_t *, const uint8_t *, const uint8_t *,
                          const __half *, const float *, const float *, float *,
                          int, int, int, int, int, float);
}
namespace fa_bwd_dq {
void launch_dq(const uint8_t *, const uint8_t *, const uint8_t *,
               const __half *, const float *, const float *, float *,
               int, int, int, int, int, float, cudaStream_t);
}
namespace fa_bwd_dv_mma_p1 {
void launch(const uint8_t *, const uint8_t *, const __half *,
            const float *, float *,
            int, int, int, int, int, float, cudaStream_t);
}
namespace fa_bwd_merged_v1 {
void launch_merged(const uint8_t *, const uint8_t *, const uint8_t *,
                    const __half *, const float *, const float *,
                    uint8_t *, uint8_t *, float *,
                    int, int, int, int, int, float, cudaStream_t);
__global__ void kernel_merged_v1(const uint8_t *, const uint8_t *, const uint8_t *,
                                 const __half *, const float *, const float *,
                                 uint8_t *, uint8_t *, float *,
                                 int, int, int, int, int, float);
}
namespace fa_bwd_dk_new {
void launch_dk_new(const uint8_t *, const uint8_t *, float *,
                    int, int, int, int, int, float, cudaStream_t);
__global__ void kernel_dk_new(const uint8_t *, const uint8_t *, float *,
                              int, int, int, int, int, float);
}
namespace fa_bwd_dq_new {
void launch_dq_new(const uint8_t *, const uint8_t *, float *,
                    int, int, int, int, int, float, cudaStream_t);
__global__ void kernel_dq_new(const uint8_t *, const uint8_t *, float *,
                              int, int, int, int, int, float);
}

struct FpExpect { const char *name; const void *fptr; int exp_regs; };
static void fingerprint_gate() {
    FpExpect gate[] = {
        {"kernel_d_precompute", (const void*)fa_bwd_dk::kernel_d_precompute,       38},
        {"kernel_merged_v1",    (const void*)fa_bwd_merged_v1::kernel_merged_v1,  252},  // 056 rollback: обе микро-пробы (A-fix +1.99%, B-fix +1.53%) КРАСНЫЕ, prod = 040 sealed
        {"kernel_dk_new",       (const void*)fa_bwd_dk_new::kernel_dk_new,        124},  // 061 S2v4: -4r vs 128 base (LDSM.x2.trans.b8 + свизл, pack удалён)
        {"kernel_dq_new",       (const void*)fa_bwd_dq_new::kernel_dq_new,         69},  // 041 KEEP: d7a11a3d разморожен (dq_new -3.47% median, вердикт 2/3 v2)
    };
    int n = sizeof(gate)/sizeof(gate[0]);
    int fails = 0;
    for (int i = 0; i < n; ++i) {
        cudaFuncAttributes fa;
        cudaError_t e = cudaFuncGetAttributes(&fa, gate[i].fptr);
        if (e != cudaSuccess) { printf("FINGERPRINT %s ERR\n", gate[i].name); fails++; continue; }
        bool ok = (fa.numRegs == gate[i].exp_regs);
        printf("FINGERPRINT %-22s numRegs=%3d (expected %3d) %s\n",
               gate[i].name, fa.numRegs, gate[i].exp_regs, ok ? "OK" : "MISMATCH");
        if (!ok) fails++;
    }
    if (fails) { fprintf(stderr, "gate FAIL\n"); std::exit(2); }
}

struct Form { const char *name; int bh; int sl; int causal; int window; };

static int bit_exact_chain() {
    const int hd = 128;
    Form forms[] = {
        {"F1", 1, 128, 0, 0}, {"F2", 1, 128, 1, 0},
        {"F3", 2, 256, 0, 0}, {"F4", 2, 256, 1, 0},
        {"F5", 4, 384, 0, 0}, {"F6", 4, 384, 1, 0},
        {"F7", 1, 512, 0, 128}, {"F8", 1, 512, 1, 128},
        {"F9", 1, 2048, 0, 0}, {"F10", 1, 2048, 1, 0},
        {"CANARY", 1, 300, 0, 96},
    };
    const int N = sizeof(forms)/sizeof(forms[0]);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.6f);

    int all_pass = 0;
    for (int f = 0; f < N; ++f) {
        Form F = forms[f];
        int bh = F.bh, sl = F.sl;
        size_t sz = (size_t)bh*sl*hd, lsz = (size_t)bh*sl;
        int stride_ds = (sl+15)&~15;
        size_t dsz = (size_t)bh*sl*stride_ds;

        std::vector<uint8_t> Q8(sz), K8(sz), V8(sz);
        std::vector<__half>  O16(sz), dO16(sz);
        std::vector<float>   L32(lsz);
        for (size_t i=0;i<sz;i++){Q8[i]=float_to_e4m3_host(dist(rng));K8[i]=float_to_e4m3_host(dist(rng));V8[i]=float_to_e4m3_host(dist(rng));O16[i]=__float2half_rn(dist(rng));dO16[i]=__float2half_rn(dist(rng));}
        for (size_t i=0;i<lsz;i++) L32[i]=dist(rng);

        uint8_t *dQ,*dK,*dV8; __half *dOG,*dOO; float *dL,*dD;
        float *rV,*rK,*rQ, *gV,*gK,*gQ;
        uint8_t *dS_nat,*dS_T;
        CKR(cudaMalloc(&dQ,sz));CKR(cudaMalloc(&dK,sz));CKR(cudaMalloc(&dV8,sz));
        CKR(cudaMalloc(&dOO,sz*sizeof(__half)));CKR(cudaMalloc(&dOG,sz*sizeof(__half)));
        CKR(cudaMalloc(&dL,lsz*sizeof(float)));CKR(cudaMalloc(&dD,lsz*sizeof(float)));
        CKR(cudaMalloc(&rV,sz*sizeof(float)));CKR(cudaMalloc(&rK,sz*sizeof(float)));CKR(cudaMalloc(&rQ,sz*sizeof(float)));
        CKR(cudaMalloc(&gV,sz*sizeof(float)));CKR(cudaMalloc(&gK,sz*sizeof(float)));CKR(cudaMalloc(&gQ,sz*sizeof(float)));
        CKR(cudaMalloc(&dS_nat,dsz)); dS_T = nullptr;   // 070: dS_T dead-alloc removed
        CKR(cudaMemcpy(dQ,Q8.data(),sz,cudaMemcpyHostToDevice));
        CKR(cudaMemcpy(dK,K8.data(),sz,cudaMemcpyHostToDevice));
        CKR(cudaMemcpy(dV8,V8.data(),sz,cudaMemcpyHostToDevice));
        CKR(cudaMemcpy(dOO,O16.data(),sz*sizeof(__half),cudaMemcpyHostToDevice));
        CKR(cudaMemcpy(dOG,dO16.data(),sz*sizeof(__half),cudaMemcpyHostToDevice));
        CKR(cudaMemcpy(dL,L32.data(),lsz*sizeof(float),cudaMemcpyHostToDevice));
        fa_bwd_dk::launch_d_precompute(dOO,dOG,dD,bh,sl,hd,0);
        CKR(cudaDeviceSynchronize());
        float scale = 1.0f/sqrtf((float)hd);

        // Reference: sealed dQ + sealed dK + sealed dV_p1
        CKR(cudaMemset(rV,0,sz*sizeof(float)));CKR(cudaMemset(rK,0,sz*sizeof(float)));CKR(cudaMemset(rQ,0,sz*sizeof(float)));
        fa_bwd_dq::launch_dq(dQ,dK,dV8,dOG,dL,dD,rQ,bh,sl,hd,F.causal,F.window,scale,0);
        fa_bwd_dk::launch_dk(dQ,dK,dV8,dOG,dL,dD,rK,bh,sl,hd,F.causal,F.window,scale,0);
        fa_bwd_dv_mma_p1::launch(dQ,dK,dOG,dL,rV,bh,sl,hd,F.causal,F.window,scale,0);
        CKR(cudaDeviceSynchronize());

        // R2C chain: merged (dS + dV) → dk_new → dq_new
        CKR(cudaMemset(gV,0,sz*sizeof(float)));CKR(cudaMemset(gK,0,sz*sizeof(float)));CKR(cudaMemset(gQ,0,sz*sizeof(float)));
        fa_bwd_merged_v1::launch_merged(dQ,dK,dV8,dOG,dL,dD,dS_nat,dS_T,gV,
                                         bh,sl,hd,F.causal,F.window,scale,0);
        fa_bwd_dk_new::launch_dk_new(dQ,dS_nat,gK,bh,sl,hd,F.causal,F.window,scale,0);
        fa_bwd_dq_new::launch_dq_new(dK,dS_nat,gQ,bh,sl,hd,F.causal,F.window,scale,0);
        CKR(cudaDeviceSynchronize());

        auto cmp = [&](const char *tag, float *a, float *b) -> size_t {
            std::vector<float> ha(sz),hb(sz);
            cudaMemcpy(ha.data(),a,sz*sizeof(float),cudaMemcpyDeviceToHost);
            cudaMemcpy(hb.data(),b,sz*sizeof(float),cudaMemcpyDeviceToHost);
            size_t m=0; double mx=0.0;
            for(size_t p=0;p<sz;p++){uint32_t ua=*(uint32_t*)&ha[p],ub=*(uint32_t*)&hb[p];if(ua!=ub){m++;double d=std::fabs((double)ha[p]-(double)hb[p]);if(d>mx)mx=d;}}
            printf("  %s mism=%zu max_abs_diff=%.3e %s\n",tag,m,mx,m==0?"BIT-EXACT":"MISMATCH");
            return m;
        };
        printf("[%-6s bh=%d sl=%4d caus=%d wnd=%d]\n",F.name,bh,sl,F.causal,F.window);
        size_t mq=cmp("dQ",rQ,gQ), mk=cmp("dK",rK,gK), mv=cmp("dV",rV,gV);
        if (mq==0 && mk==0 && mv==0) all_pass++;

        cudaFree(dQ);cudaFree(dK);cudaFree(dV8);cudaFree(dOO);cudaFree(dOG);
        cudaFree(dL);cudaFree(dD);cudaFree(rV);cudaFree(rK);cudaFree(rQ);
        cudaFree(gV);cudaFree(gK);cudaFree(gQ);cudaFree(dS_nat);cudaFree(dS_T);
    }
    printf("\n=== CHAIN BIT-EXACT SUMMARY ===\n  forms all-3 bit-exact: %d / %d\n\n",all_pass,N);
    return (all_pass==N)?0:1;
}

int main(int argc, char **argv) {
    int mode = (argc>=2 && std::string(argv[1])=="bitexact") ? 1 : 0;

    printf("=== bench_r2c_e2e: fingerprint x4 ===\n");
    fingerprint_gate();

    if (mode) { printf("\n=== BIT-EXACT chain 11 forms × 3 gradients ===\n"); return bit_exact_chain(); }

    // Wall
    int bh=128, sl=8192, hd=128, causal=0, window=0;
    int warmup=5, iters=20;
    // 063 A1: env-override CAUSAL=1 (bench-side only; kernels untouched)
    if (const char *env = std::getenv("CAUSAL")) {
        causal = std::atoi(env) ? 1 : 0;
    }
    size_t sz=(size_t)bh*sl*hd, lsz=(size_t)bh*sl;
    int stride_ds=(sl+15)&~15;
    size_t dsz=(size_t)bh*sl*stride_ds;

    printf("\nbench_r2c_e2e: bh=%d sl=%d hd=%d causal=%d window=%d warmup=%d iters=%d\n",
           bh,sl,hd,causal,window,warmup,iters);

    std::vector<uint8_t> Q8(sz),K8(sz),V8(sz);
    std::vector<__half> O16(sz),dO16(sz);
    std::vector<float> L32(lsz);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f,0.6f);
    for (size_t i=0;i<sz;i++){Q8[i]=float_to_e4m3_host(dist(rng));K8[i]=float_to_e4m3_host(dist(rng));V8[i]=float_to_e4m3_host(dist(rng));O16[i]=__float2half_rn(dist(rng));dO16[i]=__float2half_rn(dist(rng));}
    for (size_t i=0;i<lsz;i++) L32[i]=dist(rng);

    uint8_t *dQ,*dK,*dV8; __half *dOG,*dOO; float *dL,*dD,*ddV,*ddK,*ddQ; uint8_t *dS_nat,*dS_T;
    CKR(cudaMalloc(&dQ,sz));CKR(cudaMalloc(&dK,sz));CKR(cudaMalloc(&dV8,sz));
    CKR(cudaMalloc(&dOO,sz*sizeof(__half)));CKR(cudaMalloc(&dOG,sz*sizeof(__half)));
    CKR(cudaMalloc(&dL,lsz*sizeof(float)));CKR(cudaMalloc(&dD,lsz*sizeof(float)));
    CKR(cudaMalloc(&ddV,sz*sizeof(float)));CKR(cudaMalloc(&ddK,sz*sizeof(float)));CKR(cudaMalloc(&ddQ,sz*sizeof(float)));
    CKR(cudaMalloc(&dS_nat,dsz)); dS_T = nullptr;   // 070: dS_T dead-alloc removed
    CKR(cudaMemcpy(dQ,Q8.data(),sz,cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dK,K8.data(),sz,cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dV8,V8.data(),sz,cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dOO,O16.data(),sz*sizeof(__half),cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dOG,dO16.data(),sz*sizeof(__half),cudaMemcpyHostToDevice));
    CKR(cudaMemcpy(dL,L32.data(),lsz*sizeof(float),cudaMemcpyHostToDevice));
    float scale=1.0f/sqrtf((float)hd);

    for (int i=0;i<warmup;i++) {
        fa_bwd_dk::launch_d_precompute(dOO,dOG,dD,bh,sl,hd,0);
        fa_bwd_merged_v1::launch_merged(dQ,dK,dV8,dOG,dL,dD,dS_nat,dS_T,ddV,bh,sl,hd,causal,window,scale,0);
        fa_bwd_dk_new::launch_dk_new(dQ,dS_nat,ddK,bh,sl,hd,causal,window,scale,0);
        fa_bwd_dq_new::launch_dq_new(dK,dS_nat,ddQ,bh,sl,hd,causal,window,scale,0);
    }
    CKR(cudaDeviceSynchronize());

    cudaEvent_t e0,e1,e2,e3,e4;
    cudaEventCreate(&e0);cudaEventCreate(&e1);cudaEventCreate(&e2);
    cudaEventCreate(&e3);cudaEventCreate(&e4);
    double sD=0,sM=0,sK=0,sQ=0,sT=0;
    for (int i=0;i<iters;i++) {
        cudaEventRecord(e0);
        fa_bwd_dk::launch_d_precompute(dOO,dOG,dD,bh,sl,hd,0);
        cudaEventRecord(e1);
        fa_bwd_merged_v1::launch_merged(dQ,dK,dV8,dOG,dL,dD,dS_nat,dS_T,ddV,bh,sl,hd,causal,window,scale,0);
        cudaEventRecord(e2);
        fa_bwd_dk_new::launch_dk_new(dQ,dS_nat,ddK,bh,sl,hd,causal,window,scale,0);
        cudaEventRecord(e3);
        fa_bwd_dq_new::launch_dq_new(dK,dS_nat,ddQ,bh,sl,hd,causal,window,scale,0);
        cudaEventRecord(e4);
        cudaEventSynchronize(e4);
        float d,m,k,q,t;
        cudaEventElapsedTime(&d,e0,e1);cudaEventElapsedTime(&m,e1,e2);
        cudaEventElapsedTime(&k,e2,e3);cudaEventElapsedTime(&q,e3,e4);
        cudaEventElapsedTime(&t,e0,e4);
        sD+=d;sM+=m;sK+=k;sQ+=q;sT+=t;
    }
    double D=sD/iters,M=sM/iters,K=sK/iters,Q=sQ/iters,T=sT/iters;
    printf("\n=== SEQUENTIAL R2C E2E ===\n");
    printf("  D=%.3f  merged=%.3f  dk_new=%.3f  dq_new=%.3f  total=%.3f  overhead=%.3f\n",
           D,M,K,Q,T,T-D-M-K-Q);

    // FLOPS
    double base=(double)bh*sl*sl*hd;
    // R2C executes: merged (Q·K^T + dO·V^T + P^T·dO = 3 MMA) + dk_new (1 MMA) + dq_new (1 MMA) = 5 MMA
    double f_16=16.0*base, f_5mma=10.0*base;
    printf("\n=== FLOPS conventions ===\n");
    printf("  base=%.4e\n",base);
    printf("  16N^2d (Tri Dao V3 ref, executed sealed=8 MMA=16N^2d): %.4e\n",f_16);
    printf("  10N^2d (R2C actual = 5 MMA = fused-min):               %.4e\n",f_5mma);
    printf("\n=== TFLOPS ===\n");
    printf("  Sequential vs 16N^2d: %.2f T (vs sealed 285.44 T)\n", f_16/(T*1e-3)/1e12);
    printf("  Sequential vs 10N^2d: %.2f T (R2C honest, first == fused-min)\n", f_5mma/(T*1e-3)/1e12);

    // 063-r §2 CHRONO cross-check: измерение std::chrono вокруг полной цепи,
    // МИМО cudaEvent-каркаса. Env CHRONO=1 включает.
    if (const char *env = std::getenv("CHRONO")) {
        if (std::atoi(env)) {
            CKR(cudaDeviceSynchronize());
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                fa_bwd_dk::launch_d_precompute(dOO,dOG,dD,bh,sl,hd,0);
                fa_bwd_merged_v1::launch_merged(dQ,dK,dV8,dOG,dL,dD,dS_nat,dS_T,ddV,bh,sl,hd,causal,window,scale,0);
                fa_bwd_dk_new::launch_dk_new(dQ,dS_nat,ddK,bh,sl,hd,causal,window,scale,0);
                fa_bwd_dq_new::launch_dq_new(dK,dS_nat,ddQ,bh,sl,hd,causal,window,scale,0);
            }
            CKR(cudaDeviceSynchronize());
            auto t1 = std::chrono::steady_clock::now();
            double chrono_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
            printf("\n=== CHRONO cross-check ===\n");
            printf("  wall_chrono_avg_ms = %.4f (iters=%d, causal=%d)\n", chrono_ms/iters, iters, causal);
            printf("  vs cudaEvent total = %.4f  diff = %.4f ms (%.2f%%)\n",
                   T, chrono_ms/iters - T, 100.0*(chrono_ms/iters - T)/T);
        }
    }

    cudaFree(dQ);cudaFree(dK);cudaFree(dV8);cudaFree(dOO);cudaFree(dOG);
    cudaFree(dL);cudaFree(dD);cudaFree(ddV);cudaFree(ddK);cudaFree(ddQ);
    cudaFree(dS_nat);cudaFree(dS_T);
    return 0;
}
