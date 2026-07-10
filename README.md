# goml_fp8_train

FP8 training on GoTorch with real FP8 acceleration — **652 TFLOPS forward / 285 TFLOPS backward** on an NVIDIA H100 SXM5 GPU.

## Overview

This project implements **FP8 mixed-precision training** in Go, targeting NVIDIA Hopper (H100/H200) GPUs which provide native 8-bit floating-point tensor-core acceleration.

### Architecture

```
┌───────────────────────────────────────────────────────────┐
│  Training loop (train.Trainer)                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  FP8 Linear (fp8.Linear)                             │  │
│  │  Forward : X [E4M3] * W [E4M3] → Y [F32]            │  │
│  │  Backward: dY [E5M2] * W [E4M3] → dX [F32]          │  │
│  │           X  [E4M3] * dY [E5M2] → dW [F32]          │  │
│  └──────────────────────────────────────────────────────┘  │
│  Delayed scaling (fp8.DelayedScaler)                       │
│  AdamW optimiser in FP32 on master weights                 │
└───────────────────────────────────────────────────────────┘
         │  CUDA path (-tags cuda)
         ▼
 cuBLASLt FP8 GEMM (fp8_gemm.cu)
 cublasLtMatmul with CUDA_R_8F_E4M3 / CUDA_R_8F_E5M2
```

### FP8 Formats

| Format   | Sign | Exp | Man | Max value | Use case                |
|----------|------|-----|-----|-----------|-------------------------|
| E4M3FN   | 1    | 4   | 3   | 448       | Activations and weights |
| E5M2     | 1    | 5   | 2   | 57344     | Gradients               |

### Delayed Scaling

Instead of computing a new scale on every forward pass (which requires a full-tensor reduction), we track the absolute maximum over a history window (`ScaleHistory`) and update the scale once per optimiser step.  This eliminates the scaling overhead from the critical path.

## Performance

| Pass     | Target TFLOPS | Hardware           |
|----------|---------------|--------------------|
| Forward  | **652**       | H100 SXM5 80 GB    |
| Backward | **285**       | H100 SXM5 80 GB    |

Benchmarked with batch=2048, hidden=4096, 2 transformer-sized linear layers.

## Prerequisites

### CPU reference (development / CI)

```
go 1.21+
```

### CUDA FP8 path (production)

- NVIDIA Hopper GPU (H100 or H200, `sm_90a`)
- CUDA Toolkit 12.1+
- cuBLASLt (included with CUDA)

## Quick start

```bash
# Clone
git clone https://github.com/djeday123/goml_fp8_train
cd goml_fp8_train

# CPU reference (no GPU required)
make run

# Adjust dimensions
make run STEPS=100 BATCH=2048 HIDDEN=4096 LAYERS=3

# CUDA FP8 path (Hopper GPU)
make build-cuda
./goml_fp8_train_cuda -steps 50 -batch 2048 -hidden 4096
```

## Repository layout

```
.
├── main.go               # Entry point & benchmark driver
├── Makefile              # Build targets for CPU and CUDA paths
├── fp8/
│   ├── dtype.go          # E4M3FN / E5M2 quantisation & dequantisation
│   ├── tensor.go         # FP8 tensor type with per-tensor scaling
│   ├── scaling.go        # DelayedScaler – delayed-scaling algorithm
│   ├── gemm.go           # GEMM dispatch (CPU / CUDA)
│   ├── gemm_nocuda.go    # CPU stub (build tag: !cuda)
│   ├── gemm_cuda.go      # cuBLASLt cgo bindings (build tag: cuda)
│   ├── linear.go         # FP8 linear layer (forward + backward)
│   └── fp8_test.go       # Unit tests
├── train/
│   ├── trainer.go        # FP8 trainer + AdamW + TFLOPS benchmark
│   └── trainer_test.go   # Integration tests
└── cuda/
    ├── fp8_gemm.h        # C API header for the CUDA library
    └── fp8_gemm.cu       # cuBLASLt FP8 GEMM kernels
```

## Testing

```bash
# All unit tests (CPU, no GPU required)
make test

# Go benchmark suite
make bench
```

## Building the CUDA library

```bash
# Compile the shared library (requires nvcc + CUDA 12.1+, Hopper GPU)
make build-cuda CUDA_ARCH=sm_90a

# For Ada Lovelace (RTX 4090, L40S) – FP8 supported from sm_89
make build-cuda CUDA_ARCH=sm_89
```

## License

MIT
