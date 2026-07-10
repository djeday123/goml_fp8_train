# fa-blackwell-fp8 — FlashAttention FP8 forward for NVIDIA Blackwell consumer GPUs

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CUDA 13.1+](https://img.shields.io/badge/CUDA-13.1%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Arch: sm_120a](https://img.shields.io/badge/Arch-sm__120a-76B900)](#)
[![Peak: 652 TFLOPS](https://img.shields.io/badge/Peak-652_TFLOPS-success)](#headline-numbers)

Production-grade C library + Go and Python bindings for the **FlashAttention forward
pass in FP8 e4m3** on **NVIDIA Blackwell consumer cards** (compute capability 12.0,
e.g. RTX PRO 6000 Blackwell Workstation Edition).

- **Author:** Vugar Bakhshaliyev
- **License:** Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)
- **Technical write-up:** *coming soon*

---

## Headline numbers

30-run, same-thermal benches, NVIDIA driver `580.159.03`,
RTX PRO 6000 Blackwell Workstation Edition.

| Config (BH × SL × HD, window) | TFLOPS (mean ± σ) | Kernel |
|---|---:|---|
| 64 × 8192 × 128, wnd=0 (primary peak)   | **647.14 ± 0.64** | `v121r` |
| 128 × 8192 × 128, wnd=0 (secondary peak)| **652.40 ± 0.87** | `v121r` |
| 64 × 8192 × 64,  wnd=0                  | 466.8             | `v89`   |

Production kernel **v121r**: **255 registers, 0 spill, 0 stack frame**,
double-buffered K, single-buffered V with mid-iter V[N+1] prefetch into the
dead K slot, Sr→half2 softmax via `ex2.approx.f16x2`.
Source: [`src/_v121r_kernel.cu`](src/_v121r_kernel.cu).

`fa_version()` returns `0.1.0+652T-sm120a`.

---

## Why this library

FlashAttention 2/3 reference implementations target Hopper (sm_90) and do not have
a tuned FP8 forward path for **consumer Blackwell** (sm_120a) as of the day this
repo was published. NVIDIA's TensorRT-LLM tracks the same gap in
[issue #11799](https://github.com/NVIDIA/TensorRT-LLM/issues/11799), which is
the canonical upstream reference for "no FP8 FlashAttention on SM120 yet".

This library fills exactly that gap:

- one architecture (sm_120a) — kernels are tuned to its specific
  SMEM/register/warp budgets, not portable across Hopper/Ampere;
- one data path (FP8 e4m3 input → FP16 output) — packed `uint8`-byte layout, no
  hidden casts, no autograd glue;
- one C-ABI surface — opaque context, error codes (no exceptions, no `printf`),
  trivially callable from any host language;
- one dispatcher — kernel selection by `(batch_heads, seq_len, head_dim, causal,
  window)` is a **pure function**, table-fixed by a unit test, so you can read
  exactly which kernel ran for a given config.

There is **no** PyTorch autograd wrapper, **no** Python package on PyPI, **no**
multi-architecture support, **no** backward pass yet — those are not stage-1
goals. See [Roadmap](#roadmap).

---

## Support matrix

| Aspect | Supported |
|---|---|
| GPU architecture            | **sm_120a only** (compute capability 12.0) |
| Inputs Q / K / V            | **FP8 e4m3** (byte-packed `uint8`)         |
| Output O                    | **FP16** (`__half`)                         |
| head_dim                    | **64** or **128**                           |
| Mask                        | `causal` flag + sliding `window` length     |
| Layout                      | dense `[batch_heads, seq_len, head_dim]` row-major |
| Strides                     | dense only, no per-batch strides             |
| Streams                     | yes — pass a `cudaStream_t` per call         |
| Thread safety               | safe across distinct `fa_ctx_t*` contexts    |

Hard caps enforced by `fa_create` / `fa_forward`:

- non-sm_120a card → `FA_ERR_UNSUPPORTED_ARCH` (with diagnostic readable via
  `fa_last_cuda_error`);
- `head_dim ∉ {64, 128}` → `FA_ERR_UNSUPPORTED_HD`;
- shape with no matching kernel → `FA_ERR_UNSUPPORTED_SHAPE`;
- `null` Q/K/V/O, invalid `bh`, `sl`, `causal`, `window` → `FA_ERR_INVALID_ARG`.

---

## Quick start (C)

```c
#include "fa_sm120.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    fa_ctx_t* ctx = NULL;
    fa_status_t s = fa_create(&ctx);
    if (s != FA_OK) {
        fprintf(stderr, "fa_create: %s — %s\n",
                fa_status_str(s), fa_last_cuda_error(ctx));
        return 1;
    }

    // Q, K, V — uint8* device pointers (FP8 e4m3 bytes).
    //          Bytes 0x7F and 0xFF encode NaN. Full e4m3 range (±448) also
    //          overflows the FP16 softmax accumulator — for synthetic data
    //          stay in 0..0x3F (max magnitude ≈ 1.75) or pre-scale real
    //          fp16/fp32 tensors before encoding via .to(torch.float8_e4m3fn).
    // O       — __half* device pointer (FP16 output).
    // Layout: [batch_heads, seq_len, head_dim] row-major.
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
    return r == FA_OK ? 0 : 1;
}
```

Link with `-lfa_sm120` after `make lib`.

---

## Build

```bash
cd fa-blackwell-fp8
make lib    # → libfa_sm120.so, libfa_sm120.a
make test   # → builds + runs dispatcher unit test (29 cases)
```

Requirements:

- `nvcc` from **CUDA 13.1 or newer** (earlier CUDAs do not have sm_120a code-gen);
- driver new enough to expose compute capability 12.0 (tested on `580.159.03`);
- a Blackwell **consumer** GPU. The build itself succeeds on any host, but
  `fa_create` will return `FA_ERR_UNSUPPORTED_ARCH` at runtime on other cards.

Build flags worth knowing about:

- `-Xcompiler "-fPIC -fvisibility=hidden"` — only the eight ABI symbols are
  exported (`fa_create`, `fa_destroy`, `fa_forward`, `fa_version`,
  `fa_status_str`, `fa_last_cuda_error`, `fa_dispatch_select`, `fa_kernel_name`).
  Everything else is hidden, so the `.so` cannot accidentally collide with other
  CUDA libraries in the process.
- `-arch=sm_120a` is hard-coded (the `a` suffix matters: that's the
  architecture-specific code-gen flavour Blackwell consumer parts need).

---

## Usage

### Go

```go
import "github.com/djeday123/fa-blackwell-fp8/go/gofa"

ctx, err := gofa.Create()
if err != nil { log.Fatal(err) }
defer ctx.Destroy()

err = ctx.Forward(
    qPtr, kPtr, vPtr, oPtr,        // unsafe.Pointer — device memory
    64, 8192, 128,                  // batch_heads, seq_len, head_dim
    0, 0,                           // causal, window
    1.0/float32(math.Sqrt(128)),   // scale
    nil)                            // CUDA stream (nil = default)
```

Run the dispatcher table test (gracefully skips on non-sm_120a):

```bash
cd go/gofa && go test -v
```

### Python (ctypes, no torch dep at import)

```python
import torch, math, fa_sm120

ctx = fa_sm120.create()
print(fa_sm120.version())                # → 0.1.0+652T-sm120a

# FP8 e4m3 has two NaN encodings: byte 0x7F (positive) and 0xFF (negative).
# Synthetic data must avoid those bytes AND stay magnitude-bounded — the
# full e4m3 range (±448) saturates the FP16 softmax accumulator. Random
# bytes in 0..0x3F (max e4m3 magnitude ≈ 1.75) are NaN-free for both
# inputs and outputs. For real workloads, encode your fp16/fp32 tensors
# via `.to(torch.float8_e4m3fn)` (PyTorch ≥ 2.1) and then
# `.view(torch.uint8)` to get the byte layout the library consumes.
q = torch.randint(0, 0x40, (64, 8192, 128), dtype=torch.uint8, device="cuda")
k = torch.randint(0, 0x40, (64, 8192, 128), dtype=torch.uint8, device="cuda")
v = torch.randint(0, 0x40, (64, 8192, 128), dtype=torch.uint8, device="cuda")
o = torch.empty(            (64, 8192, 128), dtype=torch.float16, device="cuda")

fa_sm120.forward(ctx, q, k, v, o,
                 scale=1.0/math.sqrt(128),
                 causal=0, window=0)

assert torch.isfinite(o).all(), "output contains NaN/Inf"
fa_sm120.destroy(ctx)
```

The Python module loads `libfa_sm120.so` lazily; override its search location
with `FA_SM120_LIB=/path/to/libfa_sm120.so`.

---

## Dispatcher

`fa_forward` picks the kernel from a single **pure function**,
`fa_dispatch_select(bh, sl, hd, causal, window) → fa_kernel_id_t`. The same
function is exported and is the **only** entry point a future autotuner needs
to override.

| hd | Condition                                  | Kernel    | Mechanism in one line |
|---:|---|---|---|
| 128 | `wnd=0`, peak grid (bh≥32 ∨ sl≥4096)      | **v121r** | Sr→half2 softmax pipeline, +2–6% over v121 |
| 128 | `wnd>0` (sliding-window / causal-window)  | **v121**  | window champion, address-arith hoisted |
| 128 | `bh=4, sl≤2048, wnd=0` (wave-tail)        | **v122**  | Br=64 / M_TILES=1, equalises partial waves |
| 128 | `bh ∈ {4..16}, sl ≤ 4096, wnd=0` (mid-grid)| **v118**  | 1-producer + 3-consumer warp-specialised |
| 128 | `bh=4, sl=8192, wnd=1024` (sliding niche) | **v117b** | partial top-sync + localfix `smK` array  |
| 128 | narrow boundary configs                   | **v96b**  | universal baseline, `smK` localfix       |
| 64  | peak grid                                 | **v89**   | P-in-registers, shfl-based gather        |
| 64  | wave-tail                                 | **v80b**  | Br=64 wave-tail, V cp.async overlap      |

The dispatcher is fixed in the test suite — `test/test_dispatch.c` and
`go/gofa/gofa_test.go` lock all 29 production configs to expected kernel ids.
Any future change of selection logic that flips a row must be conscious.

### Inspect a routing decision without launching

```c
fa_kernel_id_t kid = fa_dispatch_select(64, 8192, 128, 0, 0);
printf("%d (%s)\n", kid, fa_kernel_name(kid));
// → 100 (v121r)
```

```python
print(fa_sm120.dispatch_select(64, 8192, 128))  # → (100, 'v121r')
```

---

## Production kernel: a closer look at v121r

`v121r` is the kernel that drives the peak number. It is a tight FlashAttention
forward with the following SHIP-1 characteristics:

| Metric           | Value |
|---|---:|
| Registers per thread        | 255 |
| Local memory / spill        | **0 B** |
| Stack frame                 | **0 B** |
| Distinct barrier IDs used   | 1 (ptxas `used 1 barriers` — multiple `__syncthreads()` calls all share `bar 0`) |
| FA tile                     | Br=128, Bc=64, M_TILES=2 |
| K stage                     | double-buffered (`smK[0]`, `smK[1]`) |
| V stage                     | single-buffered + transpose_v; V[N+1] cp.async lands mid-iter into the dead K slot |
| Softmax                     | `ex2.approx.f16x2` log-base-2 pipeline |
| P representation            | `half2` packed |

The other kernels in the dispatcher are variants of the same skeleton with
different `Br`, warp-specialisation patterns, or address-arithmetic hoisting.
They live in `src/` and are wired in incrementally — see the
[honest limitations](#limitations) below.

---

## Diagnostics

| Env var               | Effect |
|---|---|
| `FA_SM120_LIB`        | Override path to `libfa_sm120.so` for the Python binding |
| `FA_SM120_DEBUG=1`    | Log every dispatcher decision (kernel id + reason) to stderr |

API-side diagnostics:

```c
fa_status_t  s = fa_create(&ctx);
const char*  name = fa_status_str(s);          // human-readable status
const char*  cuda = fa_last_cuda_error(ctx);   // last CUDA error, ctx-bound
```

`fa_last_cuda_error` is **per-context, never global** — concurrent contexts on
different threads do not race on it.

---

## Honest limitations

This is a **SHIP-1** release. The library is real, the peak numbers are real,
the C-ABI is stable, but the dispatcher is not fully wired yet:

- **Only `v121r` is fully linked end-to-end in this release.** Other dispatcher
  branches return `FA_ERR_INTERNAL` with a clear "selected but not yet linked"
  message. The two **primary peak configurations** (`bh=64, sl=8192, wnd=0` and
  `bh=128, sl=8192, wnd=0` at `hd=128`) work today.
- **No backward pass yet.** Forward only. The B-chapter (backward) is in
  progress; see [Roadmap](#roadmap).
- **No multi-GPU** and **no non-dense strides** — single device, single context,
  contiguous `[BH, S, HD]` layout.
- **No other architectures.** sm_120a only. Use FlashAttention 2/3 for Hopper.
- **Benchmarks are from one part** (RTX PRO 6000 Blackwell Workstation Edition).
  Other sm_120a parts should match within ±5% but are not in CI.
- **Thermal drift exists.** Long runs show a 0.1–0.2% downward drift over 30
  samples on the test card. The mean/σ above are from same-thermal blocks.

If your config falls into a dispatcher cell that isn't `v121r`, the error message
will tell you exactly which kernel it would have wanted; that's also the SHIP-2
attachment point.

---

## Roadmap

- **SHIP-2 — full dispatcher wiring.** Link `v121`, `v118`, `v122`, `v117b`,
  `v96b`, `v89`, `v80b` into the C-ABI surface so every cell in the dispatcher
  table returns `FA_OK` instead of `FA_ERR_INTERNAL`.
- **B-chapter — backward pass.** Pass-1 (`dQ` along Q-tiles) and Pass-2
  (`dK`+`dV` along K-tiles) on the same FP8 e4m3 / FP16 layout. CPU reference
  + finite-difference check are already in place upstream.
- **Autotuner.** The dispatcher being a pure function means an autotuner only
  has to replace the routing decision per `(bh, sl, hd, causal, window)`. The
  unit test pins today's table as the regression baseline.
- **Technical write-up.** A detailed post on the kernel design choices
  (double-buffered K with V-prefetch into the dead K slot, `ex2.approx.f16x2`
  softmax, address-arithmetic hoisting in `v121r`, the dispatcher audit) is in
  preparation — link will appear in this section.

---

## Hardware tested

| Item              | Value |
|---|---|
| GPU               | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| Compute cap.      | 12.0 (sm_120a)                                    |
| Driver            | 580.159.03                                        |
| Toolkit           | CUDA 13.1                                         |
| Host OS           | Linux x86_64                                      |

Other sm_120a consumer parts (same compute capability) are expected to work
within ±5% of the reported numbers but are not exercised in CI.

---

## Repository layout

```
fa-blackwell-fp8/
├── include/fa_sm120.h         — public C ABI header
├── src/
│   ├── fa_ctx.cu              — context lifecycle, arch probe, forward entry
│   ├── fa_dispatch.cpp        — pure dispatcher (the one table to rule them all)
│   └── _v121r_kernel.cu       — production peak kernel (v121r)
├── go/gofa/                   — Go cgo binding + dispatcher table test
├── python/fa_sm120.py         — single-file ctypes binding
├── test/test_dispatch.c       — 29-case dispatcher regression test
├── Makefile                   — `make lib`, `make test`, `make clean`
├── LICENSE                    — Apache-2.0 verbatim
├── NOTICE                     — Copyright notice (Apache convention)
└── README.md                  — this file
```

---

## License

Licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE)
and [`NOTICE`](NOTICE).

Copyright © 2026 Vugar Bakhshaliyev.

---

## Citation

If this library is part of your published work, please cite it as:

```
Vugar Bakhshaliyev. fa-blackwell-fp8: FlashAttention FP8 forward for
NVIDIA Blackwell consumer GPUs. v0.1.0, 2026.
https://github.com/djeday123/fa-blackwell-fp8
```

A BibTeX entry will accompany the technical write-up when it lands.
