"""Test the README Python example verbatim. Must execute as-is without modification."""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
os.environ.setdefault("FA_SM120_LIB",
    os.path.join(os.path.dirname(__file__), "..", "libfa_sm120.so"))

# ===== begin verbatim README block =====
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
# ===== end verbatim README block =====

print(f"OK: output shape={tuple(o.shape)} dtype={o.dtype} "
      f"min={o.min().item():.4f} max={o.max().item():.4f}")
