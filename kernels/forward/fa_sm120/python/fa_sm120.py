"""
fa_sm120.py — Python ctypes binding for libfa_sm120.so.

Single-file, no build, no dependencies beyond ctypes (+ torch when used).

Example:
    >>> import fa_sm120
    >>> ctx = fa_sm120.create()
    >>> # q, k, v are torch.Tensor uint8 on CUDA, o is torch.Tensor float16 on CUDA
    >>> fa_sm120.forward(ctx, q, k, v, o, scale=1/8.0, causal=1, window=0)
    >>> fa_sm120.destroy(ctx)
"""
import ctypes
import os

# Locate libfa_sm120.so — try common locations
def _find_lib():
    paths = [
        os.environ.get("FA_SM120_LIB"),
        os.path.join(os.path.dirname(__file__), "..", "libfa_sm120.so"),
        os.path.join(os.path.dirname(__file__), "..", "..", "libfa_sm120.so"),
        "/usr/local/lib/libfa_sm120.so",
        "libfa_sm120.so",
    ]
    for p in paths:
        if p and os.path.isfile(p):
            return p
    raise RuntimeError(
        "libfa_sm120.so not found. Set FA_SM120_LIB env var or place near this file."
    )

_LIB_PATH = _find_lib()
_lib = ctypes.CDLL(_LIB_PATH)

# fa_status_t enum (mirror of fa_sm120.h)
FA_OK                    = 0
FA_ERR_INVALID_ARG       = 1
FA_ERR_UNSUPPORTED_ARCH  = 2
FA_ERR_UNSUPPORTED_HD    = 3
FA_ERR_UNSUPPORTED_SHAPE = 4
FA_ERR_CUDA              = 5
FA_ERR_OOM               = 6
FA_ERR_INTERNAL          = 7

class _FaCtx(ctypes.Structure):
    pass  # opaque
_FaCtxPtr = ctypes.POINTER(_FaCtx)

# Signatures
_lib.fa_create.argtypes  = [ctypes.POINTER(_FaCtxPtr)]
_lib.fa_create.restype   = ctypes.c_int

_lib.fa_destroy.argtypes = [_FaCtxPtr]
_lib.fa_destroy.restype  = ctypes.c_int

_lib.fa_forward.argtypes = [
    _FaCtxPtr,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # q, k, v
    ctypes.c_void_p,                                    # o
    ctypes.c_int, ctypes.c_int, ctypes.c_int,           # bh, sl, hd
    ctypes.c_int, ctypes.c_int,                         # causal, window
    ctypes.c_float, ctypes.c_void_p                     # scale, stream
]
_lib.fa_forward.restype  = ctypes.c_int

_lib.fa_version.argtypes = []
_lib.fa_version.restype  = ctypes.c_char_p
_lib.fa_status_str.argtypes = [ctypes.c_int]
_lib.fa_status_str.restype  = ctypes.c_char_p
_lib.fa_last_cuda_error.argtypes = [_FaCtxPtr]
_lib.fa_last_cuda_error.restype  = ctypes.c_char_p

_lib.fa_dispatch_select.argtypes = [ctypes.c_int]*5
_lib.fa_dispatch_select.restype  = ctypes.c_int
_lib.fa_kernel_name.argtypes = [ctypes.c_int]
_lib.fa_kernel_name.restype  = ctypes.c_char_p


class FaError(RuntimeError):
    def __init__(self, status, ctx=None, hint=""):
        msg = _lib.fa_status_str(status).decode()
        if ctx is not None:
            cuda = _lib.fa_last_cuda_error(ctx).decode()
            if cuda:
                msg = f"{msg} ({cuda})"
        if hint:
            msg = f"{msg} [{hint}]"
        super().__init__(f"fa_sm120 status={status}: {msg}")
        self.status = status


def version() -> str:
    return _lib.fa_version().decode()


def create():
    """Create a context. Returns opaque pointer. Raises on non-sm_120a card."""
    ctx_ptr = _FaCtxPtr()
    status = _lib.fa_create(ctypes.byref(ctx_ptr))
    if status != FA_OK:
        raise FaError(status, ctx_ptr if ctx_ptr else None)
    return ctx_ptr


def destroy(ctx) -> None:
    _lib.fa_destroy(ctx)


def forward(ctx, q, k, v, o, scale, causal=0, window=0, stream=0):
    """Single FA forward call.

    q, k, v: device pointers (int) or torch.Tensor (uint8, on CUDA).
    o:       device pointer (int) or torch.Tensor (float16, on CUDA).
    scale:   typically 1/sqrt(head_dim).
    causal:  0=no mask, 1=causal upper-triangular.
    window:  0=no window, >0=sliding window length.
    stream:  CUDA stream pointer (0=default).
    """
    def _ptr_and_shape(t):
        if hasattr(t, 'data_ptr'):
            return int(t.data_ptr()), tuple(t.shape)
        return int(t), None

    q_ptr, q_shape = _ptr_and_shape(q)
    k_ptr, k_shape = _ptr_and_shape(k)
    v_ptr, v_shape = _ptr_and_shape(v)
    o_ptr, o_shape = _ptr_and_shape(o)

    if q_shape is None and (q_shape != k_shape or q_shape != v_shape):
        # Raw pointers — caller must pass bh/sl/hd explicitly via kwargs (not exposed here for simplicity)
        raise FaError(FA_ERR_INVALID_ARG, ctx,
                      "torch.Tensor inputs required; raw pointers need a different overload")

    # Derive bh, sl, hd from q.shape = [BH, S, HD]
    if q_shape is None:
        raise FaError(FA_ERR_INVALID_ARG, ctx,
                      "Pass torch.Tensor inputs (with .data_ptr() and .shape).")
    if len(q_shape) != 3:
        raise FaError(FA_ERR_INVALID_ARG, ctx,
                      f"Expected 3D tensor [bh, sl, hd], got shape={q_shape}")
    bh, sl, hd = q_shape

    status = _lib.fa_forward(
        ctx,
        q_ptr, k_ptr, v_ptr, o_ptr,
        int(bh), int(sl), int(hd),
        int(causal), int(window),
        float(scale), int(stream)
    )
    if status != FA_OK:
        raise FaError(status, ctx)


def dispatch_select(bh, sl, hd, causal=0, window=0):
    """Returns (kernel_id, kernel_name)."""
    kid = _lib.fa_dispatch_select(int(bh), int(sl), int(hd), int(causal), int(window))
    name = _lib.fa_kernel_name(kid).decode()
    return kid, name


if __name__ == "__main__":
    print(f"libfa_sm120 version: {version()}")
    print(f"dispatch bh=64 sl=8192 hd=128 wnd=0  → {dispatch_select(64, 8192, 128)}")
    print(f"dispatch bh=4  sl=1024 hd=128 wnd=0  → {dispatch_select(4, 1024, 128)}")
    print(f"dispatch bh=16 sl=4096 hd=128 wnd=1024 → {dispatch_select(16, 4096, 128, 1, 1024)}")
    print(f"dispatch bh=64 sl=8192 hd=64  wnd=0  → {dispatch_select(64, 8192, 64)}")
