"""Which Sol-Attn implementation runs on a given device.

There are two, and they are not interchangeable:

  ``triton``  the CUDA kernels in ``_tri_fwd`` / ``_int8_fwd``. bf16, head_dim
              128, and an INT8 path that feeds CUDA's INT8 tensor cores.
  ``torch``   the portable implementation in ``_torch_fwd``. fp16/bf16/fp32,
              any head dim, no INT8 -- quantisation buys nothing without tensor
              cores to feed it, and Metal has no equivalent.

Triton has no Apple Silicon backend, so on MPS ``_tri_fwd`` fails at ``import
triton`` before any device is queried. Both imports are therefore lazy and
guarded, and callers go through ``select`` instead of importing a kernel module
directly, which keeps device policy in this one file.

Each backend is wrapped in a dispatch closure with a single uniform signature,
so the node code passes the full option set and each backend absorbs the
options that are its own (``use_tma`` and INT8 are Triton's; nothing here has
an equivalent).
"""

import logging

import torch

_triton = None            # (sol_attn, sol_attn_int8, has_tma), or an Exception
_torch = None


def _load_triton():
    global _triton
    if _triton is None:
        try:
            from ._tri_fwd import sol_attn, _has_tma
        except Exception as exc:
            _triton = exc
            return _triton
        try:
            from ._int8_fwd import sol_attn_int8
        except Exception as exc:
            logging.info(f"[sol_attn] INT8 kernel unavailable ({exc}); bf16 only")
            sol_attn_int8 = None
        _triton = (sol_attn, sol_attn_int8, _has_tma)
    return _triton


def _load_torch():
    global _torch
    if _torch is None:
        try:
            from ._torch_fwd import sol_attn
            _torch = sol_attn
        except Exception as exc:
            _torch = exc
    return _torch


class Backend:
    """One device's implementation, plus what it will and will not accept."""

    def __init__(self, name, dispatch, *, dtypes, head_dim=None,
                 supports_int8=False, has_tma=None, break_even=0):
        self.name = name
        self._dispatch = dispatch
        self.dtypes = dtypes
        self.head_dim = head_dim          # None: any
        self.supports_int8 = supports_int8
        self._has_tma = has_tma
        # Below this sequence length the backend is slower than the host's dense
        # attention, so it declines rather than making the model slower. It is a
        # floor under the node's own min_tokens, never a substitute for it.
        self.break_even = break_even

    def has_tma(self, device):
        return bool(self._has_tma and self._has_tma(device))

    def rejects(self, dtype, head_dim, tokens=None):
        """Why this backend cannot take the call, or None."""
        if tokens is not None and tokens < self.break_even:
            return (f"seq {tokens} below this backend's break-even "
                    f"{self.break_even}")
        if dtype not in self.dtypes:
            names = "/".join(str(d).rsplit(".", 1)[-1] for d in self.dtypes)
            return f"dtype {dtype} (this backend takes {names})"
        if self.head_dim is not None and head_dim != self.head_dim:
            return f"head_dim {head_dim} != {self.head_dim}"
        return None

    def __call__(self, q, k, v, *, scale, tau, sink_blocks, sink_q,
                 int8=False, int8_pv=True, use_tma=False):
        return self._dispatch(q, k, v, scale=scale, tau=tau,
                              sink_blocks=sink_blocks, sink_q=sink_q,
                              int8=int8, int8_pv=int8_pv, use_tma=use_tma)


def select(device):
    """The backend for ``device``, or a string explaining why there is none."""
    kind = device.type if isinstance(device, torch.device) else str(device)

    if kind == "cuda":
        loaded = _load_triton()
        if isinstance(loaded, Exception):
            return f"Triton kernels unavailable: {loaded}"
        kernel, int8_kernel, has_tma = loaded

        def dispatch(q, k, v, *, scale, tau, sink_blocks, sink_q,
                     int8, int8_pv, use_tma):
            if int8 and int8_kernel is not None:
                return int8_kernel(q, k, v, scale=scale, tau=tau,
                                   sink_blocks=sink_blocks, sink_q=sink_q,
                                   use_tma=use_tma, int8_pv=int8_pv)
            return kernel(q, k, v, scale=scale, tau=tau, sink_blocks=sink_blocks,
                          sink_q=sink_q, use_tma=use_tma)

        return Backend("triton", dispatch, dtypes=(torch.bfloat16,), head_dim=128,
                       supports_int8=int8_kernel is not None, has_tma=has_tma)

    if kind in ("mps", "cpu"):
        loaded = _load_torch()
        if isinstance(loaded, Exception):
            return f"portable kernel unavailable: {loaded}"

        def dispatch(q, k, v, *, scale, tau, sink_blocks, sink_q,
                     int8, int8_pv, use_tma):
            # INT8 and TMA are CUDA tensor-core features with no counterpart
            # here; the node reports that once and runs the same math without.
            return loaded(q, k, v, scale=scale, tau=tau,
                          sink_blocks=sink_blocks, sink_q=sink_q)

        # Measured on an M3 Max against ComfyUI's own MPS attention: the sparse
        # path loses below ~8k tokens, where the routing and gather overheads
        # outweigh the work they remove, and wins from there upward.
        return Backend("torch", dispatch,
                       dtypes=(torch.float16, torch.bfloat16, torch.float32),
                       break_even=8192)

    return f"no Sol-Attn backend for device type {kind!r}"


def status():
    """One line per usable device type, for the patch node to log."""
    lines = []
    for kind in ("cuda", "mps", "cpu"):
        if kind == "cuda" and not torch.cuda.is_available():
            continue
        if kind == "mps" and not torch.backends.mps.is_available():
            continue
        got = select(torch.device(kind))
        if isinstance(got, str):
            lines.append(f"{kind}: unavailable ({got})")
        else:
            lines.append(f"{kind}: {got.name} backend, "
                         f"{'int8' if got.supports_int8 else 'no int8'}")
    return lines


__all__ = ["select", "status", "Backend"]
