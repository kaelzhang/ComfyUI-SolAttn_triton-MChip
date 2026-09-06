"""Regression tests for the portable Sol-Attn kernel.

Run directly -- ``python test_torch_fwd.py`` -- with no ComfyUI import, so a
port can be checked on any machine that has PyTorch:

    python test_torch_fwd.py            # pick the best available device
    python test_torch_fwd.py cpu        # force one

What is actually pinned here:

  * With the threshold driven far negative every block routes exact, so the
    kernel must reproduce dense attention. That is the one property that
    catches a wrong pooled summary, a mis-ordered gather, a dropped ragged
    tail or a broken softmax combine, and it holds independently of how well
    the approximation works on the data.
  * The routing density at a given tau must match the paper's Gaussian-tail
    figures, which pins the threshold derivation rather than the plumbing.
  * The sinks must actually force their blocks and rows exact.
  * Attention output is a convex combination of value rows, so it must stay
    inside V's range whatever the routing does.
"""

import importlib.util
import math
import os
import sys

import torch

_spec = importlib.util.spec_from_file_location(
    "_torch_fwd", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_torch_fwd.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sol_attn, BLOCK = _mod.sol_attn, _mod.BLOCK

ALL_EXACT = -1e4          # threshold low enough that every block routes exact
_failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        _failures.append(name)


def dense(q, k, v):
    """Reference attention over BTHD, in fp32."""
    qs, ks, vs = (x.transpose(1, 2).float() for x in (q, k, v))
    return torch.nn.functional.scaled_dot_product_attention(qs, ks, vs).transpose(1, 2)


def relerr(a, b):
    b = b.float()
    return float((a.float() - b).norm() / b.norm())


def peaked(B, T, H, D, device, dtype):
    """Tokens with the local structure real video attention has; uniform noise
    would make every block equally important and the method meaningless."""
    t = torch.linspace(0, 8 * math.pi, T, device=device).view(1, T, 1, 1)
    harmonics = torch.cat([torch.sin(t * (i + 1) / 8) for i in range(D)], dim=-1)
    x = torch.randn(B, 1, H, D, device=device) + 2.0 * harmonics[:, :, :, :D] \
        + 0.3 * torch.randn(B, T, H, D, device=device)
    return x.to(dtype)


def test_matches_dense(device, dtype):
    """Every block exact => the kernel is dense attention, ragged tails and all."""
    for B, T, H, D in ((1, 512, 4, 128), (2, 256, 2, 64), (1, 300, 4, 128), (1, 64, 1, 128)):
        q = peaked(B, T, H, D, device, dtype)
        k = peaked(B, T, H, D, device, dtype)
        v = torch.randn(B, T, H, D, device=device, dtype=dtype)
        err = relerr(sol_attn(q, k, v, tau=ALL_EXACT), dense(q, k, v))
        tol = 2e-2 if dtype is torch.float16 else 1e-4
        check(f"all-exact == dense  B{B} T{T} H{H} D{D}", err < tol, f"rel err {err:.2e}")


def test_chunking_is_invisible(device, dtype):
    """The chunk size is a memory knob, never a result change.

    Not bit-exact by construction: a chunk's key budget is its own rows'
    maximum, so the score tile is a different width and the row sums add up in
    a different order. The padding columns contribute exactly zero, so only
    floating-point associativity moves, which is orders of magnitude below a
    real chunk-boundary bug (those shift whole rows).
    """
    q = peaked(1, 1024, 4, 128, device, dtype)
    k = peaked(1, 1024, 4, 128, device, dtype)
    v = torch.randn(1, 1024, 4, 128, device=device, dtype=dtype)
    tol = 1e-4 if dtype is torch.float16 else 1e-6
    base = sol_attn(q, k, v, tau=1.3, q_chunk=1)
    for chunk in (2, 5, 8, 16):
        err = relerr(sol_attn(q, k, v, tau=1.3, q_chunk=chunk), base)
        check(f"chunk {chunk} == chunk 1", err < tol, f"rel err {err:.2e}")


def test_routing_density(device, dtype):
    """Paper's Gaussian-tail densities: tau 1.0 ~ 16%, 1.5 ~ 7%, 2.0 ~ 2.7%."""
    torch.manual_seed(0)
    B, T, H, D = 1, 4096, 4, 128
    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    nb = T // BLOCK
    kv, _, _, _, _ = _mod._summaries(q, k, k, T, nb, 1.0, D ** -0.5)
    for tau, want in ((1.0, 0.16), (1.5, 0.07), (2.0, 0.027)):
        _, _, thr, route, _ = _mod._summaries(q, k, k, T, nb, tau, D ** -0.5)
        got = float((route > thr.unsqueeze(-1)).float().mean())
        # Half the expected density to twice it: the point is that the
        # threshold tracks the Gaussian tail, not that it hits it exactly.
        check(f"routing density tau={tau}", want / 2 < got < want * 2,
              f"{got * 100:.1f}% (paper ~{want * 100:.0f}%)")


def test_sinks(device, dtype):
    """A sink must make its span exact, so raising tau cannot change it."""
    torch.manual_seed(0)
    B, T, H, D = 1, 1024, 4, 128
    q = peaked(B, T, H, D, device, dtype)
    k = peaked(B, T, H, D, device, dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    ref = dense(q, k, v)
    rows = 2 * BLOCK
    # Query rows inside sink_q attend everything exactly, at any tau.
    out = sol_attn(q, k, v, tau=3.5, sink_q=(0, 2), sink_blocks=(0, 2))
    err = relerr(out[:, :rows], ref[:, :rows])
    tol = 2e-2 if dtype is torch.float16 else 1e-4
    check("sink_q rows are exact", err < tol, f"rel err {err:.2e}")
    # And those rows must be closer to dense than un-sunk rows at the same tau.
    plain = sol_attn(q, k, v, tau=3.5)
    check("sink_q beats no sink on the same rows",
          err < relerr(plain[:, :rows], ref[:, :rows]))


def test_output_is_convex(device, dtype):
    """Attention averages value rows, so no output may leave V's range."""
    torch.manual_seed(0)
    B, T, H, D = 1, 1024, 4, 128
    q = peaked(B, T, H, D, device, dtype)
    k = peaked(B, T, H, D, device, dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    for tau in (0.5, 1.3, 2.5):
        out = sol_attn(q, k, v, tau=tau).float()
        lo, hi = v.float().amin(1, keepdim=True), v.float().amax(1, keepdim=True)
        slack = 1e-2 * (hi - lo)
        check(f"output within V range tau={tau}",
              bool(((out >= lo - slack) & (out <= hi + slack)).all()))
        check(f"output finite tau={tau}", bool(torch.isfinite(out).all()))


def main():
    wanted = sys.argv[1] if len(sys.argv) > 1 else None
    if wanted:
        device = torch.device(wanted)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    print(f"device={device.type} dtype={str(dtype).rsplit('.', 1)[-1]} "
          f"torch={torch.__version__}\n")
    for test in (test_matches_dense, test_chunking_is_invisible, test_routing_density,
                 test_sinks, test_output_is_convex):
        print(test.__name__)
        test(device, dtype)
        print()
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
