"""Sol-Attn forward in portable PyTorch ops.

Triton has no Apple Silicon backend, so on MPS the kernels in ``_tri_fwd`` and
``_int8_fwd`` cannot run at all. This module reimplements the same algorithm --
arXiv 2607.24027, Algorithm 1 -- in plain tensor operations, and is selected by
``_backend`` wherever Triton is unavailable.

Fidelity to the Triton path:

  * Routing, thresholding, the pooled-key approximation and the sink handling
    are the same computation, including the log2-domain scores and the exact
    diagonal-Gaussian threshold of ``_preprocess._diag_threshold_kernel``.
  * The kernels stream the softmax to keep a tile resident in registers. Here a
    whole query block's scores fit in memory, so the pass is folded into one
    max-subtract-normalise over both branches. That is the same result the
    running form converges to, computed with one rescale instead of many.
  * Only the INT8 path is absent. It exists to reach CUDA's INT8 tensor cores,
    which have no counterpart here; ``_backend`` reports the capability and the
    node falls back to this kernel rather than failing.

Shape and layout notes, which are what the performance actually turns on:

  * Everything is batched over (batch * head) so the whole call is a handful of
    ``bmm``s rather than a Python loop over heads.
  * Query blocks are visited in chunks. The chunk bounds the peak size of the
    gathered key/value tile, so long video sequences run in bounded memory.
  * Scores stay in the compute dtype. ``bmm`` rounds its output to that dtype
    anyway, so widening afterwards costs a second full-size tensor and buys no
    precision; the reductions that do need range accumulate in fp32.
"""

import torch

BLOCK = 64
LOG2E = 1.4426950408889634
# Stands in for -inf: it survives the compute dtype (fp16 max is 65504) and
# exp2 of anything this far below the row max is exactly zero.
NEG = -65000.0


def _summaries(q, k, v, tokens, nb, tau, scale):
    """Pooled keys and values, the routing threshold, and the proxy scores.

    Mirrors ``_preprocess.prepare``: ``kc`` is the per-block mean of K, ``vc``
    the per-block mean of V (the kernels keep a sum and weight by block length
    later; folding the length in here keeps both branches on one weight and
    keeps the numerator inside half-precision range), and the threshold is
    ``mu + tau * sigma`` under a diagonal-Gaussian model of the block scores.
    """
    B, _, H, D = q.shape
    dev = q.device
    lens = torch.full((nb,), float(BLOCK), device=dev, dtype=torch.float32)
    lens[-1] = tokens - (nb - 1) * BLOCK          # ragged tail
    ln = lens.view(1, nb, 1, 1)

    kc = k.view(B, nb, BLOCK, H, D).float().sum(2) / ln
    vc = (v.view(B, nb, BLOCK, H, D).float().sum(2) / ln).permute(0, 2, 1, 3)
    qc = q.view(B, nb, BLOCK, H, D).float().sum(2) / ln

    kv = kc.permute(0, 2, 1, 3)                                   # [B,H,NB,D]
    centroid = qc.permute(0, 2, 1, 3)                             # [B,H,NQ,D]
    kmean = kv.mean(2, keepdim=True)
    kvar = (kv - kmean).pow(2).mean(2)
    ls = scale * LOG2E
    mean = (centroid @ kmean.transpose(-1, -2)).squeeze(-1) * ls
    var = (centroid.pow(2) @ kvar.unsqueeze(-1)).squeeze(-1) * (ls * ls)
    thr = mean + tau * torch.sqrt(var.clamp_min(0.0) + 1e-6)      # [B,H,NQ]
    route = (centroid @ kv.transpose(-1, -2)) * ls                # [B,H,NQ,NB]
    return kv, vc, thr, route, lens


# Peak elements in a chunk's gathered key/value tile. 64M elements is ~128 MB
# per tile in half precision; the chunk size is derived from it so that memory
# stays bounded as the sequence grows instead of scaling with it.
CHUNK_BUDGET = 64 << 20


def _chunk_for(bh, kmax, head_dim, nb):
    per_block = max(bh * kmax * BLOCK * head_dim, 1)
    return max(1, min(64, CHUNK_BUDGET // per_block, nb))


def sol_attn(q, k, v, *, scale=None, tau=1.0, sink_blocks=(0, 0), sink_q=(0, 0),
             q_chunk=None, compute_dtype=None):
    """Sol-Attn over BTHD inputs, returning BTHD.

    ``sink_blocks`` keeps a leading span of key blocks exact for every query,
    and ``sink_q`` runs a leading span of query blocks fully dense; both carry
    MiniMax-H3's packed conditioning rows, exactly as in the Triton path.
    """
    B, T, H, D = q.shape
    scale = D ** -0.5 if scale is None else float(scale)
    tau = float(tau)
    ct = q.dtype if compute_dtype is None else compute_dtype
    if ct not in (torch.float16, torch.bfloat16, torch.float32):
        ct = torch.float16
    dev = q.device
    nb = (T + BLOCK - 1) // BLOCK
    pad = nb * BLOCK - T
    if pad:                                    # whole blocks simplify every view below
        q, k, v = (torch.nn.functional.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v))
    BH = B * H

    kv, vc, thr, route, lens = _summaries(q, k, v, T, nb, tau, scale)

    # --- routing: threshold, plus the always-exact local window and sinks ---
    idx = torch.arange(nb, device=dev)
    exact = (route > thr.unsqueeze(-1)) | ((idx.view(nb, 1) - idx).abs() <= 1)
    if sink_blocks[1] > sink_blocks[0]:
        exact |= ((idx >= sink_blocks[0]) & (idx < sink_blocks[1])).view(1, 1, 1, nb)
    if sink_q[1] > sink_q[0]:
        rows = (idx.view(nb, 1) >= sink_q[0]) & (idx.view(nb, 1) < sink_q[1])
        exact |= rows.view(1, 1, nb, 1)
    exact = exact.reshape(BH, nb, nb)                             # [BH,NQ,NB]

    # One topk for the call: row r's selected blocks are the leading entries of
    # sel_i[r], so every chunk reads a prefix of it instead of sorting again.
    counts = exact.sum(-1)
    kmax = int(counts.amax())
    sel_v, sel_i = torch.topk(exact.to(torch.int8), kmax, dim=-1, sorted=True)
    sel_keep = sel_v > 0
    if q_chunk is None:
        q_chunk = _chunk_for(BH, kmax, D, nb)

    ls = scale * LOG2E
    qs = (q.permute(0, 2, 1, 3).reshape(BH, nb * BLOCK, D).float() * ls).to(ct)
    kcT = kv.permute(0, 1, 3, 2).reshape(BH, D, nb).to(ct)
    vcm = vc.reshape(BH, nb, D).to(ct)
    # Block-major key/value so a selected block is one contiguous row range.
    kf = k.view(B, nb, BLOCK, H, D).permute(0, 3, 1, 2, 4).reshape(BH * nb, BLOCK, D).to(ct)
    vf = v.view(B, nb, BLOCK, H, D).permute(0, 3, 1, 2, 4).reshape(BH * nb, BLOCK, D).to(ct)

    tok_ok = (idx.view(nb, 1) * BLOCK + torch.arange(BLOCK, device=dev)) < T
    base = (torch.arange(BH, device=dev) * nb).view(BH, 1, 1)
    lens_r = lens.view(1, 1, nb).to(ct)
    out = torch.empty((BH, nb * BLOCK, D), device=dev, dtype=ct)

    for c0 in range(0, nb, q_chunk):
        c1 = min(c0 + q_chunk, nb)
        C, n = c1 - c0, BH * (c1 - c0)
        qi = qs[:, c0 * BLOCK:c1 * BLOCK]                         # [BH,C*64,D]
        ex = exact[:, c0:c1]                                      # [BH,C,NB]

        # Budget for this chunk only, so a chunk of sparse rows stays cheap.
        km = int(counts[:, c0:c1].amax())
        si = sel_i[:, c0:c1, :km]
        flat = (base + si).reshape(-1)
        kg = kf.index_select(0, flat).view(n, km * BLOCK, D)
        vg = vf.index_select(0, flat).view(n, km * BLOCK, D)

        # --- exact branch over the selected blocks ---
        s = torch.bmm(qi.reshape(n, BLOCK, D), kg.transpose(1, 2))
        ok = (sel_keep[:, c0:c1, :km].unsqueeze(-1) & tok_ok[si]).reshape(n, 1, km * BLOCK)
        s.masked_fill_(~ok, NEG)

        # --- approximate branch: pooled keys stand in for the rest ---
        a = torch.bmm(qi, kcT).view(n, BLOCK, nb)
        a.masked_fill_(ex.reshape(n, 1, nb), NEG)

        # --- one softmax across both branches, in place ---
        mu = torch.maximum(a.amax(-1), s.amax(-1)).unsqueeze(-1)
        s.sub_(mu).exp2_()
        a.sub_(mu).exp2_().mul_(lens_r)
        den = a.sum(-1, dtype=torch.float32) + s.sum(-1, dtype=torch.float32)
        num = torch.bmm(a.view(BH, C * BLOCK, nb), vcm).float().view(n, BLOCK, D) \
            + torch.bmm(s, vg).float()
        out[:, c0 * BLOCK:c1 * BLOCK] = (num / den.unsqueeze(-1)).to(ct).view(
            BH, C * BLOCK, D)

    return out.view(B, H, nb * BLOCK, D)[:, :, :T].permute(0, 2, 1, 3).to(q.dtype)


__all__ = ["sol_attn", "BLOCK"]
