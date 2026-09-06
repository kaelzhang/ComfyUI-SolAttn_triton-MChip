<h1 align="center">ComfyUI-SolAttn</h1>

<h4 align="center">
  Experimental Sol-Attn for ComfyUI &mdash; CUDA and Apple Silicon
</h4>

<p align="center">
  <a href="https://arxiv.org/abs/2607.24027"><img src="https://img.shields.io/badge/📄_Paper-arXiv-b31b1b?style=flat-square" alt="Paper"/></a>
  <a href="https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends/sol_attn"><img src="https://img.shields.io/badge/💻_Code-Sol--Attn-76b900?style=flat-square" alt="Code"/></a>
  <a href="https://nvlabs.github.io/Sana/Sol-Attn/"><img src="https://img.shields.io/badge/🌐_Project-Page-blue?style=flat-square" alt="Project Page"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Apple_Silicon-supported-000000?style=flat-square&logo=apple&logoColor=white" alt="Apple Silicon supported"/>
  <img src="https://img.shields.io/badge/backend-MPS_%7C_CUDA_%7C_CPU-4a4a4a?style=flat-square" alt="Backends"/>
  <img src="https://img.shields.io/badge/benchmarked-M3_Max-000000?style=flat-square&logo=apple&logoColor=white" alt="Benchmarked on M3 Max"/>
</p>

<p align="center">
  <b>English</b> &nbsp;|&nbsp; <a href="README.zh-CN.md">简体中文</a>
</p>

---

## Overview

[Sol-Attn](https://arxiv.org/abs/2607.24027) is a training-free sparse attention
method for accelerating image and video generation. This community extension
integrates it into ComfyUI, with two interchangeable backends:

| | CUDA | Apple Silicon / CPU |
|---|---|---|
| implementation | Triton kernels | portable PyTorch |
| dtypes | bf16 | fp16 / bf16 / fp32 |
| head dim | 128 | any |
| INT8 QK/PV | yes | no |
| TMA descriptors | SM90+ | n/a |

The backend is chosen per call from the tensors it is handed, so nothing needs
configuring; options a backend has no path for are logged once and ignored
rather than failing. Routing, thresholding, the pooled-key approximation and
the conditioning sinks are the same computation on both.

**Apple Silicon is supported and benchmarked** &mdash; see [Apple Silicon](#apple-silicon)
below for the measured figures and the machine they were taken on.

> [!NOTE]
> This project is a work in progress, and the two backends are not equally
> exercised. The Triton path has been tested end to end on RTX 4090 and RTX
> 5090 with MiniMax H3. On Apple Silicon the kernel is verified against dense
> attention, the ComfyUI override plumbing is verified against ComfyUI 0.33.0,
> the figures below are measured, and a MiniMax H3 sampling run completes with
> the override installed. Long-run image quality on a Mac has not been assessed.

## Usage notes

Triton kernels are compiled on first use, so the first CUDA run will be slower.
The portable backend has no compile step.

Use `start_percent`, `end_percent`, and `tau` to balance generation quality and
speed.

## Apple Silicon

Triton has no Metal backend, so the CUDA kernels cannot run on a Mac at all.
`_torch_fwd.py` reimplements Algorithm 1 of the paper in plain tensor
operations: the same routing, the same diagonal-Gaussian threshold, the same
pooled-key approximation for unselected blocks. Only INT8 is absent, and that
exists to reach CUDA's INT8 tensor cores, which have no Metal equivalent.

### Test machine

| | |
|---|---|
| Chip | Apple M3 Max &mdash; 16-core CPU (12P + 4E), 40-core GPU |
| Memory | 128 GB unified |
| OS | macOS 15.4.1 (24E263), Metal 3 |
| Runtime | PyTorch 2.9.1, Python 3.12.9 |

### Measured

MiniMax H3's own attention shape -- `B=1 H=42 D=128`, bf16 -- against the
attention ComfyUI actually selects on this machine. That baseline matters: on
MPS ComfyUI picks `attention_sub_quad`, not
`scaled_dot_product_attention`, and the two are nowhere near each other. The
sub-quadratic path never materialises the score matrix, so it does not fall over
on long sequences the way plain SDPA does (SDPA asks for a 103 GB allocation at
32768 tokens and dies); it is simply slow.

| tokens | `attention_sub_quad` | Sol-Attn tau=1.3 | tau=2.0 |
|---|---|---|---|
| 4096 | 41 ms | 75 ms (0.55x) | 56 ms (0.73x) |
| 8192 | 646 ms | 463 ms (1.39x) | 296 ms (2.18x) |
| 16384 | 2407 ms | 1228 ms (1.96x) | 646 ms (3.72x) |
| 32768 | 79.5 s | 14.9 s (5.32x) | 1.9 s (**42x**) |

Three things worth reading off that table.

**Below ~8k tokens the sparse path loses here.** Routing, the gather and the
per-chunk bookkeeping cost more than the attention they remove. Note the *here*
-- see the end-to-end numbers below, where a 5607-token run came out ahead
anyway. Random tensors are the worst case for a method that exists to exploit
concentrated attention, so this figure is a warning, not a cutoff: the backend
logs a hint below it and still takes the call.

**The gain grows sharply with length.** By 32768 tokens the baseline is
degrading super-quadratically under memory pressure while the sparse path stays
bounded, which is where the 42x at tau=2.0 comes from -- most of it is the
baseline falling apart, not the kernel getting faster.

**tau dominates at scale.** 1.3 and 2.0 differ by 8x at 32768 tokens, far more
than at 8192, because the density difference multiplies against a much larger
sequence. Tune tau before anything else.

These are random-tensor figures, which is the pessimistic case: routing density
at a fixed tau is several times lower on structured inputs than on noise.

### End to end

The isolated numbers above measure one attention call. What a whole generation
does is a different question, so here is a real MiniMax H3 image-to-video run on
the same machine -- 480x832, 25 frames, 4 steps with the 4-step distilled LoRA,
which packs to 5607 tokens per attention call -- against
[TE-Speed-MiniMaxH3-MChip](https://github.com/kaelzhang/TE-Speed-MiniMaxH3-MChip),
the block cache this composes with:

| | wall clock | vs baseline |
|---|---|---|
| neither | 617.6 s | — |
| Sol-Attn alone (tau=1.3) | 526.8 s | 1.17x |
| block cache alone | 301.1 s | 2.05x |
| both | 376.0 s | 1.64x |

Two things this says that the microbenchmark did not.

Sol-Attn helped at 5607 tokens, below the break-even the isolated benchmark
predicted. Real attention is concentrated where random tensors are flat, so the
routing keeps far fewer blocks exact than the synthetic figures suggest.

**The two accelerators did not compose here** -- adding Sol-Attn on top of the
block cache cost 75 s rather than saving any. The plausible reading is that the
cache already skips half the blocks, so Sol-Attn's fixed per-call preprocessing
is amortised over less work while its benefit at this short sequence is small.
At this sequence length, use the block cache alone.

> [!WARNING]
> One run per configuration, on a machine that was not otherwise idle. Treat the
> ordering as real and the exact percentages as indicative. The composition
> result in particular deserves repeating before anyone builds on it.

Scores are computed in the input dtype. `bmm` rounds its output to that dtype
regardless, so at fp16 the kernel carries roughly 5e-3 relative error against
an fp32 reference -- the same order as the method's own approximation at
tau=1.3, and it does not grow with sparsity.

On the machine above the suite below passes on both `mps` and `cpu`: the
all-exact path reproduces dense attention to 3.6e-5 - 1.8e-3 across batch,
head-count, head-dim and ragged-length combinations, and routing density lands
at 15.6% / 6.7% / 2.3% for tau 1.0 / 1.5 / 2.0 against the paper's 16% / 7% /
2.7%.

To check a port or a change:

```
python test_torch_fwd.py          # best available device
python test_torch_fwd.py cpu
```

The suite pins the properties that matter rather than golden outputs: driving
tau far negative routes every block exact, so the kernel must reproduce dense
attention; routing density must track the paper's Gaussian-tail figures; sinks
must force their span exact; and no output may leave the range of V.

## Examples

### Test output

https://github.com/user-attachments/assets/8d9ed820-0417-4d68-9d1c-5199534bed3b

### SageAttention vs. Sol-Attn

<table>
<tr>
<td align="center"><b>SageAttention</b></td>
<td align="center"><b>Sol-Attn</b></td>
</tr>
<tr>
<td width="50%">
<video src="https://github.com/user-attachments/assets/27f201ea-6bfc-4f43-826c-51809eed9d15" controls muted loop></video>
</td>
<td width="50%">
<video src="https://github.com/user-attachments/assets/73f63d14-2166-4f62-b098-e817ec1d7704" controls muted loop></video>
</td>
</tr>
</table>

<img width="482" height="500" alt="Sol-Attn example result" src="https://github.com/user-attachments/assets/27ae9886-aa3e-4470-a507-3a7c52b24be5" />

## Citation

If you find Sol-Attn useful in your work, please cite the paper:

```bibtex
@article{li2026solattn,
  title={Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Sparsification},
  author={Li, Haopeng and Li, Yitong and Chen, Junsong and Ye, Tian and Liu, Haozhe and Yu, Jincheng and Wang, Duomin and Zhang, Ruihua and Xie, Zeke and Xie, Enze and Han, Song},
  journal={arXiv preprint arXiv:2607.24027},
  year={2026}
}
```

---

> This document and [README.zh-CN.md](README.zh-CN.md) are translations of each
> other and must be changed together: an edit to one needs the same edit in the
> other.
