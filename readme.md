<h1 align="center">ComfyUI-SolAttn</h1>

<h4 align="center">
  Experimental Sol-Attn for ComfyUI, on CUDA and Apple Silicon
</h4>

<p align="center">
  <a href="https://arxiv.org/abs/2607.24027"><img src="https://img.shields.io/badge/📄_Paper-arXiv-b31b1b?style=flat-square" alt="Paper"/></a>
  <a href="https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends/sol_attn"><img src="https://img.shields.io/badge/💻_Code-Sol--Attn-76b900?style=flat-square" alt="Code"/></a>
  <a href="https://nvlabs.github.io/Sana/Sol-Attn/"><img src="https://img.shields.io/badge/🌐_Project-Page-blue?style=flat-square" alt="Project Page"/></a>
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

> [!NOTE]
> This project is a work in progress. The Triton path has been tested on RTX
> 4090 and RTX 5090; the portable path on an M3 Max. Both with MiniMax H3.

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

Measured on an M3 Max, `B=1 H=24 D=128`, against PyTorch's MPS attention:

| tokens | `scaled_dot_product_attention` | Sol-Attn tau=1.3 | tau=2.0 |
|---|---|---|---|
| 4096 | 49 ms | 33 ms | 29 ms |
| 8192 | 204 ms | 83 ms | 112 ms |
| 16384 | 1669 ms | 663 ms | 365 ms |
| 32768 | out of memory | 2181 ms | 1081 ms |

Two things are worth reading off that table. The gain grows with sequence
length, because MPS attention materialises the whole score matrix -- at 32768
tokens it asks for a 103 GB allocation and dies, while the sparse path streams
query blocks in bounded memory and finishes. And these are random-tensor
figures, which are the pessimistic case: routing density at a fixed tau is
several times lower on structured inputs than on noise, so real sampling sits
further ahead than this.

Scores are computed in the input dtype. `bmm` rounds its output to that dtype
regardless, so at fp16 the kernel carries roughly 5e-3 relative error against
an fp32 reference -- the same order as the method's own approximation at
tau=1.3, and it does not grow with sparsity.

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
