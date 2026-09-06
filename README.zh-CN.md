<h1 align="center">ComfyUI-SolAttn</h1>

<h4 align="center">
  ComfyUI 的实验性 Sol-Attn &mdash; 支持 CUDA 与 Apple Silicon
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
  <a href="readme.md">English</a> &nbsp;|&nbsp; <b>简体中文</b>
</p>

---

## 概述

[Sol-Attn](https://arxiv.org/abs/2607.24027) 是一种免训练的稀疏注意力方法，用于加速图像与视频生成。本社区扩展将其集成进 ComfyUI，提供两个可互换的后端：

| | CUDA | Apple Silicon / CPU |
|---|---|---|
| 实现 | Triton kernel | 可移植 PyTorch |
| 数据类型 | bf16 | fp16 / bf16 / fp32 |
| head dim | 128 | 任意 |
| INT8 QK/PV | 支持 | 不支持 |
| TMA descriptor | SM90+ | 不适用 |

后端按调用逐次从传入的张量自行选择，无需任何配置；某个后端不具备的选项会记录一次日志并被忽略，而不是报错。路由、阈值、池化键近似以及 conditioning sink 在两个后端上是同一套计算。

**Apple Silicon 已支持并完成实测** &mdash; 实测数据与测试机器见下方 [Apple Silicon](#apple-silicon) 章节。

> [!NOTE]
> 本项目仍在演进中，两个后端的验证程度并不对等。Triton 路径已在 RTX 4090 与 RTX 5090 上配合 MiniMax H3 做过端到端测试。在 Apple Silicon 上，已验证的是：kernel 与 dense attention 的一致性、ComfyUI 挂载链路对 ComfyUI 0.33.0 的适配、下方的实测数据，以及一次装载了 override 的 MiniMax H3 采样运行可以完成。Mac 上的长时间出图质量尚未评估。

## 使用说明

Triton kernel 在首次使用时编译，所以 CUDA 上的第一次运行会更慢。可移植后端没有编译步骤。

用 `start_percent`、`end_percent` 和 `tau` 在生成质量与速度之间取舍。

## Apple Silicon

Triton 没有 Metal 后端，所以 CUDA kernel 在 Mac 上根本无法运行。`_torch_fwd.py` 用纯张量算子重新实现了论文的 Algorithm 1：同样的路由、同样的对角高斯阈值、同样的对未选中块的池化键近似。唯一缺失的是 INT8，而它的存在意义是喂 CUDA 的 INT8 tensor core，Metal 上没有对应硬件。

### 测试机器

| | |
|---|---|
| 芯片 | Apple M3 Max &mdash; 16 核 CPU（12 性能核 + 4 能效核），40 核 GPU |
| 内存 | 128 GB 统一内存 |
| 系统 | macOS 15.4.1 (24E263)，Metal 3 |
| 运行时 | PyTorch 2.9.1，Python 3.12.9 |

### 实测数据

采用 MiniMax H3 自身的注意力形状 &mdash; `B=1 H=42 D=128`，bf16 &mdash; 对比 ComfyUI 在这台机器上**实际选用**的注意力实现。这个基线的选择很关键：在 MPS 上 ComfyUI 选的是 `attention_sub_quad`，而不是 `scaled_dot_product_attention`，两者差距极大。次二次路径从不物化完整的分数矩阵，所以它不会像裸 SDPA 那样在长序列上崩溃（SDPA 在 32768 tokens 时要申请 103 GB 并直接失败）；它只是慢。

| tokens | `attention_sub_quad` | Sol-Attn tau=1.3 | tau=2.0 |
|---|---|---|---|
| 4096 | 41 ms | 75 ms (0.55x) | 56 ms (0.73x) |
| 8192 | 646 ms | 463 ms (1.39x) | 296 ms (2.18x) |
| 16384 | 2407 ms | 1228 ms (1.96x) | 646 ms (3.72x) |
| 32768 | 79.5 s | 14.9 s (5.32x) | 1.9 s (**42x**) |

这张表有三点值得读出来。

**8k tokens 以下稀疏路径是负收益。** 路由、gather 以及分块记账的开销超过了它省下的注意力计算。因此该后端会直接拒绝更短的调用、交还给宿主的注意力实现，所以装上这个节点不会让短调用变慢。

**收益随序列长度急剧增长。** 到 32768 tokens 时，基线在内存压力下已呈超平方劣化，而稀疏路径的开销是有界的——tau=2.0 那个 42x 主要来自基线的崩坏，而不是 kernel 变快了。

**规模越大 tau 越是主导因素。** 在 32768 tokens 下 tau=1.3 与 2.0 相差 8 倍，远大于 8192 时的差距，因为密度差异是乘在一个大得多的序列上的。调优时优先调 tau。

以上是随机张量的数字，属于最悲观情况：在结构化输入上，同一 tau 下的路由密度会比噪声低数倍。

分数在输入 dtype 下计算。`bmm` 无论如何都会把输出舍入到该 dtype，所以在 fp16 下 kernel 相对 fp32 参考约有 5e-3 的相对误差——与该方法自身在 tau=1.3 时的近似误差同量级，且不随稀疏度增长。

在上述机器上，下面的测试套件在 `mps` 与 `cpu` 上均通过：全精确路径在各种 batch、head 数、head dim 与非整块长度组合下复现 dense attention 到 3.6e-5 ~ 1.8e-3；tau 取 1.0 / 1.5 / 2.0 时路由密度分别落在 15.6% / 6.7% / 2.3%，对应论文的 16% / 7% / 2.7%。

检查移植或改动是否正确：

```
python test_torch_fwd.py          # 自动选择可用设备
python test_torch_fwd.py cpu
```

该套件锁定的是关键性质而非固定输出：把 tau 压到极负会使每个块都走精确路径，此时 kernel 必须复现 dense attention；路由密度必须吻合论文的高斯尾数值；sink 必须强制其跨度精确；任何输出都不得超出 V 的取值范围。

## 示例

### 测试输出

https://github.com/user-attachments/assets/8d9ed820-0417-4d68-9d1c-5199534bed3b

### SageAttention 对比 Sol-Attn

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

## 引用

如果 Sol-Attn 对你的工作有帮助，请引用该论文：

```bibtex
@article{li2026solattn,
  title={Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Sparsification},
  author={Li, Haopeng and Li, Yitong and Chen, Junsong and Ye, Tian and Liu, Haozhe and Yu, Jincheng and Wang, Duomin and Zhang, Ruihua and Xie, Zeke and Xie, Enze and Han, Song},
  journal={arXiv preprint arXiv:2607.24027},
  year={2026}
}
```

---

> 本文档与 [readme.md](readme.md) 互为翻译，两者必须同步修改：改动其一时，另一份也要一并更新。
