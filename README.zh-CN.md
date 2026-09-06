<h1 align="center">ComfyUI-SolAttn</h1>

<h4 align="center">
  ComfyUI 上的 Sol-Attn，CUDA 和 Apple Silicon 都能跑
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

[Sol-Attn](https://arxiv.org/abs/2607.24027) 是一种免训练的稀疏注意力，用来加速图像和视频生成。这个扩展把它接进 ComfyUI，带两个后端：

| | CUDA | Apple Silicon / CPU |
|---|---|---|
| 实现 | Triton kernel | 纯 PyTorch |
| 数据类型 | bf16 | fp16 / bf16 / fp32 |
| head dim | 128 | 任意 |
| INT8 QK/PV | 有 | 没有 |
| TMA descriptor | 需要 SM90+ | 用不上 |

用哪个后端是每次调用时看传进来的张量自己定的，你什么都不用配。某个后端做不了的选项，它会记一条日志然后忽略掉，而不是直接报错。路由、阈值、池化键近似、conditioning sink 这些在两个后端上是同一套算法。

**Apple Silicon 是支持的，而且测过** —— 数据和测试机器见下面的 [Apple Silicon](#apple-silicon)。

> [!NOTE]
> 这个项目还在演进，两个后端的验证程度不一样。Triton 那条路已经在 RTX 4090 和 5090 上配 MiniMax H3 做过端到端测试。Apple Silicon 这边验证过的是：kernel 跟 dense attention 结果一致、ComfyUI 的挂载链路在 0.33.0 上能用、下面那些实测数字、以及装上 override 之后 MiniMax H3 能跑完一次采样。Mac 上长时间出图的质量还没评估过。

## 用法

Triton kernel 第一次用的时候要编译，所以 CUDA 上首次运行会慢一些。纯 PyTorch 那条路没有编译这一步。

`start_percent`、`end_percent` 和 `tau` 用来在质量和速度之间取舍。

## Apple Silicon

Triton 没有 Metal 后端，所以 CUDA kernel 在 Mac 上压根跑不起来。`_torch_fwd.py` 用普通张量算子把论文的 Algorithm 1 重写了一遍：一样的路由、一样的对角高斯阈值、一样的用池化键近似没被选中的块。只有 INT8 没做——它的意义是喂 CUDA 的 INT8 tensor core，Metal 上没这个东西。

### 测试机器

| | |
|---|---|
| 芯片 | Apple M3 Max，16 核 CPU（12 性能核 + 4 能效核），40 核 GPU |
| 内存 | 128 GB 统一内存 |
| 系统 | macOS 15.4.1 (24E263)，Metal 3 |
| 运行时 | PyTorch 2.9.1，Python 3.12.9 |

### 实测

用 MiniMax H3 自己的注意力形状（`B=1 H=42 D=128`，bf16），对比的是 ComfyUI 在这台机器上**实际会选**的那个注意力实现。

这个基线选对很重要：MPS 上 ComfyUI 选的是 `attention_sub_quad`，不是 `scaled_dot_product_attention`，两者差得远。次二次那条路从不把完整的分数矩阵展开，所以它不会像裸 SDPA 那样在长序列上直接挂掉（SDPA 在 32768 tokens 时要申请 103 GB，然后就没了），它只是慢。

| tokens | `attention_sub_quad` | Sol-Attn tau=1.3 | tau=2.0 |
|---|---|---|---|
| 4096 | 41 ms | 75 ms (0.55x) | 56 ms (0.73x) |
| 8192 | 646 ms | 463 ms (1.39x) | 296 ms (2.18x) |
| 16384 | 2407 ms | 1228 ms (1.96x) | 646 ms (3.72x) |
| 32768 | 79.5 s | 14.9 s (5.32x) | 1.9 s (**42x**) |

这张表里有三件事值得注意。

**8k tokens 以下，在这个基准里稀疏路径是亏的。** 路由、gather、分块记账这些开销比省下来的注意力还多。注意「在这个基准里」——下面端到端那组数据里，5607 tokens 反而是赚的。随机张量对一个专门吃「注意力集中」的方法来说是最坏情况，所以这个数字只能当提醒，不能当硬线：低于它后端会打一条日志，但照样接。

**序列越长收益涨得越快。** 到 32768 的时候基线已经在内存压力下劣化得比平方还厉害，而稀疏路径的开销是有上限的。tau=2.0 那个 42x 里大部分是基线自己崩了，不是 kernel 突然变快。

**规模一大，tau 就成了最重要的旋钮。** 32768 下 tau 取 1.3 和 2.0 差了 8 倍，比 8192 时的差距大得多，因为密度的差异是乘在一个大得多的序列上的。调优先调 tau。

上面这些是随机张量测出来的，属于最坏情况。真实数据的注意力集中得多，同样的 tau 下路由密度会低好几倍。

### 端到端

上面那张表测的是单次注意力调用。整个生成流程是另一回事，所以下面是同一台机器上真跑的一次 MiniMax H3 图生视频：480x832、25 帧、4 步（配 4 步蒸馏 LoRA），打包后每次注意力调用是 5607 tokens。对比对象是 [TE-Speed-MiniMaxH3-MChip](https://github.com/kaelzhang/TE-Speed-MiniMaxH3-MChip)，也就是能跟它叠加的那个块缓存：

| | 总耗时 | 相对基线 |
|---|---|---|
| 都不开 | 617.6 s | — |
| 只开 Sol-Attn (tau=1.3) | 526.8 s | 1.17x |
| 只开块缓存 | 301.1 s | 2.05x |
| 两个都开 | 376.0 s | 1.64x |

这里有两件微基准没告诉我们的事。

Sol-Attn 在 5607 tokens 上是赚的，而孤立基准预测这个长度应该亏。真实的注意力是集中的，随机张量是平的，所以路由实际保留为精确计算的块比合成数据少得多。

**两个加速器在这里没能叠加起来**——在块缓存之上再加 Sol-Attn，非但没省，反而多花了 75 秒。比较合理的解释是：缓存已经跳掉了一半的块，Sol-Attn 每次调用的固定预处理开销被摊到更少的实际计算上，而它在这个短序列上本来收益就不大。这个长度下，只开块缓存就好。

> [!WARNING]
> 每种配置只跑了一次，机器当时也不是完全空闲。这几个数的**排序**可以信，具体百分比只能当参考。尤其是「叠加反而变慢」这条，谁要在它上面做判断，最好先重复测几次。

分数是按输入的 dtype 算的。`bmm` 反正会把输出舍入到那个 dtype，所以 fp16 下 kernel 相对 fp32 参考大概有 5e-3 的相对误差——跟这个方法本身在 tau=1.3 时的近似误差是一个量级，而且不会随着稀疏度增长。

在上面那台机器上，下面的测试在 `mps` 和 `cpu` 都过：全精确路径在各种 batch、head 数、head dim、非整块长度的组合下，复现 dense attention 的误差在 3.6e-5 到 1.8e-3 之间；tau 取 1.0 / 1.5 / 2.0 时路由密度分别是 15.6% / 6.7% / 2.3%，论文给的是 16% / 7% / 2.7%。

想验证移植或者改动有没有问题：

```
python test_torch_fwd.py          # 自动挑设备
python test_torch_fwd.py cpu
```

测试盯的不是固定输出，而是几条必须成立的性质：把 tau 压到极负，每个块都会走精确路径，这时 kernel 必须跟 dense attention 一模一样；路由密度得对得上论文的高斯尾数字；sink 必须真的让它覆盖的范围走精确路径；任何输出都不能跑到 V 的取值范围之外。

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

如果 Sol-Attn 对你有帮助，请引用这篇论文：

```bibtex
@article{li2026solattn,
  title={Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Sparsification},
  author={Li, Haopeng and Li, Yitong and Chen, Junsong and Ye, Tian and Liu, Haozhe and Yu, Jincheng and Wang, Duomin and Zhang, Ruihua and Xie, Zeke and Xie, Enze and Han, Song},
  journal={arXiv preprint arXiv:2607.24027},
  year={2026}
}
```

---

> 这份文档和 [readme.md](readme.md) 是同一份东西的两个语言版本，改一个必须同时改另一个。
