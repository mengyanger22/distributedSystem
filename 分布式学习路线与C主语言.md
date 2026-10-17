# 分布式系统学习路线 与 C++ 是否作主语言

> 本文合并了两个问题及其原回答，回答正文保持原样。

---

# 问题一：怎么开始分布式系统学习，包括 GPU 集群训练、多卡通信

## 先说路线总览

分布式训练和你前面学的单卡优化是**同一类问题往外扩一层**：单卡优化的核心问题是"数据在内存层级里怎么走"，分布式训练的核心问题是"数据在多张卡之间怎么走、通信能不能和计算重叠"。建议按下面五步走，前两步你用笔记本就能做。

## 先处理硬件现实

你只有一块 RTX 3050，而 **NCCL 不支持让多个 rank 共用同一块 GPU**：用多进程在一块卡上跑 NCCL，会报 "Duplicate GPU detected"；论坛里的建议是改用 gloo。所以：

- **学概念、写通信代码**：用 `gloo` 后端，在 CPU 上起多个进程模拟多卡，语义和 NCCL 一致，只是慢。
- **测真实的 NCCL 带宽、NVLink/PCIe 拓扑**：需要至少 2 块 GPU，可以按小时租云上的多卡机器，只用在第 3 步。论坛上有人在同一台两卡机器上两种后端都能跑，NCCL 明显更快。

## 第 0 步：通信原语（1 周左右）

先搞懂这六个操作的语义：`broadcast`、`reduce`、`all-reduce`、`all-gather`、`reduce-scatter`、`all-to-all`。最小实验：

```python
# allreduce_demo.py
import torch, torch.distributed as dist
dist.init_process_group("gloo")
rank = dist.get_rank()
x = torch.tensor([float(rank)])
dist.all_reduce(x, op=dist.ReduceOp.SUM)
print(rank, x)
```

```bash
torchrun --nproc_per_node=4 allreduce_demo.py
```

接着**自己用 `send/recv` 实现一个 ring all-reduce**，和库函数的结果对比。ring all-reduce 里每个 rank 收发的数据量大约是 `2(N-1)/N × 数据大小`，这个公式后面估算通信时间会反复用到。

## 第 1 步：数据并行 DDP（1–2 周）

- 跟着 PyTorch 官方的 DDP 入门教程走一遍。
- 理解两件事：梯度为什么要 all-reduce、**all-reduce 为什么能在反向传播期间和计算重叠**（Ultra-Scale Playbook 里也讲到了这点）。
- 练习：用 gloo 起 4 个 CPU 进程，**手写一个不用 DDP 的数据并行**（每个进程算局部梯度，自己调用 all-reduce 求平均再更新），验证和单进程用大 batch 的结果一致。

## 第 2 步：为什么需要切模型（2–3 周）

这是 LLM 训练的核心。建议主线读 **Hugging Face 的 Ultra-Scale Playbook**：它基于最多 512 张 GPU 上的 4000 多次实验，假设你了解 LLM 基础、但不熟悉分布式训练，从单卡讲到数百上千卡。

要掌握的内容：

1. **显存账**：参数、梯度、优化器状态各占多少。用 Adam 做混合精度训练时，每个参数的总开销是个很大的数字（ZeRO 论文里的经典算法，我记得是约 16 字节每参数，具体请以原文为准）。你可以手算一个 7B 模型要多少显存，再用一个小模型验证。
2. **ZeRO / FSDP**：把优化器状态、梯度、参数依次切开，对应的通信从 all-reduce 变成 reduce-scatter + all-gather。
3. **张量并行（TP）**：Megatron 里按列、按行切线性层，前向各做一次 all-reduce。
4. **流水线并行（PP）**：气泡怎么产生、怎么用 micro-batch 缩小。
5. 序列并行、专家并行（MoE 的 all-to-all），了解即可。

另一份可选材料是 Google DeepMind 的 "How to Scale Your Model"，免费，但**面向 TPU、偏数学、难度较高**，适合以后补 roofline 和 sharding 的定量分析，不建议作为第一本。

## 第 3 步：NCCL 和真实多卡（需要租机器）

租一台 2–8 卡的机器，做这几件事：

- 编译并运行 **nccl-tests**（`all_reduce_perf` 等），扫描不同消息大小，画出带宽曲线，对比理论值。
- 设置 `NCCL_DEBUG=INFO`，读日志里的拓扑探测结果，弄清走的是 NVLink、PCIe 还是网络。
- 理解 `algbw` 和 `busbw` 的区别，对照 ring 与 tree 算法。
- 用 `nvidia-smi topo -m` 看卡间连接关系。

这一步可以花几个小时就做完，不必长期租。

## 第 4 步：集群层面

了解即可，等有机会接触真实集群再深入：Slurm 调度、容器、checkpoint 与故障恢复、InfiniBand/RoCE 网络。

## 第 5 步：读源码

按顺序：`torch.distributed` 的 DDP 和 FSDP → Megatron-LM → DeepSpeed → NCCL 本身。边读边回头对照第 2 步的概念。

## 并行的一条线：经典分布式系统

*Designing Data-Intensive Applications*（我们之前提过）和 MIT 的分布式系统课程（有 Raft 实验），讲一致性、容错。这条线和 GPU 训练**不是一回事**，更偏服务端；我的判断是，做训练和推理 infra，先学上面五步的收益更直接，这条线按兴趣和需要补充即可。

## 三个可以马上做的小项目

1. **CPU 版数据并行**：gloo 起 4 进程，手写 all-reduce 梯度，对比单进程结果。
2. **显存估算器**：写一个函数，输入参数量和并行策略，输出每张卡的显存，再用小模型实测对账。
3. **（租机器）NCCL 带宽实验**：扫描消息大小，画 `busbw` 曲线，找出带宽饱和的拐点。

## 一点建议

别等有了多卡再开始：第 0、1 步和第 2 步的大部分内容，在你的笔记本上就能学完，真正需要硬件的只有第 3 步。需要的话，我可以帮你把第 1 步的"手写数据并行"写成一个可直接运行的练习骨架，你来补全关键部分。

---

# 问题二：C++ 作为主语言吗

## 先说结论

**不建议把 C++ 当成唯一的主语言，更合适的是 "Python 每天用，C++ 往深处挖"。** 两者在 AI infra 里分工不同，要看你站在哪一层。

## 按层次看

| 你在做什么 | 主要语言 |
|---|---|
| 训练脚本、实验、`torchrun`、DDP/FSDP 配置、数据流水线 | Python |
| 调用库做分布式训练、读论文复现 | Python |
| 写 CUDA kernel、CUTLASS、NCCL 相关、PyTorch 底层算子、推理引擎的核心 | C++ |
| 写 Triton / CuTe DSL 这类 Python 嵌入式 kernel | Python 语法，但底下是编译器和 C++ 工具链 |
| 集群调度、平台工具 | 据我了解常见 Go、Python、shell，这块我没有核实 |

所以如果你的目标是我们前面一直聊的 kernel、通信、推理系统，**C++ 是你需要练到比较深的语言**，但你日常的实验、测试、对拍几乎都是在 Python 里完成的。

## 招聘信息的信号

- NVIDIA 的 Triton 编译器岗位要求扎实的 C++ 和性能分析调试能力。
- NVIDIA 的 CUTLASS kernel 岗位，工作内容是用 CUDA C++ 和 Python DSL 写 Tensor Core kernel。
- AMD 的 Triton kernel 岗位要求 C/C++ 的 AI 开发经验，以及 Triton、CUDA 或 HIP。
- Reddit 上有人说，大多数 kernel 岗位仍然把 C++17、CuTe、CUTLASS 列为硬性要求。

四条都是"C++ 加 Python 并列"，没有哪个岗位只要其中一个。

## 你的路线里怎么分配

1. **现在到分布式学习的第 0–2 步**：主要写 Python（`torch.distributed`、DDP、手写 all-reduce），这些和 C++ 无关。
2. **kernel 和通信库部分**：继续用 CUDA C++，这是你正在学的 PMPP 路线。
3. **把两者接起来的技能很有价值**：用 PyTorch 的 `torch.utils.cpp_extension` 把自己写的 CUDA 算子编译成 Python 可调用的函数。这个我印象里是标准做法，具体用法请以 PyTorch 官方文档为准。做一次，你就同时用上了 C++ 和 Python。

## C++ 学到什么程度

不需要把 C++ 学成"语言律师"，目标是**能读懂和修改 CUTLASS、PyTorch 的 ATen、vLLM 这类项目的代码**。优先掌握：

- 指针、引用、值语义和移动语义
- RAII 和智能指针
- 模板基础（函数模板、类模板、非类型模板参数，这些你已经在 `cuda::tiles` 的 `shape<64>` 里见到了）
- `constexpr`、`inline`、命名空间（我们前面都聊过）
- STL 容器和算法
- CMake 的基本用法

模板元编程、复杂的 SFINAE 之类，等读源码读到了再补。

## 一句话

**把 C++ 当成"深度语言"来练，把 Python 当成"工作语言"来用，两个都要会。** 只押注一个，要么写不了底层，要么做不了实验。
