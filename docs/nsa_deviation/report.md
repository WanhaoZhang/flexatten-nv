# 项目二：NSA 稀疏偏差分析 — 理论稀疏率 vs 实际加速比

> NVIDIA L4 (24GB) | PyTorch 2.6.0+cu124 | FlexAttention + Triton 3.2.0

## 1. 研究背景

### 1.1 稀疏注意力的承诺与现实

标准 Self-Attention 的计算复杂度为 O(S²)，随序列长度增长迅速成为瓶颈。理论上，如果只计算"重要"的 attention 连接，跳过其余 90%，就能获得 10x 加速。这就是**稀疏注意力**的核心承诺。

然而，GPU 并非"算多少就快多少"的理想机器。实际加速受制于：
- **内存对齐**：GPU 以 block（如 128×128）为单位加载和计算，即使 block 内只有一个有效元素也要加载整个 block
- **Kernel launch 开销**：每次 kernel 调用有固定开销（~5-10μs），粒度过细反而拖慢
- **带宽瓶颈**：当计算量已经足够小，性能上限由内存带宽决定，减少计算量不再有帮助

这导致**理论稀疏率与实际加速比之间存在巨大偏差**——这正是本实验要定量分析的问题。

### 1.2 NSA：Native Sparse Attention

DeepSeek 的 NSA（Native Sparse Attention）代表了当前稀疏注意力的前沿设计。它抛弃了简单的"滑动窗口"，采用**三层混合架构**：

```
Attention(x) = GlobalSink(x) + LocalWindow(x) + DynamicBlock(x)
```

- **Global Sink**：前 N 个 token（如 64 个）作为"attention sink"，所有 token 都能访问。保持全局信息流
- **Local Window**：每个 token 只关注最近 M 个相邻 token（如 256-512）。捕获局部依赖
- **Dynamic Block Selection**：中间部分的 token 按 block 粒度（如 64 个一组）选择性访问，由路由网络决定哪些 block 重要

这种设计的理论稀疏率可以非常高（如 90%+），同时保留关键信息传递通道。

### 1.3 研究目标

本实验的核心目标是**定量揭示稀疏注意力的理论加速与实际加速之间的偏差**：

1. **偏差曲线**：从 0% 到 93.8% 稀疏率，实际加速比如何变化？偏差有多大？
2. **NSA 模式验证**：Global Sink + Local Window + Dynamic Block 的混合模式在 FlexAttention/Triton 上能否获得加速？
3. **Block 大小影响**：BLOCK_SIZE（64/128/256）是否影响稀疏性的利用效率？
4. **根因定位**：为什么稀疏性没有转化为加速？是 Triton kernel 的限制还是 GPU 硬件的限制？

---

## 2. 实验设计

### 2.1 NSA-like Mask 实现

模拟 NSA 的三层注意力结构：

```
NSA Mask = Global Sink + Local Window + Dynamic Block Selection
```

- **Global Sink**: 前 N 个 token 永远可见（attention sink）
- **Local Window**: 最近 M 个 token 永远可见（局部窗口）
- **Dynamic Block**: 中间部分按 block 选择性可见（每 K 个 block 保留 1 个）

### 2.2 实验组与目标

| 实验 | 目标 | 方法 |
|------|------|------|
| Exp1 | 对比 NSA-like pattern 与 dense causal 的实际性能 | seq=2048, 5 种 NSA 配置 |
| Exp2 | 验证 BLOCK_SIZE 对稀疏性利用的影响 | BLOCK_SIZE=64/128/256 |
| Exp3 | 绘制理论稀疏率 vs 实际加速比的偏差曲线 | 8 个窗口大小，稀疏率 0%-93.8% |

---

## 3. 核心发现

### 发现 1：稀疏性完全没有转化为加速

| 配置 | 理论加速 | 实际加速 | 偏差 |
|------|---------|---------|------|
| Dense Causal | 1.00x | 0.89x | 11.4% |
| NSA Sink64 W256 | 1.33x | 0.90x | 32.7% |
| NSA Sink64 W256 B4 | 1.67x | 0.90x | 46.3% |
| NSA Sink128 W512 | 1.82x | 0.89x | 50.8% |
| NSA Sink128 W512 B4 | 2.50x | 0.89x | **64.4%** |
| SW-256 | 1.14x | 0.90x | 21.6% |
| SW-512 | 1.33x | 0.90x | 32.5% |
| SW-1024 | 2.00x | 0.89x | 55.7% |

**所有 NSA 配置的实际延迟几乎完全相同（~53ms），无论理论稀疏率如何。**

### 发现 2：偏差曲线 — 效率随稀疏率线性退化

| 理论稀疏率 | 理论加速 | 实际加速 | 效率（实际/理论） |
|-----------|---------|---------|-----------------|
| 93.8% | 16.0x | 1.02x | **6.3%** |
| 87.5% | 8.0x | 1.01x | 12.7% |
| 81.2% | 5.3x | 1.01x | 19.0% |
| 75.0% | 4.0x | 1.01x | 25.2% |
| 62.5% | 2.7x | 1.00x | 37.7% |
| 50.0% | 2.0x | 1.00x | 50.1% |
| 25.0% | 1.3x | 1.00x | 74.8% |
| 0.0% | 1.0x | 0.99x | 99.3% |

**93.8% 的理论稀疏率只带来了 1.02x 的实际加速（效率 6.3%）。** 这是"稀疏率谎言"的定量证据。

### 发现 3：BLOCK_SIZE 完全无影响

| BLOCK_SIZE | Forward | Backward | Total |
|-----------|---------|----------|-------|
| 64 | 53.63ms | 59.73ms | 113.36ms |
| 128 | 53.55ms | 59.74ms | 113.28ms |
| 256 | 53.61ms | 59.80ms | 113.40ms |

**三种 block 大小的延迟差异 < 0.2ms。** 这说明 Triton kernel 的 block 遍历是固定的，没有根据 BlockMask 的稀疏模式做任何优化。

## 4. 根因分析

为什么 FlexAttention 的 Triton kernel 不能利用稀疏性？

1. **Triton kernel 的 block 遍历是静态的**：即使 BlockMask 标记了某些 block 为"空"，kernel 仍然遍历所有可能的 block 位置，只是对空 block 跳过计算。但遍历本身的开销接近于计算。

2. **BlockMask 的 BCSR 索引查找有开销**：`kv_num_blocks` + `kv_indices` 的间接寻址引入了额外的全局内存访问，部分抵消了跳过计算节省的时间。

3. **L4 (sm89) 的 Tensor Core 调度**：在 128x128 的 block 上，L4 的 Tensor Core 已经饱和了内存带宽，减少计算量不会减少内存访问。

4. **Triton kernel 编译时未感知稀疏模式**：所有 mask 共享同一个 Triton kernel 模板，只是在运行时通过不同的 BlockMask 数据驱动。编译器无法针对特定稀疏模式做优化。

## 5. 结论

1. **"稀疏率是谎言"在 FlexAttention/Triton 路径上成立**：93.8% 稀疏率的实际加速仅 1.02x（效率 6.3%）

2. **根因是 Triton kernel 的架构限制**：静态 block 遍历 + 运行时 BlockMask 查找 + 无法编译期优化

3. **真正的稀疏加速需要手写 kernel**：NSA 论文中的加速来自手写的 CUDA kernel，可以完全跳过空 block 的内存访问和计算

4. **对 L4 的实际建议**：
   - FlexAttention 用于**快速验证**稀疏模式的正确性
   - 需要高性能时，必须手写 CUDA kernel（如 FlashInfer、ThunderKittens）
   - 或等待 PyTorch FlexAttention 的 Triton kernel 优化

## 6. 复现

```bash
cd ~/flexatten-nv/docs/nsa_deviation
python nsa_deviation.py    # 生成 results/*.json (~15min)
```

---

*实验日期：2026-04-28 | NVIDIA L4 | PyTorch 2.6.0+cu124*
