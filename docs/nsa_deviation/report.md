# 项目二：NSA 稀疏偏差分析 — 理论稀疏率 vs 实际加速比

> NVIDIA L4 (24GB) | PyTorch 2.6.0+cu124 | FlexAttention + Triton 3.2.0

## 1. 研究问题

理论上过滤掉 90% 的 Attention 计算，应该快 10 倍。但实际 GPU kernel 运行中，由于内存对齐、block 调度开销，往往达不到。这个"偏差"在哪里？

以 DeepSeek 的 NSA（Native Sparse Attention）为代表，前沿正在抛弃单纯的"滑动窗口"，转向"局部窗口 + 全局 Sink + 动态选择块"的混合稀疏模式。

## 2. 实验设计

### 2.1 NSA-like Mask 实现

模拟 NSA 的三层注意力结构：

```
NSA Mask = Global Sink + Local Window + Dynamic Block Selection
```

- **Global Sink**: 前 N 个 token 永远可见（attention sink）
- **Local Window**: 最近 M 个 token 永远可见（局部窗口）
- **Dynamic Block**: 中间部分按 block 选择性可见

### 2.2 实验组

| 实验 | 测量内容 |
|------|---------|
| Exp1 | NSA-like pattern vs baseline（FlexAttention dense causal）|
| Exp2 | BLOCK_SIZE (64/128/256) 对 NSA 性能的影响 |
| Exp3 | 理论稀疏率 vs 实际加速比偏差曲线（8 个窗口大小）|

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
