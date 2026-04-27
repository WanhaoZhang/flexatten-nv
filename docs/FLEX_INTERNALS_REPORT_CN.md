# FlexAttention 原理深度剖析与性能实验报告

> 实验环境: NVIDIA L4 (24GB) | PyTorch 2.6.0+cu124 | FP16  
> 实验代码: `src/flex_internals_experiment.py` | 绘图: `src/plot_flex_internals.py`  
> 实验数据: `data/flex_internals_results.json` | 图表: `docs/figures/flex_fig*.png`

---

## 目录

1. [引言：为什么需要 FlexAttention](#1-引言为什么需要-flexattention)
2. [FlexAttention 的设计哲学](#2-flexattention-的设计哲学)
3. [核心 API 详解](#3-核心-api-详解)
4. [内部执行流程：从 Python 到 Triton](#4-内部执行流程从-python-到-triton)
5. [实验一：BlockMask 内部结构解剖](#5-实验一blockmask-内部结构解剖)
6. [实验二：score_mod 编译追踪](#6-实验二score_mod-编译追踪)
7. [实验三：稀疏性 vs 性能](#7-实验三稀疏性-vs-性能)
8. [实验四：mask_mod + score_mod 组合](#8-实验四mask_mod--score_mod-组合)
9. [实验五：torch.compile 编译开销](#9-实验五torchcompile-编译开销)
10. [实验六：FlexAttention vs SDPA 延迟对比](#10-实验六flexattention-vs-sdpa-延迟对比)
11. [实验七：逐步计算追踪（完整例子）](#11-实验七逐步计算追踪完整例子)
12. [实验八：不同注意力模式性能剖析](#12-实验八不同注意力模式性能剖析)
13. [性能分析：为什么 FlexAttention 慢](#13-性能分析为什么-flexattention-慢)
14. [性能分析：FlexAttention 什么时候快](#14-性能分析flexattention-什么时候快)
15. [PyTorch 2.6.0 的已知限制与改进](#15-pytorch-260-的已知限制与改进)
16. [结论](#16-结论)

---

## 1. 引言：为什么需要 FlexAttention

### 1.1 标准注意力的问题

标准的缩放点积注意力（SDPA）计算公式为：

```
Output = softmax(Q @ K^T / sqrt(d)) @ V
```

这个公式非常简洁，但实际模型中经常需要在其中插入各种修改：

| 修改类型 | 例子 | 传统实现方式 |
|----------|------|------------|
| 因果掩码 | Causal mask | 传入 `is_causal=True` |
| 滑动窗口 | Sliding window | 手写 mask 矩阵 |
| 位置偏置 | ALiBi, RoPE | 手写偏置矩阵 |
| 分数缩放 | Tanh softcap | 手写缩放逻辑 |
| 前缀注意力 | Prefix LM | 复杂的 mask 拼接 |
| 稀疏模式 | Dilated, block-local | 自定义 CUDA kernel |

每种修改都需要手写不同的实现。更糟糕的是，当你需要**同时**使用多种修改时（比如 causal + softcap + sliding window），几乎没有现成的 kernel 可以直接用。

### 1.2 FlexAttention 的承诺

PyTorch 2.5 引入了 **FlexAttention**，它的核心承诺是：

> **用纯 Python 函数描述注意力修改，框架自动编译为高效的 Triton kernel。**

你只需要写一个简单的函数，描述你想怎么修改注意力分数，PyTorch 就会自动把它编译成可以在 GPU 上高效运行的代码。

---

## 2. FlexAttention 的设计哲学

### 2.1 两个核心抽象

FlexAttention 提供了两个核心抽象：

**1. `score_mod`**：修改注意力分数的函数

```python
def score_mod(score, batch, head, q_idx, kv_idx):
    # score: 当前 (q_idx, kv_idx) 位置的注意力分数（标量）
    # 返回：修改后的分数
    return modified_score
```

**2. `mask_mod`**（通过 `BlockMask`）：指定哪些位置需要计算

```python
def mask_mod(batch, head, q_idx, kv_idx):
    # 返回 True 表示这个位置需要计算注意力
    # 返回 False 表示跳过（不计算）
    return should_attend
```

### 2.2 分工设计

```
score_mod → "怎么算"  → 控制分数的值（缩放、偏置、截断等）
mask_mod  → "算哪些"  → 控制计算范围（因果、滑动窗口、稀疏等）
```

这两者可以独立使用，也可以组合使用。组合时，mask_mod 先过滤掉不需要计算的位置，score_mod 再修改需要计算的位置的分数。

### 2.3 与传统实现的对比

**传统方式（手写 kernel）**：
```
每个注意力变体 → 手写 CUDA/Triton kernel → 数百行底层代码
```

**FlexAttention 方式**：
```
每个注意力变体 → 写一个 5 行的 Python 函数 → 框架自动编译
```

**代价**：自动编译生成的 kernel 通常不如手写优化 kernel 快。FlexAttention 用**灵活性**换取了**性能**。

---

## 3. 核心 API 详解

### 3.1 `flex_attention()` 函数签名

```python
def flex_attention(
    query,          # [B, Hq, L, E]  Query 张量
    key,            # [B, Hkv, S, E] Key 张量
    value,          # [B, Hkv, S, Ev] Value 张量
    score_mod=None, # 分数修改函数
    block_mask=None,# BlockMask 对象
    scale=None,     # 缩放因子（默认 1/sqrt(E)）
    enable_gqa=False,# 是否启用 GQA
    return_lse=False,# 是否返回 logsumexp
    kernel_options=None, # Triton kernel 选项
)
```

### 3.2 `score_mod` 签名

```python
def score_mod(score, batch, head, q_idx, kv_idx):
    # score: 标量张量，当前位置的 QK^T / sqrt(d)
    # batch, head, q_idx, kv_idx: 标量索引（torch.int 类型的 0 维张量）
    # 返回：修改后的标量分数
    return modified_score
```

**关键约束**：这些参数都是**标量**（0 维张量）。`score_mod` 在概念上对注意力矩阵的**每个元素**独立调用。但实际上它会被编译成向量化的 Triton kernel。

### 3.3 `mask_mod` 和 `create_block_mask()`

```python
def mask_mod(batch, head, q_idx, kv_idx):
    # 返回 bool：True 表示允许关注，False 表示屏蔽
    return should_attend

block_mask = create_block_mask(
    mask_mod,
    B, H, Q_LEN, KV_LEN,  # 形状参数
    device="cuda",
    BLOCK_SIZE=128,         # 块大小（PyTorch 2.6.0 中已支持自定义）
)
```

### 3.4 `BlockMask` 对象

`BlockMask` 是一个**块级别**的稀疏掩码，包含以下内部数据：

```python
class BlockMask:
    kv_num_blocks     # 每个查询块对应的部分 KV 块数量
    kv_indices        # 每个查询块对应的 KV 块索引
    full_kv_num_blocks # 完全填充的 KV 块数量
    full_kv_indices    # 完全填充的 KV 块索引
    q_num_blocks      # 查询块的数量
    q_indices         # 查询块的索引
    BLOCK_SIZE        # (Q_BLOCK_SIZE, KV_BLOCK_SIZE) 元组
    mask_mod          # 原始 mask_mod 函数引用
```

**重要概念**：BlockMask 不存储每个元素的掩码（太大了），而是将序列分成固定大小的**块**（block），记录每个查询块需要关注哪些 KV 块。

---

## 4. 内部执行流程：从 Python 到 Triton

### 4.1 完整的执行路径

```
用户调用 flex_attention(q, k, v, score_mod, block_mask)
    │
    ├── 1. 输入验证（维度、dtype 检查）
    │
    ├── 2. 设置默认值
    │   ├── score_mod = identity (如果为 None)
    │   ├── block_mask = empty_mask (如果为 None)
    │   └── scale = 1/sqrt(d) (如果为 None)
    │
    ├── 3. 配置 kernel_options
    │   ├── ROWS_GUARANTEED_SAFE = False
    │   ├── PRESCALE_QK = False
    │   └── OUTPUT_LOGSUMEXP = (需要梯度时 True)
    │
    ├── 4. torch.compile 编译
    │   ├── mark_static(q, -3)  # 头数为静态
    │   ├── mark_static(q, -1)  # 头维度为静态
    │   └── flex_attention_hop(...) # 核心 HOP (Higher-Order Operator)
    │
    ├── 5. HOP 内部：Triton kernel 生成
    │   ├── 将 score_mod 通过 make_fx 转为 FX 图
    │   ├── 将 FX 图内联到 Triton 模板中
    │   ├── 配置 kernel 参数 (BLOCK_M, BLOCK_N, etc.)
    │   └── 调用 Triton JIT 编译器
    │
    └── 6. 执行编译好的 Triton kernel
        ├── Grid: (ceil(S/BLOCK_M), B*H, 1)
        ├── 每个 thread block 处理 BLOCK_M 个 query
        ├── 遍历 KV blocks（由 BlockMask 决定）
        └── 在线 softmax + 累积输出
```

### 4.2 Triton Kernel 的核心结构

FlexAttention 的 Triton kernel 使用**在线 softmax（online softmax）**算法，也称为 Flash Attention 的核心算法：

```python
# 简化的 kernel 伪代码
for each (batch, head) pair:           # 并行维度 2
    for each query_block (q_start):     # 并行维度 1
        m_i = -inf                      # 当前行最大值
        l_i = 0                         # 当前行 softmax 分母
        acc = 0                         # 当前行累积输出

        load Q_block = Q[q_start:q_start+BLOCK_M, :]

        for each kv_block:              # 由 BlockMask 决定迭代顺序
            load K_block, V_block

            # 计算 score
            score = Q_block @ K_block^T  # [BLOCK_M, BLOCK_N]

            # 应用 score_mod（编译内联的 Python 函数）
            score = compiled_score_mod(score, ...)
            # 应用 BlockMask（跳过全零块）

            # 在线 softmax 更新
            m_new = max(m_i, max(score))
            correction = exp(m_i - m_new)
            l_new = l_i * correction + sum(exp(score - m_new))
            acc_new = acc * correction + exp(score - m_new) @ V_block

            m_i, l_i, acc = m_new, l_new, acc_new

        # 最终归一化
        output[q_start:q_start+BLOCK_M, :] = acc / l_i
```

### 4.3 Grid 并行策略

```
Grid = (ceil(Q_LEN / BLOCK_M), B * H, 1)

           Q Block 0  Q Block 1  Q Block 2  Q Block 3
Batch 0, Head 0:  [kernel]    [kernel]    [kernel]    [kernel]
Batch 0, Head 1:  [kernel]    [kernel]    [kernel]    [kernel]
Batch 0, Head 2:  [kernel]    [kernel]    [kernel]    [kernel]
...
Batch 1, Head 0:  [kernel]    [kernel]    [kernel]    [kernel]
```

每个 kernel 实例独立处理一个 (batch, head) 组合中的一个 query 块。这意味着：
- B×H 个组合之间完全并行
- 同一组合内的不同 query 块之间也并行
- 同一 query 块内的 KV 块必须顺序处理（在线 softmax 的依赖）

---

## 5. 实验一：BlockMask 内部结构解剖

![BlockMask Anatomy](figures/flex_fig1_blockmask.png)

### 5.1 PyTorch 2.6.0 的 BLOCK_SIZE 改进

**重大改进：`BLOCK_SIZE` 参数现在被正确接受。**

```
Requested BLOCK_SIZE=16,  Actual=(16, 16),  Sparsity=46.9%
Requested BLOCK_SIZE=32,  Actual=(32, 32),  Sparsity=43.8%
Requested BLOCK_SIZE=64,  Actual=(64, 64),  Sparsity=37.5%
Requested BLOCK_SIZE=128, Actual=(128, 128), Sparsity=25.0%
```

**与 PyTorch 2.5.1 的对比**：在 2.5.1 中，`BLOCK_SIZE` 参数被完全忽略，所有值都被强制上取整到 128。在 2.6.0 中，请求的 `BLOCK_SIZE` 被正确使用。这是一个重要的改进，因为：

1. **更细粒度的稀疏性**：较小的 BLOCK_SIZE（如 16 或 32）可以在块级别捕捉更精细的稀疏模式
2. **短序列受益**：S=256 时用 BLOCK_SIZE=16 可以有 16 个块，稀疏率从 25% 提升到 46.9%
3. **灵活的精度-性能权衡**：用户可以根据实际稀疏模式选择最优的块大小

**注意**：默认行为（不指定 `BLOCK_SIZE` 时）仍然是 128。这意味着 `create_block_mask(causal_mask, 1, 1, 16, 16)` 仍会使用 128 的块大小，导致 S=16 整个序列变成一个块（1×1），稀疏率为 0%。

### 5.2 BlockMask 的内部表示

以 S=16、Causal mask、默认 BLOCK_SIZE=128 为例：

```
dense mask (1x1 block, 因为 16 < 128):
  [[1]]

kv_num_blocks: [1]      ← 1 个 KV 块需要计算
kv_indices: [[0]]       ← 这个块的索引是 0
full_kv_num_blocks: [0] ← 0 个完全填充的块
sparsity: 0.0%          ← 没有任何稀疏性（整个序列一个块）
```

对于 S=256 的 Causal mask（使用默认 BLOCK_SIZE=128，2 个块）：

```
Block 0 (q=0-127):   可以关注 kv_block 0
Block 1 (q=128-255): 可以关注 kv_block 0 和 kv_block 1

理论上 sparsity = 25%（2x2 块矩阵中，只有 3/4 的块需要计算）
实际测量 sparsity = 25%
```

### 5.3 BLOCK_SIZE 对稀疏率的影响

对于 Causal mask（S=256），不同 BLOCK_SIZE 下的块级稀疏率：

| BLOCK_SIZE | 块数量 (S/BS)² | 非零块数 | 稀疏率 | 含义 |
|-----------|--------------|---------|--------|------|
| 16 | 256 | 136 | 46.9% | 精细粒度，接近理论 50% |
| 32 | 64 | 36 | 43.8% | |
| 64 | 16 | 10 | 37.5% | |
| 128 | 4 | 3 | 25.0% | 粗糙，丢失大量稀疏信息 |

**关键洞察**：BLOCK_SIZE 越小，块级稀疏率越接近像素级真实稀疏率（Causal 的 50%）。BLOCK_SIZE=128 时只有 25%，一半的潜在稀疏性被粗粒度块"浪费"了。

### 5.4 BLOCK_SIZE 对性能的实际影响

虽然更小的 BLOCK_SIZE 提供了更高的稀疏率，但并不一定带来更好的性能：

- **更小的块** → 更多块数量 → 更多 kernel 循环迭代 → 更高的循环开销
- **更大的块** → 更少的循环迭代 → 但可能包含不必要的计算
- **最优 BLOCK_SIZE** 取决于具体的稀疏模式和序列长度

在实际使用中，BLOCK_SIZE=128 仍然是一个合理的默认值，因为它在块管理开销和计算效率之间取得了平衡。

---

## 6. 实验二：score_mod 编译追踪

![Score Mod](figures/flex_fig2_score_mod.png)

### 6.1 不同 score_mod 的性能

| score_mod 类型 | 延迟 (ms) | 说明 |
|---------------|----------|------|
| Identity | 2.90 | 不修改分数（仅基准开销） |
| Causal | 2.90 | `where(q_idx >= kv_idx, score, -inf)` |
| Tanh Softcap(50) | 2.98 | `50 * tanh(score / 50)` |
| RelPosBias | 2.96 | `score + 0.5 * (q_idx - kv_idx)` |
| ALiBi | 3.27 | `score - slope * (q_idx - kv_idx)` |

配置：B=1, H=1, S=64, D=32

### 6.2 分析

1. **所有 score_mod 的延迟非常接近**（2.90-3.27ms），差异仅 ~13%
2. Identity 和 Causal 几乎没有差异，说明 `torch.where` 被高效编译
3. RelPosBias 和 ALiBi 稍慢，因为需要额外的浮点运算（减法 + 乘法）
4. **这些延迟主要是编译后的 kernel 启动开销**，而非实际计算

### 6.3 编译过程追踪

当 FlexAttention 第一次遇到一个 `score_mod` 函数时：

```
Step 1: make_fx(score_mod) → 将 Python 函数转为 FX 图（IR）
Step 2: FX 图 → 内联到 Triton 模板中
Step 3: Triton JIT 编译器 → 生成 PTX/SASS 代码
Step 4: 缓存编译结果
```

后续调用相同签名（相同的计算图结构）时，直接使用缓存的 kernel。

---

## 7. 实验三：稀疏性 vs 性能

![Sparsity vs Performance](figures/flex_fig3_sparsity_perf.png)

### 7.1 实验结果

| S | SDPA | FA dense | FA causal | FA sw64 | FA prefix |
|---|------|---------|----------|---------|----------|
| 256 | 0.04ms | 3.09ms (75x) | 2.43ms (59x) | 2.47ms (60x) | 2.63ms (64x) |
| 512 | 0.05ms | 2.98ms (57x) | 2.42ms (46x) | 2.51ms (48x) | 2.50ms (48x) |
| 1024 | 0.10ms | 7.54ms (76x) | 6.70ms (68x) | 6.75ms (68x) | 6.71ms (68x) |
| 2048 | 0.26ms | 25.93ms (100x) | 24.54ms (95x) | 24.78ms (95x) | 24.59ms (95x) |

### 7.2 关键发现

1. **FlexAttention 比 SDPA 慢 57-100 倍**，在 L4 上开销比 3090 更显著
2. **稀疏性几乎不影响性能**：causal (25-47% sparse) 只比 dense 快 ~5-10%
3. **SDPA 在 S=256 时仅需 0.04ms**，而 FlexAttention 需要 ~3ms
4. S=1024 到 S=2048 出现了延迟跳变（7ms → 25ms），说明 L4 的计算能力在这个区间成为瓶颈

### 7.3 为什么稀疏性没有显著帮助？

理论上，causal mask 应该跳过约 50% 的计算。但实际中：

1. **Block 级别稀疏粒度**：默认 BLOCK_SIZE=128，S=256 时只有 2 个块，Causal 只能跳过 1/4
2. **Kernel 启动开销是主要瓶颈**：在短序列时，kernel 的固定开销远大于实际计算时间
3. **L4 GPU 的 SM 数量较少（60 个）**，对 Triton kernel 的并行度发挥有影响

---

## 8. 实验四：mask_mod + score_mod 组合

### 8.1 实验结果

| 配置 | 延迟 (ms) | 稀疏性 | vs SDPA |
|------|----------|--------|---------|
| 无修改 | 3.05 | 0% | 81.4x |
| Causal mask | 2.49 | 25% | 66.9x |
| Causal + Softcap | 2.68 | 25% | 69.5x |
| Causal + ALiBi | 2.88 | 25% | 73.6x |
| Sliding window | 2.54 | 0% | 67.1x |
| SW + Softcap | 2.71 | 0% | 70.5x |
| Prefix | 2.54 | 25% | 67.3x |

配置：B=2, H=4, S=256, D=64

### 8.2 分析

1. **添加 score_mod 只增加 ~7-16% 的延迟**：组合使用几乎免费
2. **BlockMask 提供了约 18% 的加速**（相比无 mask）
3. **所有 FlexAttention 配置都比 SDPA 慢 67-81 倍**
4. FlexAttention 的价值不在于速度，而在于**用 5 行代码实现任意注意力模式**

---

## 9. 实验五：torch.compile 编译开销

![Compile Overhead](figures/flex_fig4_compile.png)

### 9.1 实验结果

| 配置 | S | 首次调用 | 缓存后 | 编译开销 | SDPA |
|------|---|---------|--------|---------|------|
| no_mod | 64 | **198.3ms** | 2.96ms | 195.4ms | 0.04ms |
| no_mod | 256 | 4.2ms | 2.96ms | 1.2ms | 0.04ms |
| no_mod | 1024 | 4.9ms | 2.98ms | 1.9ms | 0.06ms |
| causal_score | 64 | **194.6ms** | 3.09ms | 191.5ms | 0.04ms |
| causal_score | 256 | 4.4ms | 3.07ms | 1.3ms | 0.04ms |
| causal_score | 1024 | 4.7ms | 3.10ms | 1.6ms | 0.06ms |
| causal_block | 64 | **217.2ms** | 2.29ms | 214.9ms | 0.04ms |
| causal_block | 256 | **246.0ms** | 2.47ms | 243.5ms | 0.04ms |
| causal_block | 1024 | 3.1ms | 2.42ms | 0.7ms | 0.06ms |

### 9.2 关键发现

1. **首次编译开销高达 195-246ms**！比 PyTorch 2.5.1 的 130-180ms 还高
2. BlockMask 首次编译最慢（246ms），因为需要编译额外的稀疏 kernel
3. **第二次调用同一配置时立即变快**（缓存后仅 2.3-3.1ms）
4. 值得注意的是，一旦某个 S 的 kernel 编译过，相近的 S 值可能复用缓存（如 no_mod S=64 → S=256 只需 4.2ms）

### 9.3 实际影响

在训练或长推理任务中，编译开销只发生一次，会被后续大量调用均摊。但在短时推理或交互式场景中，200ms 的首次延迟是不可接受的。

---

## 10. 实验六：FlexAttention vs SDPA 延迟对比

![Latency Showdown](figures/flex_fig5_showdown.png)

### 10.1 完整对比数据

| S | SDPA dense | SDPA causal | FA dense | FA causal(block) | FA causal(score) | FA sw64 | FA GQA |
|---|-----------|------------|---------|-----------------|-----------------|---------|--------|
| 64 | 0.04 | 0.04 | 3.07 | 2.42 | 3.13 | 2.46 | 2.98 |
| 128 | 0.04 | 0.05 | 3.03 | 2.42 | 3.15 | 2.46 | 3.01 |
| 256 | 0.04 | 0.04 | 3.03 | 2.59 | 3.13 | 2.62 | 2.97 |
| 512 | 0.06 | 0.06 | 3.19 | 2.53 | 3.18 | 2.54 | 2.97 |
| 1024 | 0.10 | 0.10 | 7.43 | 6.94 | 8.00 | 6.88 | 7.43 |
| 2048 | 0.26 | 0.18 | 25.98 | 24.77 | 27.73 | 24.74 | 25.85 |

### 10.2 分析

1. **SDPA 始终比 FlexAttention 快 30-100 倍**
2. FlexAttention 有一个 ~3ms 的**固定开销**（即使 S=64 也需要 3.07ms）
3. SDPA 在 S ≤ 512 时几乎"零延迟"（0.04-0.06ms）
4. S ≥ 1024 时出现延迟跳变：FlexAttention 从 ~3ms 跳到 7-8ms，S=2048 更是达到 25ms
5. BlockMask 方式的 causal 比 score_mod 方式快约 23%（2.42 vs 3.13ms at S=64）

### 10.3 固定开销的来源

FlexAttention 的 ~3ms 固定开销来自：

1. **Triton kernel 启动开销**（~0.1-0.5ms）
2. **BlockMask 的间接寻址**（加载 kv_num_blocks, kv_indices 等）
3. **在线 softmax 的循环控制**（即使只有一个 KV 块也要走循环）
4. **score_mod 的间接调用**（通过编译后的 FX 图）

相比之下，SDPA 直接调用 cuDNN 的 FlashAttention kernel，没有这些额外开销。

---

## 11. 实验七：逐步计算追踪（完整例子）

### 11.1 输入数据（S=4, D=4）

```
Q = [[ 1.93,  1.49,  0.90, -2.11],
     [ 0.68, -1.23, -0.04, -1.60],
     [-0.75,  1.65, -0.39, -1.40],
     [-0.73, -0.56, -0.77,  0.76]]

K = [[ 1.64, -0.16, -0.50,  0.44],
     [-0.76,  1.08,  0.80,  1.68],
     [ 1.28,  1.30,  0.61,  1.33],
     [-0.23,  0.04, -0.25,  0.86]]

V = [[-1.38, -0.87, -0.22,  1.72],
     [ 0.32, -0.42,  0.31, -0.77],
     [-1.56,  1.00, -0.88, -0.60],
     [-1.27,  2.12, -1.23, -0.49]]
```

### 11.2 Step 1: 计算 QK^T

```
scale = 1/sqrt(4) = 0.5
raw_scores = Q @ K^T × 0.5

       kv=0    kv=1    kv=2    kv=3
q=0 [  0.777, -1.337,  1.066, -1.211]
q=1 [  0.314, -2.288, -1.450, -0.789]
q=2 [ -0.960, -0.163, -0.469, -0.433]
q=3 [ -0.194,  0.307, -0.554,  0.497]
```

**解释**：`scores[0][0] = 0.777` 表示第 0 个 query 和第 0 个 key 的相似度（缩放后）。

### 11.3 Step 2: 应用 Causal Mask

```
Causal mask:
[ 1, 0, 0, 0]    ← q=0 只能看 kv=0
[ 1, 1, 0, 0]    ← q=1 能看 kv=0,1
[ 1, 1, 1, 0]    ← q=2 能看 kv=0,1,2
[ 1, 1, 1, 1]    ← q=3 能看 kv=0,1,2,3

Masked scores:
[  0.777,   -inf,   -inf,   -inf]
[  0.314, -2.288,   -inf,   -inf]
[ -0.960, -0.163, -0.469,   -inf]
[ -0.194,  0.307, -0.554,  0.497]
```

**解释**：`-inf` 位置的分数经过 softmax 后会变成 0，相当于"不看"那些位置。

### 11.4 Step 3: Softmax

```
Attention weights:
[1.000, 0.000, 0.000, 0.000]    ← q=0 只看 kv=0（100%）
[0.931, 0.069, 0.000, 0.000]    ← q=1 主要看 kv=0（93%），少量看 kv=1（7%）
[0.206, 0.457, 0.337, 0.000]    ← q=2 均匀分布
[0.187, 0.309, 0.131, 0.374]    ← q=3 分布最均匀
```

### 11.5 Step 4: 加权求和

```
Output = Attn_weights @ V

[-1.385, -0.871, -0.223,  1.717]    ← 等于 V[0]（因为 q=0 只看 kv=0）
[-1.267, -0.840, -0.187,  1.545]    ← 93% V[0] + 7% V[1]
[-0.664, -0.038, -0.202, -0.203]
[-0.840,  0.629, -0.523, -0.179]
```

### 11.6 验证：FlexAttention 输出

```
FlexAttention (FP16):
[-1.385, -0.871, -0.223,  1.718]    ← 与手动计算几乎完全一致
[-1.268, -0.840, -0.187,  1.546]
[-0.664, -0.039, -0.203, -0.203]
[-0.840,  0.629, -0.523, -0.178]

SDPA (FP16):
[-1.385, -0.871, -0.223,  1.718]
[-1.268, -0.840, -0.187,  1.546]
[-0.664, -0.038, -0.203, -0.203]
[-0.840,  0.629, -0.523, -0.179]
```

三者（手动计算、FlexAttention、SDPA）在 FP16 精度下完全一致。

### 11.7 score_mod 例子：Tanh Softcap

```
Softcap(2.0) 修改后的分数：
Before: [ 0.777, -1.337,  1.066, -1.211]
After:  [ 0.740, -1.168,  0.976, -1.082]

效果：大分数被"压扁"到 [-2, 2] 范围内
  0.777 → 0.740（减少了 4.8%）
  1.066 → 0.976（减少了 8.4%）

最终输出的差异：
Causal only:    [-1.385, -0.871, -0.223,  1.717]  (q=0)
Causal+Softcap: [-1.385, -0.871, -0.223,  1.717]  (q=0, 只看一个 token 无差异)

q=1: Causal only    → [-1.267, -0.840, -0.187,  1.545]
     Causal+Softcap → [-1.171, -0.815, -0.157,  1.405]  (显著不同)
```

---

## 12. 实验八：不同注意力模式性能剖析

![Pattern Analysis](figures/flex_fig6_patterns.png)

### 12.1 实验结果（S=1024, B=2, H=8, D=64）

| 模式 | 延迟 (ms) | 稀疏性 | vs SDPA |
|------|----------|--------|---------|
| Dense (无修改) | 7.43 | 0% | 75.4x |
| Causal | 6.84 | 44% | 69.5x |
| Causal + Softcap | 8.56 | 44% | 86.9x |
| Sliding window 64 | 6.92 | 33% | 70.2x |
| Sliding window 256 | 7.00 | 23% | 71.0x |
| Prefix(32) + Causal | 6.85 | 44% | 69.5x |
| Prefix(128) + Causal | 6.85 | 44% | 69.5x |
| Dilated(4) | 7.11 | 0% | 72.1x |
| **SDPA dense** | **0.10** | **0%** | **1.0x** |

### 12.2 关键发现

1. **所有模式的延迟在 6.84-8.56ms 范围内**，差异仅 ~25%
2. **44% 的稀疏性只带来 8% 的加速**（7.43ms → 6.84ms）
3. **Dilated 模式虽然稀疏但有 0% 的块级稀疏性**——因为每个块内都有需要计算的元素
4. **Softcap 增加 ~25% 的开销**（6.84ms → 8.56ms），比 PyTorch 2.5.1 下的 17% 更多
5. **滑动窗口大小 (64 vs 256) 对性能几乎没有影响**

---

## 13. 性能分析：为什么 FlexAttention 慢

### 13.1 原因一：torch.compile 的间接开销

FlexAttention 通过 `torch.compile(backend="eager")` 编译。编译过程引入了多层间接调用：

```
Python score_mod → make_fx → FX Graph → Triton Template → Triton IR → PTX → SASS
```

每一层转换都会引入一些开销。特别是：
- FX 图需要为每个操作创建 IR 节点
- Triton 模板中的子图内联需要额外的缓冲区管理
- 生成的 kernel 不如手写 kernel 紧凑

### 13.2 原因二：通用性的代价

FlexAttention 的 kernel 必须支持**任意** score_mod 函数。这意味着：
- 不能像 FlashAttention 那样针对特定模式（causal）做特化优化
- 必须为 score_mod 的执行保留寄存器空间
- 循环结构必须最通用化

### 13.3 原因三：Triton kernel 的效率限制

Triton 编程模型虽然简化了 GPU 编程，但也引入了限制：
- 不能手动管理共享内存（由 Triton 自动管理）
- 不能使用 tensor cores 的 MMA（矩阵乘累加）指令
- 自动内存合并（memory coalescing）可能不是最优的

### 13.4 原因四：BlockMask 的粗粒度稀疏

默认 BLOCK_SIZE=128 意味着：
- S=256 时只有 2 个块，Causal 只能跳过 1/4
- S=1024 时有 8 个块，Causal 能跳过 ~44%
- 真正的细粒度稀疏（element-wise）无法被利用

**PyTorch 2.6.0 改进**：现在 BLOCK_SIZE 可以自定义了。设置较小的 BLOCK_SIZE（如 16 或 32）可以显著提升稀疏率，但需要权衡块管理开销。

### 13.5 原因五：固定开销占比大

在短序列下，FlexAttention 的 ~3ms 固定开销使得性能看起来特别差：

```
S=64:   SDPA=0.04ms, Flex=3.07ms → 79x overhead
S=2048: SDPA=0.26ms, Flex=25.98ms → 100x overhead
```

与 3090 的结果对比，L4 上 S=2048 的 overhead 更高（100x vs 26x），这主要是因为 L4 的绝对计算性能较弱，而 FlexAttention 的 Triton kernel 开销与硬件算力无关。

---

## 14. 性能分析：FlexAttention 什么时候快

### 14.1 FlexAttention 不适合的场景

1. **标准注意力模式**（dense, causal）：直接用 SDPA，快 60-100 倍
2. **短序列推理**（S < 1024）：固定开销占比太大
3. **低延迟要求的应用**：首次编译需要 200ms+
4. **只需要一种固定模式**：手写或专用 kernel 更快

### 14.2 FlexAttention 适合的场景

1. **研究原型**：快速实验新的注意力模式，无需写 kernel
2. **复杂组合模式**：causal + softcap + sliding window 等，手写 kernel 很难
3. **长序列训练**：编译开销被均摊，~3ms 的延迟在训练中可接受
4. **非标准稀疏模式**：prefix、dilated、自定义 mask，没有现成的 SDPA 实现

### 14.3 BlockMask 方式 vs score_mod 方式

对于 Causal mask：
- `score_mod` 方式：3.13ms（仍然计算所有位置的分数，只是把 mask 掉的设为 -inf）
- `block_mask` 方式：2.42ms（直接跳过整个块的 KV 加载和计算）

**结论**：能用 `block_mask`（mask_mod）就尽量用，比用 `score_mod` 做 masking 更高效。

---

## 15. PyTorch 2.6.0 的已知限制与改进

### 15.1 BLOCK_SIZE 已支持自定义（改进！）

PyTorch 2.6.0 修复了 2.5.1 中 `BLOCK_SIZE` 被强制忽略的问题。现在 `create_block_mask(..., BLOCK_SIZE=16)` 会实际使用 16 作为块大小。但仍需注意：

- 默认值仍然是 128，需要显式指定才能使用更小的块
- 更小的 BLOCK_SIZE 提供更高的稀疏率，但可能增加循环迭代开销

### 15.2 首次编译开销更大

PyTorch 2.6.0 的首次编译开销（195-246ms）比 2.5.1（130-180ms）更高。这可能是因为：
- Triton 编译器的优化 pass 更多
- PyTorch 2.6 的编译路径有所变化
- L4 GPU 的编译时间可能比 3090 更长

### 15.3 score_mod 不支持动态张量索引

在 `score_mod` 中使用 `q_idx` 或 `kv_idx` 作为张量的索引（如 `k_pe[b, kv_idx, :]`）仍会触发 `DataDependentOutputException`。

### 15.4 编译缓存行为

- 首次调用需要 195-246ms 编译时间
- 相同签名（相同计算图）的后续调用使用缓存
- 不同序列长度可能复用已编译的 kernel（取决于是否超出编译范围）

---

## 16. 结论

### 16.1 核心发现

| 发现 | 数据支持 |
|------|---------|
| FlexAttention 比 SDPA 慢 57-100 倍 | Exp6: S=256 时 75x, S=2048 时 100x |
| 首次编译开销 ~195-246ms | Exp5: 首次 198ms, 缓存后 3.0ms |
| BLOCK_SIZE 已支持自定义 | Exp1: 请求的值被正确使用（2.5.1 会被忽略） |
| 稀疏性带来的加速有限 | Exp8: 44% 稀疏只快 8% |
| score_mod 几乎免费组合 | Exp4: 组合只增加 7-16% 延迟 |
| BlockMask 比 score_mod 做掩码更高效 | Exp6: 2.42ms vs 3.13ms |

### 16.2 FlexAttention 的真正价值

FlexAttention 的价值**不在于性能**，而在于：

1. **开发效率**：用 5 行 Python 代码替代数百行 CUDA kernel
2. **可组合性**：score_mod 和 mask_mod 可以任意组合
3. **可实验性**：研究者可以快速尝试新的注意力模式
4. **正确性**：框架保证编译后的行为与 Python 描述一致

### 16.3 推荐使用策略

```
如果只需标准模式（dense/causal）       → 用 SDPA
如果需要标准模式 + 位置偏置            → 用 SDPA + 手写偏置
如果需要非标准稀疏模式                 → 用 FlexAttention (mask_mod)
如果需要复杂的分数修改                 → 用 FlexAttention (score_mod)
如果需要组合多种修改                   → 用 FlexAttention (mask + score)
如果需要生产级性能                     → 手写 Triton/CUDA kernel
```

### 16.4 图表索引

| 图表 | 文件 | 描述 |
|------|------|------|
| 图1 | `flex_fig1_blockmask.png` | BlockMask 内部结构解剖 |
| 图2 | `flex_fig2_score_mod.png` | score_mod 编译延迟对比 |
| 图3 | `flex_fig3_sparsity_perf.png` | 稀疏性 vs 性能 |
| 图4 | `flex_fig4_compile.png` | torch.compile 编译开销 |
| 图5 | `flex_fig5_showdown.png` | FlexAttention vs SDPA 延迟对比 |
| 图6 | `flex_fig6_patterns.png` | 不同注意力模式性能剖析 |
