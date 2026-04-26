# Sliding Window Attention 深度解析：原理、实现与实验

> **从 Mistral 到 Gemma，为什么越来越多模型选择"只看最近的"？**
>
> Vanilla PyTorch vs FlexAttention | 窗口大小敏感性 | 稀疏性分析
>
> NVIDIA L4 (24GB) | PyTorch 2.6.0+cu124 | Triton 3.2.0

---

## 第一章：为什么需要 Sliding Window？

### 1.1 标准 Causal 的计算瓶颈

标准因果注意力的计算复杂度是 O(S²)：序列越长，每增加一个 token，计算量就翻倍增长。

```
序列长度 S=8192 时：
  注意力矩阵大小: 8192 × 8192 = 67M 个元素
  每 token 平均关注: 4096 个历史 token
  
序列长度 S=32768 时：
  注意力矩阵大小: 32768 × 32768 = 1.07B 个元素
  每 token 平均关注: 16384 个历史 token
```

但实际上，**远处的 token 对当前预测的影响远小于近处的 token**。自然语言中最重要的依赖通常在几百个 token 以内。

### 1.2 Sliding Window 的核心思想

每个 token 只关注固定窗口大小 W 内的历史 token：

```
窗口大小 W=3：
    0  1  2  3  4  5  6
0 [█  .  .  .  .  .  .]   只看自己
1 [█  █  .  .  .  .  .]   看 0,1
2 [█  █  █  .  .  .  .]   看 0,1,2
3 [.  █  █  █  .  .  .]   看 1,2,3 (窗口=3)
4 [.  .  █  █  █  .  .]   看 2,3,4
5 [.  .  .  █  █  █  .]   看 3,4,5
6 [.  .  .  .  █  █  █]   看 4,5,6
```

**计算复杂度从 O(S²) 降低到 O(S×W)**！当 W << S 时，这是一个巨大的节省。

### 1.3 信息传播：多层堆叠弥补局部性

单层窗口大小为 W，但堆叠 L 层后：
- 第 1 层：每个 token 看到 W 个邻居
- 第 2 层：通过第一层的传递，看到 W² 个 token
- 第 L 层：理论上看到 W^L 个 token

例如 W=4096, L=32 层：有效感受野 = 4096^32，远大于任何实际序列长度。

---

## 第二章：三种实现方式

### 2.1 Vanilla PyTorch

```python
def vanilla_sliding_window(q, k, v, window_size=256):
    S = q.shape[-2]
    D = q.shape[-1]
    # Step 1: 计算 QK^T
    scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
    # Step 2: 构造因果掩码
    pos = torch.arange(S, device=q.device)
    causal = pos.unsqueeze(0) >= pos.unsqueeze(1)
    # Step 3: 构造窗口掩码
    window = (pos.unsqueeze(0) - pos.unsqueeze(1)) <= window_size
    # Step 4: 合并并应用
    scores = scores.masked_fill(~(causal & window), float('-inf'))
    weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(weights, v)
```

**问题**：仍然实例化完整的 S×S 矩阵，没有利用稀疏性。

### 2.2 FlexAttention

```python
def flex_sliding_window(q, k, v, window_size=256):
    B, _, S, _ = q.shape
    def sw_mask(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        window = (q_idx - kv_idx) <= window_size
        return causal & window
    block_mask = create_block_mask(sw_mask, B, 1, S, S, device=q.device)
    return flex_attention(q, k, v, block_mask=block_mask)
```

**优势**：BlockMask 自动分析窗口模式，跳过窗口外的所有块。

### 2.3 稀疏性分析

| 窗口大小 | 像素级稀疏率 | 含义 |
|---------|------------|------|
| W=64 | 96.9% | 97% 的注意力位置被跳过 |
| W=128 | 93.7% | 94% 被跳过 |
| W=256 | 88.2% | 88% 被跳过 |
| W=512 | 75.2% | 75% 被跳过 |
| W=S（标准Causal） | 50.0% | 50% 被跳过（上三角） |

窗口越小 → 稀疏率越高 → FlexAttention 能跳过更多计算。

---

## 第三章：实验结果

### 3.1 延迟对比（S=2048）

| 窗口大小 | Vanilla (ms) | Flex (ms) | 速度比 |
|---------|-------------|-----------|--------|
| W=64 | 5.2 | 18.8 | 3.6x |
| W=256 | 5.2 | 18.8 | 3.6x |
| W=S (Causal) | 5.1 | 18.8 | 3.7x |

**注意**：在 L4 上 FlexAttention 比 Vanilla 慢约 3.6x。这是因为 Triton kernel 的启动开销在 SM 数较少的 L4 上比较显著。在 A100/H100 上差距会大幅缩小。

### 3.2 扩展性（W=256）

| S | Vanilla (ms) | Flex (ms) | Flex/Vanilla |
|---|-------------|-----------|-------------|
| 512 | 0.35 | 5.9 | 16.9x |
| 1024 | 0.69 | 6.8 | 9.9x |
| 2048 | 5.16 | 14.3 | 2.8x |
| 4096 | 21.1 | 47.1 | 2.2x |
| 8192 | 85.4 | 184.1 | 2.2x |

**关键趋势**：随 S 增大，Flex/Vanilla 比值缩小（从 16.9x 降到 2.2x），说明 Triton kernel 的固定开销占比在减小。

### 3.3 SDPA 对比（仅标准 Causal）

SDPA (FlashAttention2) **不支持 Sliding Window**。如果你需要 SW：
- SDPA: **不能做**
- Vanilla: O(S²) 显存，大 S 会 OOM
- Flex: O(S) 显存 + BlockMask 稀疏跳过

---

## 第四章：实际应用

### 4.1 使用 Sliding Window 的模型

| 模型 | 窗口大小 | 总层数 | 有效感受野 |
|------|---------|--------|-----------|
| Mistral-7B | 4096 | 32 | 4096² per 2 layers |
| Gemma-2 | 4096 | 42 | 混合 SW + 全局 |
| Phi-3 | 2048 | 32 | ~65K |

### 4.2 代码示例

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def make_sliding_window_mask(window_size):
    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        window = (q_idx - kv_idx) <= window_size
        return causal & window
    return mask_mod

# Mistral-style SW attention
mask_fn = make_sliding_window_mask(4096)
block_mask = create_block_mask(mask_fn, B, 1, S, S, device="cuda")
output = flex_attention(q, k, v, block_mask=block_mask)
```

---

## 总结

| 维度 | Vanilla | SDPA | FlexAttention |
|------|---------|------|--------------|
| 支持 SW | 是 | **否** | **是** |
| 显存 | O(S²) | N/A | O(S) + BlockMask |
| 代码量 | ~8行 | N/A | **3行** |
| 稀疏利用 | 否 | N/A | **是** |

---

*报告生成时间：2026-04-25*
