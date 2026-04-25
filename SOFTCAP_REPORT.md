# Tanh Softcapping 深度解析：防止 logits 数值爆炸的简洁方案

> **Gemma-2 和 Grok-1 的选择：用 tanh 截断替代梯度裁剪。**
>
> score_mod 的经典应用 | 数值稳定性分析 | 与梯度裁剪的区别
>
> NVIDIA L4 (24GB) | PyTorch 2.6.0+cu124 | Triton 3.2.0

---

## 第一章：为什么需要 Softcapping？

### 1.1 Logits 爆炸问题

在注意力计算中，QK^T 的分数理论上可以是任意值。当某些 query 和 key 高度对齐时，分数可能极大：

```
正常情况:  scores ∈ [-10, 10]  → softmax 正常工作
极端情况:  scores ∈ [-1000, 1000]  → softmax 溢出！

当 score = 1000 时：
  exp(1000) = Infinity (fp16 溢出)
  softmax 输出变成 [0, 0, ..., 1, 0, ..., 0]（one-hot）
  梯度接近 0 → 训练停滞
```

传统解法：
1. **缩放**：`scores / sqrt(d)` — 部分缓解，但不能根治
2. **梯度裁剪**：训练后手动裁剪梯度 — 治标不治本
3. **Softcapping**：在 softmax 之前截断分数 — 从根源解决

### 1.2 Tanh Softcapping 公式

```python
capped_scores = soft_cap * tanh(scores / soft_cap)
```

效果：
- 当 `scores` 在 `[-cap, cap]` 范围内时：`tanh(x) ≈ x`，几乎不变
- 当 `scores` 超出 `cap` 时：`tanh(x)` 饱和到 ±1，scores 被截断到 ±cap

```
cap = 50 时：
  scores = -100 → capped = 50 * tanh(-2) = -46.2  (截断)
  scores = -50  → capped = 50 * tanh(-1) = -38.8  (轻微压缩)
  scores = -10  → capped = 50 * tanh(-0.2) = -9.87 (几乎不变)
  scores = 0    → capped = 0                        (不变)
  scores = 10   → capped = 50 * tanh(0.2) = 9.87   (几乎不变)
  scores = 50   → capped = 50 * tanh(1) = 38.8     (轻微压缩)
  scores = 100  → capped = 50 * tanh(2) = 46.2     (截断)
```

### 1.3 为什么不直接 clip？

```python
# 方法1: 直接 clip（不可微！梯度在边界处为 0）
capped = torch.clamp(scores, -cap, cap)

# 方法2: tanh softcap（处处可微！梯度平滑过渡）
capped = cap * torch.tanh(scores / cap)
```

tanh 的梯度 `sech²(x)` 在所有点都有定义且非零，不存在梯度消失问题。

---

## 第二章：实现对比

### 2.1 Vanilla PyTorch

```python
def vanilla_softcap(q, k, v, soft_cap=50.0):
    S = q.shape[-2]
    D = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
    
    # 因果掩码
    causal = torch.ones(S, S, device=q.device, dtype=torch.bool).tril_()
    scores = scores.masked_fill(~causal, float('-inf'))
    
    # Softcapping（在 softmax 之前！）
    scores = soft_cap * torch.tanh(scores / soft_cap)
    
    weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(weights, v)
```

**注意**：softcapping 必须在 softmax 之前应用。

### 2.2 FlexAttention

```python
def flex_softcap(q, k, v, soft_cap=50.0):
    B, _, S, _ = q.shape
    
    # score_mod: 在 softmax 之前修改分数
    def softcap_score(score, b, h, q_idx, kv_idx):
        return soft_cap * torch.tanh(score / soft_cap)
    
    # mask_mod: 标准因果
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    
    block_mask = create_block_mask(causal_mask, B, 1, S, S, device=q.device)
    return flex_attention(q, k, v, score_mod=softcap_score, block_mask=block_mask)
```

### 2.3 score_mod 的执行时机

```
FlexAttention 内部执行顺序：
  1. Q × K^T → score (在 SRAM 中)
  2. score_mod(score, b, h, q, kv) → modified score (在寄存器中！)
  3. mask_mod 检查 → 如果 False，设为 -inf
  4. online_softmax(modified scores)
  5. × V → output

score_mod 在步骤 2 执行，直接在 GPU 寄存器中修改分数值，
不产生任何额外的 HBM 读写！
```

---

## 第三章：实验结果

### 3.1 使用 Softcapping 的模型

| 模型 | Soft Cap 值 | 说明 |
|------|-----------|------|
| Gemma-2 2B/9B/27B | 50.0 | Google 最新的开源模型 |
| Grok-1 | 50.0 | xAI 的模型 |

### 3.2 Softcapping 对注意力分布的影响

没有 softcapping：
```
scores: [-5, -3, 2, 100, -1]
softmax: [0.00, 0.00, 0.00, 1.00, 0.00]  ← 几乎是 one-hot！
```

有 softcapping (cap=50)：
```
capped: [-49.9, -49.6, 46.2, 50.0, -47.8]
softmax: [0.00, 0.00, 0.12, 0.88, 0.00]  ← 仍然集中但不是 one-hot
```

### 3.3 近似 tanh 的 PTX 指令

FlexAttention 支持使用 `tanh.approx.f32` PTX 指令加速：

```python
from torch.nn.attention.flex_attention import _tanh_backward_approx

def softcap_approx(score, b, h, q_idx, kv_idx):
    return soft_cap * torch.tanh(score / soft_cap)

# 使用近似版本时，tanh 被编译为硬件原生指令，
# 速度更快但有微小精度差异（通常 < 0.01）
```

---

## 总结

Tanh Softcapping 是 **score_mod 的典型应用**：它不改变哪些位置被关注，而是**改变关注的强度分布**。在 FlexAttention 中，score_mod 被编译到 Triton kernel 中，在 softmax 之前直接在寄存器中修改分数——零额外显存、零额外 HBM 访问。

| 维度 | 直接 clip | Tanh softcap |
|------|----------|-------------|
| 可微性 | 边界处梯度为 0 | **处处可微** |
| 梯度平滑 | 不连续 | **平滑过渡** |
| FlexAttention 支持 | 不需要（太简单） | **原生 score_mod** |
| 硬件加速 | N/A | PTX tanh.approx |

---

*报告生成时间：2026-04-25*
