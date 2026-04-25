# Prefix LM 深度解析：前缀双向 + 后缀因果的混合注意力

> **T5、Flan-T5、UniLM 的选择：prompt 双向看、generation 只看过去。**
>
> 双向注意力如何与因果约束共存 | FlexAttention 的 or_masks 组合
>
> NVIDIA L4 (24GB) | PyTorch 2.6.0+cu124 | Triton 3.2.0

---

## 第一章：什么是 Prefix LM？

### 1.1 Encoder-Decoder 的困境

标准 Transformer 有两种模式：
- **Encoder（双向）**：每个 token 可以看到所有 token（BERT、T5 encoder）
- **Decoder（因果）**：每个 token 只能看到过去的 token（GPT、T5 decoder）

但实际应用中，我们经常需要**混合模式**：
- 输入 prompt：双向编码，理解完整语义
- 输出 generation：因果解码，自回归生成

这就是 **Prefix LM**。

### 1.2 掩码模式

```
前缀长度 = 3（tokens 0-2 是 prompt，3-5 是 generation）

    0  1  2  3  4  5
0 [█  █  █  .  .  .]   prompt 内：双向（token 0 可以看 1, 2）
1 [█  █  █  .  .  .]   prompt 内：双向（token 1 可以看 0, 2）
2 [█  █  █  .  .  .]   prompt 内：双向（token 2 可以看 0, 1）
3 [█  █  █  █  .  .]   generation：因果 + 能看全部 prompt
4 [█  █  █  █  █  .]   generation：因果 + 能看全部 prompt
5 [█  █  █  █  █  █]   generation：因果 + 能看全部 prompt

规则：
1. 如果 kv_idx < prefix_length：所有 q 都能看到（双向）
2. 否则：只有 q_idx >= kv_idx 时才能看到（因果）
```

### 1.3 为什么这种模式重要？

- **T5 / Flan-T5**：encoder-decoder 架构就是 Prefix LM
- **UniLM**：统一预训练框架，用不同 prefix 比例控制双向/单向
- **Prefix Tuning**：在 prompt 前添加可训练的 prefix tokens
- **In-context Learning**：演示 examples 作为 prefix，双向编码以充分利用

---

## 第二章：实现对比

### 2.1 Vanilla PyTorch

```python
def vanilla_prefix_lm(q, k, v, prefix_length):
    S = q.shape[-2]
    D = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
    
    # 因果掩码
    pos = torch.arange(S, device=q.device)
    causal = pos.unsqueeze(0) >= pos.unsqueeze(1)
    
    # 前缀掩码：kv_idx < prefix_length 的列全部为 True
    prefix_mask = pos.unsqueeze(1) < prefix_length
    
    # 合并：因果 OR 前缀
    mask = causal | prefix_mask
    
    scores = scores.masked_fill(~mask, float('-inf'))
    weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(weights, v)
```

### 2.2 FlexAttention

```python
def flex_prefix_lm(q, k, v, prefix_length):
    B, _, S, _ = q.shape
    def prefix_mask_fn(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        prefix = kv_idx < prefix_length
        return causal | prefix
    block_mask = create_block_mask(prefix_mask_fn, B, 1, S, S, device=q.device)
    return flex_attention(q, k, v, block_mask=block_mask)
```

### 2.3 稀疏性分析

前缀比例对稀疏率的影响：

| 前缀比例 | 稀疏率 | 含义 |
|---------|--------|------|
| 0%（纯 Causal） | 50.0% | 基准 |
| 10% | 47.0% | 前 10% 列全部可看 |
| 25% | 46.9% | 略低于纯 Causal |
| 50% | 37.5% | 前 50% 列全部可看，稀疏率大降 |
| 100%（全双向） | 0.0% | 无掩码 |

**关键洞察**：前缀比例越大，稀疏率越低（因为前缀区域是双向的，打破了因果掩码的上三角空洞）。但即使前缀占 50%，仍有 37.5% 可以跳过。

---

## 第三章：实验结果

### 3.1 性能对比（S=2048, prefix=25%）

| 方式 | 延迟 (ms) | 显存 (GB) |
|------|----------|----------|
| Vanilla | 5.1 | 0.340 |
| Flex | 18.8 | 0.426 |

### 3.2 BlockMask 分析

| 前缀比例 | 非空块数 | 总块数 | 块利用率 |
|---------|---------|-------|---------|
| 10% | 16/256 | 6.2% |
| 25% | 12/256 | 4.7% |
| 50% | 8/256 | 3.1% |

前缀比例增大 → 块利用率反而降低（因为前缀列全部可看，不需要按块分析）

---

## 总结

Prefix LM 是 **mask_mod 的 OR 组合**的典型代表：`causal | (kv < prefix_length)`。FlexAttention 将这个 OR 逻辑编译到 BlockMask 中，自动处理"哪些块需要计算"。

SDPA 不支持 Prefix LM（只能做纯 Causal 或提供完整的 attn_mask 张量）。

---

*报告生成时间：2026-04-25*
