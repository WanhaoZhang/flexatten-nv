# 项目五：从 GQA 到 MLA 的端到端推理 — 显存公式推演

> 理论分析 | MHA / GQA / MLA 架构对比 | KV Cache 显存建模

## 1. 三种注意力架构

### 1.1 Multi-Head Attention (MHA)
Llama-2 标准。每个 head 有独立的 K 和 V。

```
Q: [B, num_heads, S, head_dim]
K: [B, num_heads, S, head_dim]   ← 每个 head 独立
V: [B, num_heads, S, head_dim]   ← 每个 head 独立
```

**KV Cache per token**: `2 × num_heads × head_dim × bytes_per_elem`

### 1.2 Grouped-Query Attention (GQA)
Llama-3 / Qwen-2 标准。K 个 Q head 共享 1 组 KV。

```
Q: [B, num_heads, S, head_dim]
K: [B, kv_heads, S, head_dim]    ← kv_heads << num_heads
V: [B, kv_heads, S, head_dim]    ← kv_heads << num_heads
```

**KV Cache per token**: `2 × kv_heads × head_dim × bytes_per_elem`

### 1.3 Multi-Head Latent Attention (MLA)
DeepSeek-V2 核心。K/V 通过低秩投影压缩到潜在空间。

```
Q: [B, num_heads, S, head_dim]
C: [B, 1, S, latent_dim]          ← 只存压缩后的 latent
推理时: K = C @ W_k, V = C @ W_v   ← 在线解压
```

**KV Cache per token**: `2 × latent_dim × bytes_per_elem`（latent_dim << num_heads × head_dim）

## 2. 显存公式推演

### 2.1 统一公式

```
KV_Cache_Bytes = 2 × seq_len × batch_size × KV_DIM × bytes_per_elem
```

其中 KV_DIM:
- MHA: `num_heads × head_dim`
- GQA: `kv_heads × head_dim`
- MLA: `latent_dim`

### 2.2 具体计算（7B 模型，FP16，B=1）

| 架构 | 参数 | KV_DIM | seq=4K | seq=32K | seq=128K |
|------|------|--------|--------|---------|----------|
| **MHA** (Llama-2) | 32 heads, 128 dim | 4096 | 0.062 GB | 0.500 GB | 2.000 GB |
| **GQA-8** (Llama-3) | 8 kv_heads, 128 dim | 1024 | 0.016 GB | 0.125 GB | 0.500 GB |
| **GQA-4** (Qwen-2) | 4 kv_heads, 128 dim | 512 | 0.008 GB | 0.062 GB | 0.250 GB |
| **MLA-512** (DeepSeek) | latent_dim=512 | 512 | 0.008 GB | 0.062 GB | 0.250 GB |

### 2.3 关键发现

1. **MHA → GQA-8**: KV Cache 缩小 **4x**（128K: 2.0GB → 0.5GB）
2. **MHA → GQA-4**: KV Cache 缩小 **8x**（128K: 2.0GB → 0.25GB）
3. **GQA-4 ≈ MLA-512**: 理论 KV Cache 大小相同，但 MLA 在推理时有额外优势

### 2.4 MLA 的真正优势不在显存大小

MLA 的 latent_dim=512 与 GQA-4 的 kv_heads×head_dim=512 相同时，纯显存占用相同。MLA 的真正优势：

1. **推理时 KV 解压有计算复用**：latent → K/V 的投影矩阵可以与 RoPE、注意力计算融合
2. **训练时更低秩的正则化效果**：latent 压缩迫使模型学习更紧凑的 KV 表示
3. **吸收（Absorption）优化**：在某些情况下，W_k 可以被吸收到 W_q 中，避免显式解压

## 3. 最大上下文长度（L4 24GB）

假设 7B 模型权重占 14GB（FP16），剩余 8GB 给 KV Cache：

| 架构 | bytes/token | 最大上下文 |
|------|------------|-----------|
| MHA FP16 | 16,384 | ~524K tokens |
| GQA-8 FP16 | 4,096 | ~2.1M tokens |
| GQA-8 FP8 | 2,048 | ~4.2M tokens |
| GQA-8 INT4 | 1,024 | ~8.4M tokens |
| GQA-4 FP16 | 2,048 | ~4.2M tokens |
| GQA-4 FP8 | 1,024 | ~8.4M tokens |
| MLA-512 FP16 | 2,048 | ~4.2M tokens |
| MLA-512 FP8 | 1,024 | ~8.4M tokens |

**结论**：GQA-4 + FP8 和 MLA-512 + FP8 在 L4 上可支持 **8.4M tokens** 的上下文（理论上界，实际受 batch 和框架开销限制）。

## 4. Decode 带宽瓶颈分析

在自回归解码阶段，每步需要读取完整 KV Cache：

```
T_decode ≈ KV_Bytes / Memory_Bandwidth
L4 Memory Bandwidth: 300 GB/s
```

| 架构 | seq=4K decode | seq=32K decode | seq=128K decode |
|------|--------------|----------------|-----------------|
| GQA-8 FP16 | 0.053 ms (18.9K tok/s) | 0.42 ms (2.4K tok/s) | 1.67 ms (600 tok/s) |
| GQA-8 FP8 | 0.027 ms (37.8K tok/s) | 0.21 ms (4.7K tok/s) | 0.83 ms (1.2K tok/s) |
| GQA-8 INT4 | 0.013 ms (75.5K tok/s) | 0.10 ms (9.5K tok/s) | 0.42 ms (2.4K tok/s) |
| MLA-512 FP16 | 0.027 ms (37.8K tok/s) | 0.21 ms (4.7K tok/s) | 0.83 ms (1.2K tok/s) |
| MLA-512 FP8 | 0.014 ms (75.5K tok/s) | 0.10 ms (9.5K tok/s) | 0.42 ms (2.4K tok/s) |

**关键发现**：
- FP8 → INT4 的 decode 吞吐提升 **2x**（因为带宽减半）
- 128K 上下文时，GQA-8 FP16 的理论 decode 速度仅 600 tok/s
- 这解释了为什么工业界在疯狂卷 INT4 KV Cache

## 5. 量化精度退化风险

### 5.1 GQA 下的量化精度退化更大？

理论分析：
- GQA 的 K/V 是多个 Q head 共享的，量化误差影响所有相关 head
- MHA 每个 head 独立，量化误差只影响一个 head
- MLA 的 latent 维度更低，量化误差可能被解压投影放大

**风险排序**：MLA > GQA > MHA（从高到低）

### 5.2 实际影响

INT4 KV Cache 的精度退化主要体现在：
1. 长文本的"大海捞针"任务（needle-in-a-haystack）
2. 长上下文的指令遵循能力
3. 代码生成和数学推理（对数值精度敏感）

## 6. 结论

1. **MLA 的显存优势被高估**：在相同 latent_dim 下，MLA 与 GQA-4 的 KV Cache 大小相同
2. **MLA 的真正优势在计算效率**：通过 Absorption 优化减少推理计算量
3. **INT4 KV Cache 是带宽瓶颈的终极解法**：decode 吞吐翻倍，但需要验证精度退化
4. **128K 上下文的 decode 受带宽限制严重**：GQA-8 FP16 理论仅 600 tok/s

---

*分析日期：2026-04-28 | 基于数学建模 + L4 硬件参数*
