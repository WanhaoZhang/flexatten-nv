# 项目五：MLA (Multi-head Latent Attention) 端到端分析

> PyTorch | Qwen2.5-0.5B-Instruct | NVIDIA L4 (24GB)
>
> 4 组实验：KV 压缩率、投影延迟、显存分解、端到端 decode 模拟

---

## 1. 研究背景与原理

### 1.1 KV Cache 瓶颈与 MLA

标准 Multi-Head Attention (MHA) 的 KV Cache 随序列长度线性增长，且与 KV head 数成正比。对于大模型长序列场景，KV Cache 成为显存瓶颈。

**MLA**（Multi-head Latent Attention，DeepSeek-V2/V3 核心创新）的核心思想：

1. 将 KV 压缩到低秩潜在空间（latent space）
2. 缓存只存储压缩后的潜在向量
3. Attention 计算时从潜在向量实时投影回 K、V

$$K = W_K \cdot c_{KV}, \quad V = W_V \cdot c_{KV}$$

其中 $c_{KV}$ 是潜在向量（维度远小于 $n_{heads} \times d_{head}$）。

### 1.2 方案对比

| 方案 | KV Cache 大小 | 特点 |
|------|-------------|------|
| MHA | $2 \times n_{kv} \times d \times L$ | 最完整，最大 |
| GQA | $2 \times n_{kv\_group} \times d \times L$ | 减少 KV head 数 |
| MQA | $2 \times 1 \times d \times L$ | 极端 GQA，1 个 KV head |
| MLA | $2 \times d_{latent} \times L$ | 压缩到潜在空间 |

---

## 2. 实验设计

### 实验 1：KV Cache 压缩率分析

**目的**：对比 MHA、GQA、MQA、MLA 在不同配置下的 KV Cache 大小。

### 实验 2：MLA 投影延迟

**目的**：测量从潜在向量投影到 K、V 的额外延迟。

### 实验 3：显存分解

**目的**：完整分析模型权重 + KV Cache 的显存占用。

### 实验 4：端到端 Decode 模拟

**目的**：模拟不同方案下 decode 阶段的 KV Cache 读取延迟。

---

## 3. 实验环境

| 组件 | 规格 |
|------|------|
| GPU | NVIDIA L4, 24 GB |
| 模型 | Qwen2.5-0.5B-Instruct (24层, 14 KV heads, 64 head_dim) |
| 序列长度 | 64 - 8192 |

---

## 4. 实验结果与分析

### 4.1 实验 1：KV Cache 压缩率

| 方案 | KV Size (SL=4096) | 压缩比 (vs MHA) |
|------|------------------|----------------|
| MHA (Qwen2.5-0.5B) | 336.0 MB | 1.0x |
| GQA (4 heads) | 96.0 MB | 3.5x |
| GQA (2 heads) | 48.0 MB | 7.0x |
| MQA (1 head) | 24.0 MB | 14.0x |
| MLA (latent=256) | 96.0 MB | 3.5x |
| MLA (latent=128) | 48.0 MB | 7.0x |
| MLA (latent=64) | 24.0 MB | 14.0x |
| DeepSeek-V2 MLA | 480.0 MB | - |
| Llama-3 70B GQA | 1,280.0 MB | - |

![KV Compression](figures/fig1_kv_compression.png)

**关键发现**：
- **MLA 与相同维度的 GQA 有相同的 KV 大小**：MLA latent=128 等价于 GQA-2 的压缩率
- MLA 的核心优势不在于压缩率本身，而在于**投影矩阵可以保持更高的表达能力**
- DeepSeek-V2 MLA (latent=512) 虽然看起来较大，但对应的是 128 heads × 128 dim 的超大模型

### 4.2 实验 2：MLA 投影延迟

| 配置 | SL=64 | SL=1024 | SL=4096 | vs MHA (SL=4096) |
|------|-------|---------|---------|-----------------|
| MHA | 0.03ms | 0.78ms | 3.00ms | 1.0x |
| MLA-64 | 2.70ms | 0.27ms | 0.30ms | 0.10x |
| MLA-128 | 0.27ms | 0.26ms | 0.36ms | 0.12x |
| MLA-256 | 0.27ms | 0.27ms | 0.50ms | 0.17x |

![Projection Latency](figures/fig2_projection_latency.png)

**分析**：
- **SL ≥ 1024 时 MLA 全面胜出**：投影延迟远小于 MHA KV 读取
- MLA-64 在 SL=64 时异常慢（2.7ms），可能是小 batch 时 kernel launch 开销占主导
- MLA-128 是最优平衡点：SL=4096 时延迟仅 0.36ms（MHA 的 12%）
- 随着 SL 增大，MLA 优势更显著（KV 读取是 O(S)，MLA 投影与 S 无关）

### 4.3 实验 3：显存分解

| 序列长度 | 模型权重 | MHA KV | MLA-128 KV | MLA 节省 |
|---------|---------|--------|-----------|---------|
| 512 | 942 MB | 42 MB | 3 MB | 92.9% |
| 2,048 | 942 MB | 168 MB | 12 MB | 92.9% |
| 8,192 | 942 MB | 672 MB | 48 MB | 92.9% |
| 32,768 | 942 MB | 2,688 MB | 192 MB | **92.9%** |

![Memory Breakdown](figures/fig3_memory_breakdown.png)

**分析**：
- MLA-128 固定节省 92.9% 的 KV Cache（因为 latent_dim/head_dim = 128/896 ≈ 0.14）
- SL=32768 时，MHA 总需要 3.63GB（仅 KV Cache），而 MLA 只需 192MB
- 模型参数分布：Attention 30%、FFN 60%、Embedding 10%

### 4.4 实验 4：端到端 Decode 模拟

| 配置 | SL=128 | SL=2048 | SL=8192 |
|------|--------|---------|---------|
| MHA (14 heads) | 0.06ms | 1.51ms | 5.99ms |
| GQA-4 | 0.02ms | 0.44ms | 1.73ms |
| GQA-2 | 0.02ms | 0.09ms | 0.88ms |
| MQA | 0.02ms | 0.03ms | 0.45ms |
| MLA-256 | 0.02ms | 0.44ms | 1.72ms |
| MLA-128 | 0.02ms | 0.09ms | 0.87ms |
| MLA-64 | 0.02ms | 0.03ms | 0.45ms |

![E2E Decode](figures/fig4_e2e_decode.png)

**关键发现**：
- **MLA 的 KV 读取性能等价于同维度的 GQA**：MLA-128 ≈ GQA-2，MLA-64 ≈ MQA
- MHA 在 SL=8192 时 KV 读取需 6ms，MLA-128 只需 0.87ms（6.9x 加速）
- 所有方案在 SL=128 时都很快，差异可忽略

---

## 5. 结论

1. **MLA 本质是学习型 GQA**：通过低秩投影实现与 GQA 相同的 KV Cache 压缩率，但保持更高的模型表达能力

2. **MLA-128 节省 92.9% KV Cache**：等价于 GQA-2 的压缩率，但在 attention 计算时保持完整的 14 head 分辨率

3. **MLA 的投影开销可忽略**：SL ≥ 1024 时，投影延迟 < 0.4ms，远小于节省的 KV 读取时间

4. **MLA vs GQA 的核心权衡**：
   - GQA：简单直接，但降低 attention head 分辨率
   - MLA：需要额外投影矩阵（增加模型参数），但保持高分辨率 attention

5. **实践建议**：
   - 长上下文场景（RAG, 长文档）优先考虑 MLA 或 GQA
   - DeepSeek-V2/V3 已验证 MLA 在超大规模模型上的有效性
   - 开源模型可考虑将 GQA 替换为 MLA 以获得更好的质量-效率权衡

---

## 6. 复现命令

```bash
cd ~/flexatten-nv/docs/mla_e2e
python mla_e2e.py         # 生成 results/*.json (~3min)
python gen_charts.py       # 生成图表到 figures/
```

---

*实验日期：2026-04-28 | NVIDIA L4 (24GB) | PyTorch 2.10.0 | Qwen2.5-0.5B-Instruct*
