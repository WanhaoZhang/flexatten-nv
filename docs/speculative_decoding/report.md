# 项目四：Speculative Decoding 加速分析

> PyTorch + vLLM 0.19.1 | Qwen2.5-0.5B-Instruct | NVIDIA L4 (24GB)
>
> 4 组实验：自回归 decode 分析、投机解码模拟、理论加速分析、vLLM 吞吐量

---

## 1. 研究背景

### 1.1 自回归解码的 memory-bound 困境

LLM 的 decode 阶段面临根本性的效率问题：**每生成 1 个 token，都需要加载全部模型权重**。对于 Qwen2.5-0.5B（942MB FP16），L4 的 300 GB/s 带宽意味着：

$$\text{理论极限} = \frac{300 \text{ GB/s}}{942 \text{ MB}} \approx 318 \text{ tok/s}$$

实际单 stream decode 约 35 tok/s，效率仅 11%。其余 89% 时间花在 kernel launch、内存延迟和 GPU 空闲上。这是所有 memory-bound 操作的通病——**GPU 算力大量浪费**。

Batching 可以提升利用率（BS=32 时达 5,500 tok/s），但对**延迟敏感场景**（如交互式对话），用户只等待单个请求的响应，batch 并不能减少 per-token 延迟。

### 1.2 Speculative Decoding 原理

Speculative Decoding（投机解码）是解决单 stream decode 低效的核心方案。核心思想：

```
传统: Target → tok1 → Target → tok2 → Target → tok3 ...  (N 次 forward)
投机: Draft → tok1,tok2,tok3,tok4 → Target 验证全部 → 接受 tok1,tok2 → 丢弃 tok3,tok4
```

1. **Draft phase**：用小模型（3-10x 更小）快速生成 γ 个候选 token
2. **Verify phase**：用 target model 一次 forward pass 并行验证所有 γ 个 token
3. **Accept/Reject**：保留通过验证的 token，丢弃第一个不匹配的及之后的所有 token

**接受率推导**：设 draft model 与 target model 的 token 分布分别为 $q(x)$ 和 $p(x)$，接受概率：

$$\alpha = \mathbb{E}\left[\min\left(1, \frac{p(x)}{q(x)}\right)\right]$$

当 draft model 越接近 target model，α 越接近 1。每轮期望生成的 token 数：

$$\mathbb{E}[\text{tokens}] = \frac{1 - \alpha^\gamma}{1 - \alpha}$$

γ=4, α=0.9 时，期望 3.44 个 token/轮，即 3.44x 加速（扣除 draft 开销后约 2.6x）。

### 1.3 Speculative Decoding 的变体

| 方法 | Draft 来源 | 特点 |
|------|-----------|------|
| Classic SpecDec | 独立小模型 | 最简单，需要额外的 draft model |
| Medusa | Multiple heads 接在 LM head 后 | 无需额外模型，多 head 并行预测 |
| Eagle | 利用 1 层 transformer + feature | 极低延迟 draft，接近 target 质量 |
| Self-speculative | 同一模型 early exit | 无需额外模型，利用中间层输出 |

### 1.4 研究目标

本实验的核心目标是**评估 Speculative Decoding 在 L4 上的实际可行性**：

1. **Decode 性能画像**：0.5B 模型在 L4 上的 per-token 延迟和 KV Cache 影响有多大？
2. **同模型投机测试**：用同一模型模拟 draft+verify，验证投机解码的加速机制
3. **理论最优配置**：不同 (α, γ) 组合的理论加速比热力图
4. **vLLM 基线**：vLLM 批量推理的吞吐量上限

---

## 2. 实验设计

### 2.1 实验组与目标

| 实验 | 目标 | 方法 |
|------|------|------|
| Exp1 | 画像 decode 阶段的延迟和 KV Cache 影响 | 不同 prompt 长度, 测量 per-token 延迟 |
| Exp2 | 模拟 draft+verify 流程, 测量实际加速 | 同一模型作 draft, K=2/4/6/8/10/16 |
| Exp3 | 计算理论加速比热力图 | α=0.5-0.99, γ=1-16 |
| Exp4 | 测量 vLLM 批量推理吞吐上限 | BS=8, max_tok=16/32/64/128/256 |

---

## 3. 实验环境

| 组件 | 规格 |
|------|------|
| GPU | NVIDIA L4, 24 GB |
| 模型 | Qwen2.5-0.5B-Instruct |
| vLLM | 0.19.1 |
| PyTorch | 2.10.0 |

---

## 4. 实验结果与分析

### 4.1 实验 1：自回归 Decode 性能

| Prompt Length | Decode (ms/tok) | KV Cache (MB) | tok/s |
|--------------|----------------|--------------|-------|
| 64 | 31.29 | 1.0 | 32 |
| 128 | 27.59 | 1.7 | 36 |
| 256 | 28.05 | 3.2 | 36 |
| 512 | 28.19 | 6.2 | 35 |
| 1024 | 28.37 | 12.2 | 35 |
| 2048 | 29.21 | 24.2 | 34 |

![Decode Latency](figures/fig1_decode_latency.png)

**分析**：
- Decode 延迟稳定在 28-31 ms/tok（~35 tok/s），几乎不受 KV Cache 大小影响
- 这是因为 0.5B 模型的 KV Cache 读取（最大 24MB）远小于权重加载（942MB）
- **权重加载是 decode 瓶颈**，不是 KV Cache

### 4.2 实验 2：投机解码模拟

| Draft Size (K) | Draft Time (ms) | Verify (ms) | 接受率 | 加速比 |
|---------------|----------------|------------|-------|-------|
| 2 | 29.2 | 48.6 | 1.000 | 0.77x |
| 4 | 83.7 | 27.5 | 1.000 | 1.08x |
| 6 | 139.0 | 28.2 | 1.000 | 1.08x |
| 8 | 197.3 | 27.9 | 0.750 | 1.07x |
| 10 | 251.5 | 30.2 | 0.600 | 1.06x |
| 16 | 420.2 | 28.4 | 0.375 | 1.07x |

![Speculation Sim](figures/fig2_speculation_sim.png)

**分析**：
- **使用同一模型作为 draft model 时，加速比仅 1.08x**：因为 draft 模型与 target 一样慢
- 真正的投机解码需要 **3-10x 更小的 draft model**（如 0.5B draft + 7B target）
- K=2 时甚至更慢（0.77x），因为 verify 开销大于节省的 decode 步骤
- 接受率从 K=8 开始下降（0.75），说明 draft token 越多偏差越大

### 4.3 实验 3：理论加速分析

![Speedup Heatmap](figures/fig3_speedup_heatmap.png)

**关键数据点**：

| 接受率 α | γ=4 | γ=8 | γ=16 |
|---------|-----|-----|------|
| 0.70 | 1.50x | 0.98x | 0.53x |
| 0.80 | 1.97x | 1.50x | 0.84x |
| 0.90 | **2.60x** | **2.49x** | 1.75x |
| 0.95 | 3.48x | 3.82x | 2.96x |

**分析**：
- α ≥ 0.8, γ=4-8 时可获得 1.5-2.5x 加速
- **γ=4 是最优 draft size**：在大多数 α 下都有正收益
- γ 过大反而降低加速比（draft 时间线性增长，但接受率指数下降）
- α=0.9, γ=4-8 是实际推荐的配置范围

### 4.4 实验 4：vLLM 吞吐量基线

| Max Tokens | 总耗时 (ms) | Output TPS | Total TPS |
|-----------|-----------|-----------|----------|
| 16 | 118 | 1,085 | 1,178 |
| 32 | 180 | 1,424 | 1,564 |
| 64 | 327 | 1,566 | 1,690 |
| 128 | 642 | 1,596 | 1,680 |
| 256 | 1,277 | 1,603 | 1,657 |

![vLLM Throughput](figures/fig4_vllm_throughput.png)

**分析**：
- max_tok ≥ 64 后吞吐量趋于饱和（~1,600 tok/s）
- 短输出（16 tokens）吞吐量低是因为 prefill 开销占比大
- vLLM 的 continuous batching 在 BS=8 时效率很高

---

## 5. 结论

1. **0.5B 模型作为 draft model 无法有效加速同尺寸 target**：draft 开销抵消了验证收益

2. **理论最优配置：α=0.9, γ=4-8，可获 2-2.5x 加速**：需要 3-10x 更小的 draft model

3. **Decode 延迟稳定在 28-31ms/tok**：权重加载是唯一瓶颈，KV Cache 影响可忽略

4. **实际应用建议**：
   - 使用 0.1B 模型 draft + 7B target 是最佳配置
   - vLLM 的 speculative decoding 通过 `--speculative-model` 参数配置
   - 在延迟敏感场景（单用户），speculative decoding 可将 TTFT 后的吞吐提升 2x+
   - 在吞吐敏感场景（多用户 batch），continuous batching 已足够

---

## 6. 复现命令

```bash
cd ~/flexatten-nv/docs/speculative_decoding
python speculative_decoding.py   # 生成 results/*.json (~5min)
python gen_charts.py              # 生成图表到 figures/
```

---

*实验日期：2026-04-28 | NVIDIA L4 (24GB) | PyTorch 2.10.0 + vLLM 0.19.1 | Qwen2.5-0.5B-Instruct*
