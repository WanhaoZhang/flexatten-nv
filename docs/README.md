# FlexAttention 实验项目

> NVIDIA L4 (24GB) | PyTorch 2.6.0+cu124 | Triton 3.2.0 | vLLM 0.19.1 | Qwen2.5-0.5B-Instruct

---

## 一、核心实验系列（项目一 ~ 六）

FlexAttention + vLLM 在 L4 上的系统性实验，覆盖 backward 性能、稀疏注意力、KV Cache 量化、投机解码、MLA 架构、Serving 框架对比。

| 项目 | 报告 | 核心发现 | 实验/图表 |
|------|------|---------|----------|
| 项目一 | [backward_benchmark/report.md](backward_benchmark/report.md) | FlexAttention 比 SDPA 慢 70x；稀疏 mask 未生效；backward/forward ≈ 1.1x | 4 实验 / 6 图 |
| 项目二 | [nsa_deviation/report.md](nsa_deviation/report.md) | 93.8% 理论稀疏率仅 1.02x 实际加速；Triton kernel 静态遍历是根因 | 3 实验 / 图 |
| 项目三 | [int4_kv_cache/report.md](int4_kv_cache/report.md) | FP8 KV 容量翻倍、精度 cos_sim=0.999995；INT4 理论 4x 但需专用 kernel | 4 实验 / 4 图 |
| 项目四 | [speculative_decoding/report.md](speculative_decoding/report.md) | 同模型 draft 仅 1.08x；理论 α=0.9 γ=4 可达 2.6x 加速 | 4 实验 / 4 图 |
| 项目五 | [mla_e2e/report.md](mla_e2e/report.md) | MLA-128 节省 92.9% KV Cache；投影延迟 <0.4ms；性能等价同维度 GQA | 4 实验 / 4 图 |
| 项目六 | [trt_llm_serving/report.md](trt_llm_serving/report.md) | vLLM BS=64 达 8,929 tok/s；TRT-LLM INT4 理论 4x decode 加速 | 4 实验 / 4 图 |

---

## 二、Serving & 调度实验（项目七 ~ 八）

| 项目 | 报告 | 核心发现 | 实验/图表 |
|------|------|---------|----------|
| 项目七 | [multi_lora/report.md](multi_lora/report.md) | Multi-LoRA 多租户适配器调度与显存隔离 | 5 实验 / 5 图 |
| 项目八 | [prefix_caching/report.md](prefix_caching/report.md) | Automatic Prefix Caching 在 RAG 中的真实收益 | 实验 / 图 |

---

## 三、架构与推理实验（项目十一 ~ 十五）

| 项目 | 报告 | 核心发现 | 实验/图表 |
|------|------|---------|----------|
| 项目五(B) | [mla_gqa_analysis/report.md](mla_gqa_analysis/report.md) | MHA/GQA/MLA 架构端到端对比：显存公式 + GPU 实测 | 实验 / 4 图 |
| 项目十一 | [galore_dora/report.md](galore_dora/report.md) | GaLore vs DoRA vs LoRA vs Full FT 显存/收敛对比 | 实验 / 图 |
| 项目十三 | [vlm_kv_cache/report.md](vlm_kv_cache/report.md) | VLM 视觉 Token 的 KV Cache 显存灾难分析 | 实验 / 4 图 |
| 项目十四 | [continuous_batching/report.md](continuous_batching/report.md) | Continuous Batching 调度时间轴分析 | 实验 / 图 |
| 项目十五 | [exl2_awq/report.md](exl2_awq/report.md) | EXL2 vs AWQ INT4 权重量化推理对比 | 实验 / 图 |

---

## 四、FlexAttention 源码分析

### 综合分析报告

| 报告 | 内容 |
|------|------|
| [deep_dive/report.md](deep_dive/report.md) | FlexAttention 深度原理剖析：痛点→解法 |
| [doc_packing/report.md](doc_packing/report.md) | Document Packing 专题：三种实现全解析 |
| [pattern_analysis/report.md](pattern_analysis/report.md) | 8 种注意力模式全解析：Vanilla vs FlexAttention |
| [source_analysis/report.md](source_analysis/report.md) | 源码级分析：Vanilla→SDPA→FlexAttention 三条路径 |
| [flex_internals/report.md](flex_internals/report.md) | FlexAttention 内部机制分析 |
| [kv_cache_bandwidth/report.md](kv_cache_bandwidth/report.md) | KV Cache 带宽分析 |

### 单一模式深度报告

| 报告 | 模式 | 核心特点 |
|------|------|---------|
| [SLIDING_WINDOW_REPORT.md](SLIDING_WINDOW_REPORT.md) | Sliding Window | O(S×W) 复杂度，稀疏率 88-97% |
| [ALIBI_REPORT.md](ALIBI_REPORT.md) | ALiBi | score_mod 经典应用，消除 for 循环 |
| [PREFIX_LM_REPORT.md](PREFIX_LM_REPORT.md) | Prefix LM | mask OR 组合，前缀双向+后缀因果 |
| [SOFTCAP_REPORT.md](SOFTCAP_REPORT.md) | Tanh Softcapping | 数值稳定性，score_mod 防溢出 |

### GPU 执行链路分析

| 报告 | 内容 |
|------|------|
| [trace/FLEXATTENTION_GPU_PIPELINE_ANALYSIS.md](trace/FLEXATTENTION_GPU_PIPELINE_ANALYSIS.md) | Python API → GPU kernel 完整链路 |
| [trace/CAUSAL_FLEXATTENTION_CUTEDSL_TRACE.md](trace/CAUSAL_FLEXATTENTION_CUTEDSL_TRACE.md) | Causal FlexAttention CuteDSL trace |
| [trace/FLEXATTENTION_FLASH_CUTEDSL_BACKEND_REPORT.md](trace/FLEXATTENTION_FLASH_CUTEDSL_BACKEND_REPORT.md) | Flash/CuteDSL backend 分析 |
| [trace/FLEXATTENTION_CATLASS_INTEGRATION_DESIGN.md](trace/FLEXATTENTION_CATLASS_INTEGRATION_DESIGN.md) | CUTLASS 集成设计 |
| [trace/NPU_FLEXATTENTION_TRITON_ANALYSIS.md](trace/NPU_FLEXATTENTION_TRITON_ANALYSIS.md) | NPU FlexAttention Triton 分析 |
| [trace/NPU_MM_CATLASS_ANALYSIS.md](trace/NPU_MM_CATLASS_ANALYSIS.md) | NPU 矩阵乘法 CUTLASS 分析 |

### 其他实验

| 报告 | 内容 |
|------|------|
| [liger_kernel/report.md](liger_kernel/report.md) | Liger Kernel 实验分析 |
| [moe_wall/report.md](moe_wall/report.md) | MoE 内存墙分析 |
| [paged_attention/report.md](paged_attention/report.md) | PagedAttention 原理分析 |
| [triton_fusion/report.md](triton_fusion/report.md) | Triton kernel 融合分析 |
| [mla/report.md](mla/report.md) | MLA 早期实验 |
| [spec_decoding/report.md](spec_decoding/report.md) | 投机解码早期实验 |
| [trtllm_serving/report.md](trtllm_serving/report.md) | TRT-LLM Serving 早期实验 |

---

## 环境信息

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA L4, 24GB VRAM, 300 GB/s 带宽 |
| PyTorch | 2.6.0+cu124 |
| Triton | 3.2.0 |
| vLLM | 0.19.1 |
| 模型 | Qwen2.5-0.5B-Instruct (942MB FP16) |

## 快速复现

```bash
# SSH 到 GCP
ssh zhangwh@104.197.143.214

# 激活环境
conda activate flexatten

# 运行实验（每个 ~5-15 min）
cd ~/flexatten-nv/docs/<project_dir>
python <experiment>.py      # 生成 results/*.json
python gen_charts.py         # 生成 figures/*.png
```

---

*最后更新：2026-04-28*
