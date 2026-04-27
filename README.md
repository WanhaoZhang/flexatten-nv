# FlexAttention 实验项目

> 基于 [attention-gym](https://github.com/pytorch/attention-gym) | NVIDIA L4 (24GB) | PyTorch 2.6.0+cu124 | Triton 3.2.0

FlexAttention 是 PyTorch 2.5+ 引入的灵活注意力机制 API，允许用纯 Python 函数描述注意力修改（mask_mod / score_mod），框架自动编译为 Triton kernel。本仓库通过系统化实验，全面剖析其原理、性能和适用场景。

---

## 目录结构

每个报告都有独立的文件夹，包含报告、实验脚本、绘图脚本、实验数据和图表：

```
docs/
├── flex_internals/        # FlexAttention 原理深度剖析
│   ├── report.md          # 报告正文
│   ├── flex_internals_experiment.py  # 实验脚本
│   ├── plot_flex_internals.py        # 绘图脚本
│   ├── flex_internals_results.json   # 实验数据
│   └── figures/           # 图表 (6张)
├── mla/                   # Multi-Head Latent Attention
│   ├── report.md
│   ├── mla_experiment.py
│   ├── plot_mla.py
│   ├── mla_results.json
│   └── figures/           # 图表 (8张)
├── paged_attention/       # Paged Attention
│   ├── report_cn.md       # 中文报告
│   ├── report_en.md       # English report
│   ├── paged_attention_experiment.py
│   ├── plot_paged_attention.py
│   ├── paged_attention_results.json
│   └── figures/           # 图表 (8张)
├── doc_packing/           # Document Packing 专题
│   ├── report.md
│   ├── doc_packing_experiments.py
│   ├── doc_packing_results.json
│   └── figures/           # 图表 (5张)
├── pattern_analysis/      # 8种注意力模式全解析
│   ├── report.md
│   ├── flexatten_pattern_analysis.py
│   ├── pattern_analysis_results.json
│   └── figures/           # 图表 (9张)
├── source_analysis/       # 源码级分析
│   ├── report.md
│   ├── flexatten_source_analysis.py
│   ├── source_analysis_results.json
│   └── figures/           # 图表 (7张)
├── deep_dive/             # FlexAttention 深度原理剖析
│   ├── report.md
│   ├── flexatten_deep_dive.py
│   ├── deep_dive_results.json
│   └── figures/           # 图表 (8张)
├── trace/                 # GPU 链路 & CuteDSL 接入分析
│   ├── CAUSAL_FLEXATTENTION_CUTEDSL_TRACE.md
│   ├── FLEXATTENTION_FLASH_CUTEDSL_BACKEND_REPORT.md
│   └── causal_attention_trace/   # trace 实验脚本
├── SLIDING_WINDOW_REPORT.md  # Sliding Window Attention
├── ALIBI_REPORT.md           # ALiBi 位置编码
├── PREFIX_LM_REPORT.md       # Prefix LM 注意力
└── SOFTCAP_REPORT.md         # Tanh Softcapping
```

---

## 报告索引

### 综合分析报告

| 报告 | 内容 | 图表 |
|------|------|------|
| [FlexAttention 原理深度剖析](docs/flex_internals/report.md) | 9组实验：BlockMask结构、score_mod编译、稀疏性vs性能、编译开销、延迟对比、PT 2.5.1 vs 2.6.0 对比 | 6 |
| [FlexAttention 深度原理剖析](docs/deep_dive/report.md) | 痛点→解法：内存爆炸、带宽饥饿、工程噩梦 | 8 |
| [Document Packing 专题](docs/doc_packing/report.md) | 三种实现全解析：Vanilla / SDPA / FlexAttention | 5 |
| [8种注意力模式全解析](docs/pattern_analysis/report.md) | Vanilla vs FlexAttention：Causal/SW/Prefix/ALiBi/Softcap/Dilated | 9 |
| [源码级分析](docs/source_analysis/report.md) | Vanilla → SDPA → FlexAttention 三条执行路径 | 7 |
| [Causal FlexAttention GPU链路](docs/trace/CAUSAL_FLEXATTENTION_CUTEDSL_TRACE.md) | Dynamo/HOP/Inductor/Triton trace、CuteDSL/CUTLASS 接入分析 | - |
| [FlexAttention FLASH/CuteDSL 后端复测](docs/trace/FLEXATTENTION_FLASH_CUTEDSL_BACKEND_REPORT.md) | 确认 FlexAttention 进入 CuteDSL/CUTLASS 调用链 | - |

### 单一模式深度报告

| 报告 | 核心发现 |
|------|---------|
| [Sliding Window Attention](docs/SLIDING_WINDOW_REPORT.md) | O(S×W) 复杂度，稀疏率 88-97%，SDPA 不支持 |
| [ALiBi 位置编码](docs/ALIBI_REPORT.md) | score_mod 经典应用，消除 for 循环，支持长度外推 |
| [Prefix LM 注意力](docs/PREFIX_LM_REPORT.md) | mask_mod OR 组合，前缀双向 + 后缀因果 |
| [Tanh Softcapping](docs/SOFTCAP_REPORT.md) | 数值稳定性方案，score_mod 防溢出，PTX 硬件加速 |

### 高级注意力机制报告

| 报告 | 内容 | 图表 |
|------|------|------|
| [Multi-Head Latent Attention](docs/mla/report.md) | DeepSeek-V2 核心创新：KV 压缩、LoRA 投影、推理优化 (PT 2.6.0 L4 重测) | 8 |
| [Paged Attention (中文)](docs/paged_attention/report_cn.md) | vLLM 核心：非连续 KV 存储、内存碎片消除 (PT 2.6.0 L4 重测) | 8 |
| [Paged Attention (English)](docs/paged_attention/report_en.md) | Paged Attention detailed analysis (PT 2.6.0 L4 re-tested) | 8 |

---

## 快速复现

每个报告文件夹都是自包含的，可以直接运行实验：

```bash
# 以 MLA 报告为例
cd flexatten-nv/docs/mla
python mla_experiment.py    # 生成 mla_results.json
python plot_mla.py           # 生成图表到 figures/

# 运行全部实验
python src/run_all_tests.py
```

---

## 实验环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA L4, 24GB VRAM, 121 TFLOPs (FP16) |
| PyTorch | 2.6.0+cu124 |
| Triton | 3.2.0 |
| CUDA | 12.4 |

---

*最后更新：2026-04-27*
