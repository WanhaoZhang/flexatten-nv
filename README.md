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
├── trace/                 # GPU 执行管线 & CuteDSL 分析
│   ├── FLEXATTENTION_GPU_PIPELINE_ANALYSIS.md  # 全新管线静态分析报告
│   ├── analyze_flex_pipeline.py               # 分析图表生成脚本
│   ├── figures/           # 分析图表 (6张)
│   ├── CAUSAL_FLEXATTENTION_CUTEDSL_TRACE.md  # 原始 trace 报告
│   ├── FLEXATTENTION_FLASH_CUTEDSL_BACKEND_REPORT.md  # FLASH 后端复测报告
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

### GPU 编译管线与后端分析

| 报告 | 内容 | 图表 |
|------|------|------|
| [FlexAttention GPU 执行管线静态分析](docs/trace/FLEXATTENTION_GPU_PIPELINE_ANALYSIS.md) | 6层编译架构全景、Triton/CuteDSL双路径源码分析、Autotuning配置、BlockMask BCSR结构、BLOCK_SIZE影响、L4(sm89)限制分析 | 6 |
| [Causal FlexAttention GPU链路追踪](docs/trace/CAUSAL_FLEXATTENTION_CUTEDSL_TRACE.md) | 最小causal实例全链路trace：Dynamo→HOP→Inductor→TritonTemplate→Triton kernel，CuteDSL/FLASH后端探测结果 | - |
| [FlexAttention FLASH/CuteDSL 后端复测](docs/trace/FLEXATTENTION_FLASH_CUTEDSL_BACKEND_REPORT.md) | PT nightly + nvidia-cutlass-dsl + FA4 Cute环境搭建，确认CuteDSL代码生成路径正确，sm89架构限制验证 | - |

### Ascend NPU 后端接入分析

| 报告 | 内容 | 图表 |
|------|------|------|
| [NPU mm/CATLASS 接入分析](docs/trace/NPU_MM_CATLASS_ANALYSIS.md) | mm/addmm 通过 register_lowering 接入 CATLASS C++ 模板，Autotuning 策略，与 NVIDIA CUTLASS 对比 | - |
| [NPU FlexAttention/Triton 接入分析](docs/trace/NPU_FLEXATTENTION_TRITON_ANALYSIS.md) | FlexAttention 通过 NPUTritonTemplate 接入 Triton，K/V 转置修复，load_checked_2d 替代 make_block_ptr，NPU autotuning 配置（910B3），与 NVIDIA Triton 路径差异分析 | - |

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
