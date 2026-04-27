# FlexAttention 实验项目

> 基于 [attention-gym](https://github.com/pytorch/attention-gym) | NVIDIA L4 (24GB) | PyTorch 2.6.0+cu124 | Triton 3.2.0

FlexAttention 是 PyTorch 2.5+ 引入的灵活注意力机制 API，允许用纯 Python 函数描述注意力修改（mask_mod / score_mod），框架自动编译为 Triton kernel。本仓库通过系统化实验，全面剖析其原理、性能和适用场景。

---

## 报告目录

### 综合分析报告

| 报告 | 内容 | 图表 |
|------|------|------|
| [FlexAttention 原理深度剖析](docs/FLEX_INTERNALS_REPORT_CN.md) | 8组实验：BlockMask结构、score_mod编译、稀疏性vs性能、编译开销、延迟对比、逐步追踪 | 6 |
| [FlexAttention 深度原理剖析](docs/FLEXATTENTION_DEEP_DIVE.md) | 痛点→解法：内存爆炸、带宽饥饿、工程噩梦 | 8 |
| [Document Packing 专题](docs/DOC_PACKING_ATTENTION.md) | 三种实现全解析：Vanilla / SDPA / FlexAttention | 5 |
| [8种注意力模式全解析](docs/PATTERN_ANALYSIS_REPORT.md) | Vanilla vs FlexAttention：Causal/SW/Prefix/ALiBi/Softcap/Dilated | 9 |
| [源码级分析](docs/SOURCE_ANALYSIS_REPORT.md) | Vanilla → SDPA → FlexAttention 三条执行路径 | 7 |
| [Causal FlexAttention GPU链路与CuteDSL接入分析](docs/CAUSAL_FLEXATTENTION_CUTEDSL_TRACE.md) | 最小 causal attention 实现、Dynamo/HOP/Inductor/Triton trace、CuteDSL/CUTLASS 接入点分析 | - |

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
| [Multi-Head Latent Attention (MLA)](docs/MLA_REPORT_CN.md) | DeepSeek-V2 核心创新：KV 压缩、LoRA 投影、推理优化 | 8 |
| [Paged Attention](docs/PAGED_ATTENTION_REPORT_CN.md) | vLLM 核心：非连续 KV 存储、内存碎片消除 | 8 |
| [Paged Attention (English)](docs/PAGED_ATTENTION_REPORT.md) | Paged Attention 详细分析 | 8 |

---

## 实验代码与数据归档

### 实验脚本

| 脚本 | 对应报告 | 运行时间 |
|------|---------|---------|
| [`src/flex_internals_experiment.py`](src/flex_internals_experiment.py) | [FlexAttention 原理深度剖析](docs/FLEX_INTERNALS_REPORT_CN.md) | ~15 min |
| [`src/plot_flex_internals.py`](src/plot_flex_internals.py) | 上述报告的绘图脚本 | <1 min |
| [`src/flexatten_experiments.py`](src/flexatten_experiments.py) | 基础对比实验（5组） | ~10 min |
| [`src/flexatten_deep_dive.py`](src/flexatten_deep_dive.py) | [深度原理剖析](docs/FLEXATTENTION_DEEP_DIVE.md) | ~20 min |
| [`src/flexatten_pattern_analysis.py`](src/flexatten_pattern_analysis.py) | [8种模式全解析](docs/PATTERN_ANALYSIS_REPORT.md) | ~30 min |
| [`src/flexatten_source_analysis.py`](src/flexatten_source_analysis.py) | [源码级分析](docs/SOURCE_ANALYSIS_REPORT.md) | ~15 min |
| [`src/flexatten_source_fix.py`](src/flexatten_source_fix.py) | 补充分析（F6-F8） | ~10 min |
| [`src/doc_packing_experiments.py`](src/doc_packing_experiments.py) | [Document Packing](docs/DOC_PACKING_ATTENTION.md) | ~15 min |
| [`src/mla_experiment.py`](src/mla_experiment.py) | [MLA 报告](docs/MLA_REPORT_CN.md) | ~20 min |
| [`src/plot_mla.py`](src/plot_mla.py) | MLA 绘图脚本 | <1 min |
| [`src/paged_attention_experiment.py`](src/paged_attention_experiment.py) | [Paged Attention](docs/PAGED_ATTENTION_REPORT_CN.md) | ~20 min |
| [`src/plot_paged_attention.py`](src/plot_paged_attention.py) | Paged Attention 绘图脚本 | <1 min |
| [`src/run_all_tests.py`](src/run_all_tests.py) | 一键运行全部实验 | ~2 hr |

### 实验数据

所有实验的原始结果以 JSON 格式归档在 `data/` 目录：

| 数据文件 | 对应报告 |
|---------|---------|
| [`data/flex_internals_results.json`](data/flex_internals_results.json) | [FlexAttention 原理深度剖析](docs/FLEX_INTERNALS_REPORT_CN.md) |
| [`data/deep_dive_results.json`](data/deep_dive_results.json) | [深度原理剖析](docs/FLEXATTENTION_DEEP_DIVE.md) |
| [`data/pattern_analysis_results.json`](data/pattern_analysis_results.json) | [8种模式全解析](docs/PATTERN_ANALYSIS_REPORT.md) |
| [`data/source_analysis_results.json`](data/source_analysis_results.json) | [源码级分析](docs/SOURCE_ANALYSIS_REPORT.md) |
| [`data/doc_packing_results.json`](data/doc_packing_results.json) | [Document Packing](docs/DOC_PACKING_ATTENTION.md) |
| [`data/mla_results.json`](data/mla_results.json) | [MLA 报告](docs/MLA_REPORT_CN.md) |
| [`data/paged_attention_results.json`](data/paged_attention_results.json) | [Paged Attention](docs/PAGED_ATTENTION_REPORT_CN.md) |

### 图表文件

所有图表统一存放在 [`docs/figures/`](docs/figures/) 目录，共 50+ 张 PNG。

---

## 实验环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA L4, 24GB VRAM, 121 TFLOPs (FP16) |
| PyTorch | 2.6.0+cu124 |
| Triton | 3.2.0 |
| CUDA | 12.4 |

## 快速复现

```bash
# 激活环境
conda activate flexatten

# 运行实验（以原理剖析为例）
cd flexatten-nv
python src/flex_internals_experiment.py    # 生成数据到 data/
python src/plot_flex_internals.py          # 生成图表到 docs/figures/

# 一键运行全部
python src/run_all_tests.py
```

---

*最后更新：2026-04-27*
