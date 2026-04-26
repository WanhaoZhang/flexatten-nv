# FlexAttention 实验项目

> 基于 [attention-gym](https://github.com/meta-pytorch/attention-gym) | NVIDIA L4 (24GB) | PyTorch 2.6.0+cu124 | Triton 3.2.0

---

## 报告目录

### 综合分析报告

| 报告 | 内容 | 图表数 |
|------|------|--------|
| [README.md](README.md) | 基础知识与机制深度解析（5组实验） | 5 |
| [FLEXATTENTION_DEEP_DIVE.md](FLEXATTENTION_DEEP_DIVE.md) | FlexAttention 深度原理剖析：痛点→解法（8组实验） | 8 |
| [DOC_PACKING_ATTENTION.md](DOC_PACKING_ATTENTION.md) | Document Packing 专题：三种实现全解析 | 6 |
| [PATTERN_ANALYSIS_REPORT.md](PATTERN_ANALYSIS_REPORT.md) | 8种注意力模式全解析：Vanilla vs FlexAttention | 8 |
| [SOURCE_ANALYSIS_REPORT.md](SOURCE_ANALYSIS_REPORT.md) | 源码级分析：Vanilla→SDPA→FlexAttention 三条路径 | 8 |

### 单一模式深度报告

| 报告 | 模式 | 核心特点 |
|------|------|---------|
| [SLIDING_WINDOW_REPORT.md](SLIDING_WINDOW_REPORT.md) | Sliding Window | O(S×W) 复杂度，稀疏率 88-97% |
| [ALIBI_REPORT.md](ALIBI_REPORT.md) | ALiBi | score_mod 经典应用，消除 for 循环 |
| [PREFIX_LM_REPORT.md](PREFIX_LM_REPORT.md) | Prefix LM | mask OR 组合，前缀双向+后缀因果 |
| [SOFTCAP_REPORT.md](SOFTCAP_REPORT.md) | Tanh Softcapping | 数值稳定性，score_mod 防溢出 |

---

## 实验脚本

| 脚本 | 实验 | 运行时间 |
|------|------|---------|
| `flexatten_experiments.py` | 实验 1-5：基础对比 | ~10 min |
| `flexatten_fix.py` | 实验 2/4 修复 | ~5 min |
| `flexatten_deep_dive.py` | 实验 A1-C3：深度原理 | ~20 min |
| `doc_packing_experiments.py` | 实验 Exp1-6：Document Packing | ~15 min |
| `flexatten_pattern_analysis.py` | 实验 E1-E8：8种模式全对比 | ~30 min |
| `flexatten_source_analysis.py` | 实验 F1-F5：源码分析 | ~15 min |
| `flexatten_source_fix.py` | 实验 F6-F8：补充分析 | ~10 min |

## 图表目录

| 目录 | 内容 | 图表数 |
|------|------|--------|
| `figures/` | 深度原理报告图表 (A1-C3) | 8 |
| `figures_doc_packing/` | Document Packing 报告图表 | 6 |
| `figures_patterns/` | 8种模式对比图表 (E1-E8) | 8 |
| `figures_source/` | 源码分析图表 (F1-F8) | 8 |

---

## 环境信息

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA L4, 24GB VRAM, 121 TFLOPs (FP16) |
| PyTorch | 2.6.0+cu124 |
| Triton | 3.2.0 |
| CUDA | 12.4 |

## 快速复现

```bash
# SSH 到 GCP 服务器
gcloud compute ssh zhangwh@instance-20260405-l4gpu --zone=us-central1-a --tunnel-through-iap

# 激活环境
conda activate flexatten

# 运行所有实验
cd ~/flexatten-nv
python flexatten_pattern_analysis.py   # 8种模式对比 (~30 min)
python flexatten_source_fix.py         # 源码分析 (~10 min)
```

---

*最后更新：2026-04-25*
