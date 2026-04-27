# Ascend NPU FlexAttention 接入 PyTorch 与 Triton 后端分析报告

> 分析方法：torch_npu 源码静态分析  
> 分析文件：`torch_npu/_inductor/kernel/flex_attention.py` (1874 行)  
> 对比参考：`torch/_inductor/kernel/flex_attention.py` (NVIDIA, 2379 行)  
> 硬件环境：Ascend 910B3 (×4)  

---

## 目录

1. [概述](#1-概述)
2. [源码文件对比](#2-源码文件对比)
3. [NPU FlexAttention 完整接入链路](#3-npu-flexattention-完整接入链路)
4. [NPUTritonTemplate：核心适配层](#4-nputritontemplate核心适配层)
5. [Forward Kernel 分析](#5-forward-kernel-分析)
6. [Backward Kernel 分析](#6-backward-kernel-分析)
7. [Autotuning 配置](#7-autotuning-配置)
8. [与 NVIDIA Triton 路径的差异](#8-与-nvidia-triton-路径的差异)
9. [NPU 没有 CuteDSL 路径](#9-npu-没有-cutedsl-路径)
10. [关键发现与结论](#10-关键发现与结论)

---

## 1. 概述

Ascend NPU 的 FlexAttention 实现**完全基于 Triton 后端**，没有 CATLASS/CuteDSL 路径。torch_npu 通过继承 PyTorch 原生的 FlexAttention Triton 模板，替换为 NPU 特定的 `NPUTritonTemplate`，并调整 autotuning 配置以适配 Ascend 910B3 的硬件特征。

核心设计原则：**最大化复用 PyTorch 上游代码，仅在 Inductor 降级层做 NPU 适配**。

---

## 2. 源码文件对比

### 2.1 导入对比

```python
# NVIDIA (torch/_inductor/kernel/flex_attention.py)
from torch._inductor.select_algorithm import TritonTemplate  # 标准 Triton 模板

# NPU (torch_npu/_inductor/kernel/flex_attention.py)
from torch_npu._inductor.select_algorithm import NPUTritonTemplate  # NPU 专用 Triton 模板
```

NPU 版本从 PyTorch 上游导入了大量工具函数，复用了所有通用逻辑：

```python
from torch._inductor.kernel.flex_attention import (
    Mode, lower_cpu, maybe_realize, is_power_of_2, next_power_of_two,
    construct_strides, create_placeholder, set_head_dim_values,
    create_indices_fake, flex_attention_grid, infer_dense_strides,
    validate_joint_graph, process_joint_outputs, build_subgraph_buffer,
    flex_attention_backward_grid, create_num_blocks_fake_generator
)
```

### 2.2 代码复用率

| 组件 | NVIDIA 行数 | NPU 行数 | 复用/重写 |
|------|-----------|---------|----------|
| Triton 辅助函数 (get_offset, load_checked 等) | ~150 行 | ~150 行 | **完全重写**（修复 K/V 转置） |
| Forward kernel 模板 | ~200 行 | ~200 行 | **完全重写**（修复 K/V 转置） |
| Backward kernel 模板 | ~500 行 | ~500 行 | **完全重写**（修复 K/V 转置） |
| Inductor 降级逻辑 | ~500 行 | ~600 行 | **重写注册函数** |
| Autotuning 配置 | ~150 行 | ~30 行 | **NPU 自定义配置** |
| 共享工具函数 | - | 导入 | **完全复用** |

---

## 3. NPU FlexAttention 完整接入链路

```
用户代码:
  flex_attention(query, key, value, block_mask=mask)
      ↓
Dynamo 追踪 (共用 PyTorch 上游):
  FlexAttentionHigherOrderVariable → FX Graph
      ↓
HOP 分发 (共用 PyTorch 上游):
  FlexAttentionHOP → proxy tracing
      ↓
Inductor 降级 (NPU 覆盖):
  torch_npu/_inductor/kernel/flex_attention.py:
  _register_npu_inductor_flex_attention()
  @register_lowering(torch.ops.higher_order.flex_attention)
      ↓
Triton 模板渲染 (NPU 使用 NPUTritonTemplate):
  NPUTritonTemplate.generate() → Triton kernel 源码
      ↓
Triton JIT 编译 → NPU 执行
```

### 3.1 注册入口

```python
def _register_npu_inductor_flex_attention():
    @register_lowering(torch.ops.higher_order.flex_attention, type_promotion_kind=None)
    def flex_attention(query, key, value, subgraph, block_mask, scale,
                       kernel_options, score_mod_other_buffers, mask_mod_other_buffers):
        # NPU 路径
        ...
```

这个注册在 torch_npu 初始化时执行，覆盖 PyTorch 原生的 flex_attention 降级。

---

## 4. NPUTritonTemplate：核心适配层

### 4.1 继承关系

```
KernelTemplate (PyTorch)
  └── TritonTemplate (PyTorch)
        └── NPUTritonTemplate (torch_npu)
```

### 4.2 NPUTritonTemplate 的关键修改

```python
class NPUTritonTemplate(TritonTemplate):
    def __init__(self, name, grid, source, debug=False):
        super().__init__(name, grid, source, debug)

    def generate(self, input_nodes, layout, ...):
        # 与 TritonTemplate 几乎相同
        # 关键差异:
        # 1. 使用 NPUTritonTemplateKernel 替代 TritonTemplateKernel
        # 2. ALLOW_TF32 = False (NPU 不支持 TF32)
        # 3. use_jit=False (NPU Triton 不使用 JIT 模式)
        ...
```

### 4.3 模板实例化

```python
# NPU 的 flex_attention 模板
# 注意：使用 NPUTritonTemplate 而非 TritonTemplate
# （实际上 flex_attention_template 变量不在 npu 文件中定义，
#   而是通过模板源码字符串 compute_flex_attention 传入）

flex_attention_backward_template = NPUTritonTemplate(
    name="flex_attention_backward",
    grid=flex_attention_backward_grid,
    source=compute_flex_attention_backward + ...,
)
```

---

## 5. Forward Kernel 分析

### 5.1 与 NVIDIA 版本的核心差异：K/V 转置

NPU 版本中散布着 `# pta` 标记的修改，主要是 K/V 加载方式的调整：

```python
# NVIDIA (原始):
Q_block_ptr = tl.make_block_ptr(
    base=Q,
    offsets=(q_start * BLOCK_M, 0),
    block_shape=(BLOCK_M, QK_HEAD_DIM),   # Q: [BLOCK_M, D]
    ...
)
K_block_ptr = tl.make_block_ptr(
    base=K,
    offsets=(kv_start, 0),
    block_shape=(QK_HEAD_DIM, BLOCK_N),   # K: [D, BLOCK_N] ← 转置加载
    ...
)

# NPU (修改后):
# Q 加载方式不变
# K 加载方式调整（通过手动 offset 计算）
# 在 backward kernel 中可见明确的 # pta 标记:

# pytorch (原始):
kT_ptrs = K + offs_n2[None, :] * stride_kn + offs_k[:, None] * stride_kd
# pta (NPU 修改):
kT_ptrs = K + offs_n2[:, None] * stride_kn + offs_k[None, :] * stride_kd
# ===

# 然后：
kT = load_checked_2d(kT_ptrs, offs_k, offs_n2, ...)
# pta
kT = tl.trans(kT)  # NPU 显式转置
# ===
```

### 5.2 转置差异的含义

NVIDIA 的 Triton 后端可以直接用 `tl.make_block_ptr` 加载转置的 K 矩阵（通过 stride 交换）。NPU 的 Triton 实现不支持这种模式，因此需要：
1. 按原始顺序加载 K
2. 用 `tl.trans()` 显式转置

这对性能有一定影响（额外的转置操作），但保证了正确性。

### 5.3 Triton 辅助函数

NPU 版本自定义了以下 Triton 辅助函数（同样在 NVIDIA 版本中存在但实现略有不同）：

| 函数 | 用途 |
|------|------|
| `get_offset_for_next_block` | 稀疏块间接跳转 |
| `get_bounded_indices` | 序列长度边界处理 |
| `load_checked_block` | 带 boundary check 的块加载 |
| `load_checked_2d` | 带 mask 的 2D 加载（NPU 新增，替代 `tl.make_block_ptr`） |

`load_checked_2d` 是 NPU 特有的辅助函数，用 `tl.load` + mask 替代 `tl.make_block_ptr`：

```python
@triton.jit
def load_checked_2d(ptr, offs_m, offs_n, stride_m, stride_n,
                    IS_DIVISIBLE_M, IS_DIVISIBLE_N, M_LEN, N_DIM):
    # 手动计算指针偏移
    if stride_m is not None and stride_n is not None:
        ptr = ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    # 根据 divisibility 选择 mask 策略
    if not IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN) & (offs_n[None, :] < N_DIM), other=0.0)
    ...
```

---

## 6. Backward Kernel 分析

### 6.1 Backward 结构

与 NVIDIA 版本一致，backward kernel 分为两个阶段：

```
Phase 1: 计算 dK, dV（遍历 Q blocks）
  bwd_dkdv_inner → bwd_dkdv_block_mn
  加载 Q, DO → 计算 attention weights → 累加 dK, dV

Phase 2: 计算 dQ（遍历 KV blocks）
  bwd_dq_inner → bwd_dq_block_mn
  加载 K, V → 计算 attention weights → 累加 dQ
```

### 6.2 与 Forward 相同的转置修复

Backward kernel 中同样存在 `# pta` 标记的转置修复：

```python
# Phase 1 (dK/dV):
qT_ptrs = Q + offs_m1[:, None] * stride_qm + offs_k[None, :] * stride_qd  # pta
qT = load_checked_2d(qT_ptrs, offs_k, offs_m1, ...)
qT = tl.trans(qT)  # pta: 显式转置

# Phase 2 (dQ):
kT_ptrs = K + offs_n2[:, None] * stride_kn + offs_k[None, :] * stride_kd  # pta
kT = load_checked_2d(kT_ptrs, offs_k, offs_n2, ...)
kT = tl.trans(kT)  # pta: 显式转置
```

### 6.3 score_mod 和 mask_mod 的 inline

与 NVIDIA 完全一致的 `{{ modification() }}` Jinja2 模板调用：

```python
# Forward:
{{ modification(subgraph_number=0, output_name="post_mod_scores", ...) }}  # score_mod
{{ modification(subgraph_number=2, output_name="mask_mod_output", ...) }}  # mask_mod

# Backward:
{{ modification(subgraph_number=0, ...) }}   # forward score_mod (recompute)
{{ modification(subgraph_number=1, ..., grad_score_mod="ds") }}  # joint backward score_mod
{{ modification(subgraph_number=2, ...) }}   # mask_mod
```

---

## 7. Autotuning 配置

### 7.1 NPU 专用配置函数

```python
def _get_npu_config(query, mode: Mode) -> tuple[int, int, int, int]:
    dtype = query.get_dtype()
    head_dim = query.get_size()[-1]
    
    if mode == Mode.fwd:
        if head_dim <= 256:
            return (64, 64, 4, 3)  # BLOCK_M=64, BLOCK_N=64, warps=4, stages=3
        else:
            if dtype == torch.float32:
                return (32, 16, 4, 3)
            else:
                return (32, 32, 4, 3)
    else:  # bwd
        return (16, 16, 4, 1)  # 非常保守的 backward 配置
```

### 7.2 与 NVIDIA 配置对比

| 配置 | H100 | A100 | L4 | **910B3 (NPU)** |
|------|------|------|----|----|
| **Forward BLOCK_M** | 128 | 128 | 128 | **64** |
| **Forward BLOCK_N** | 128 | 64 | 64 | **64** |
| **Forward num_warps** | 8 | 8 | 4 | **4** |
| **Forward num_stages** | 3 | 3 | 3 | **3** |
| **Backward BLOCK** | 64 | 64 | 64 | **16** |
| **Backward num_stages** | 3 | 3 | 3 | **1** |

**关键观察**：

1. **BLOCK_M/N 更小**：910B3 的 forward 使用 64×64（NVIDIA 默认 128×64），backward 使用 16×16（NVIDIA 使用 64×64）。这反映了 NPU AICore 的内存层次结构与 GPU 的差异
2. **Backward 非常保守**：num_stages=1 意味着没有共享内存流水线，这可能是 NPU Triton 的实现限制
3. **没有 GPU 架构分支**：NPU 不需要像 NVIDIA 那样区分 H100/A100/sm75

### 7.3 Max Autotune 模式

```python
if config.max_autotune:
    configs += [
        (128, 64, 4, 3),
        (128, 128, 4, 3),
        (128, 128, 8, 2),
        (64, 128, 4, 3),
        (64, 64, 4, 3),
    ]
```

与 NVIDIA 完全一致——在 max_autotune 模式下搜索更大的配置空间。

---

## 8. 与 NVIDIA Triton 路径的差异

### 8.1 整体架构对比

```
NVIDIA:
  Python API → Dynamo → HOP → Inductor → TritonTemplate → Triton JIT → PTX → GPU SM

NPU:
  Python API → Dynamo → HOP → Inductor → NPUTritonTemplate → Triton JIT → MLIR → NPU AICore
```

### 8.2 逐层对比

| 层 | NVIDIA | NPU | 差异 |
|----|--------|-----|------|
| User API | `flex_attention()` | 相同 | 无差异 |
| Dynamo Trace | `FlexAttentionHigherOrderVariable` | 相同 | 无差异 |
| HOP Dispatch | `FlexAttentionHOP` | 相同 | 无差异 |
| Inductor Lowering | `register_lowering` in `kernel/flex_attention.py` | `register_lowering` in `torch_npu/kernel/flex_attention.py` | **NPU 覆盖注册** |
| Template | `TritonTemplate` | `NPUTritonTemplate` | **NPU 子类** |
| Kernel 生成 | `tl.make_block_ptr` | `load_checked_2d` + `tl.trans` | **NPU 不支持 block_ptr** |
| JIT 编译 | Triton → LLVM → PTX | Triton → MLIR → AICore binary | **不同后端编译器** |
| TF32 | 允许 | `ALLOW_TF32 = False` | **NPU 不支持 TF32** |

### 8.3 Triton 后端差异

| 方面 | NVIDIA Triton | NPU Triton |
|------|--------------|------------|
| `tl.make_block_ptr` | 支持 | **不支持** → 用 `load_checked_2d` 替代 |
| `tl.dot` | 自动映射到 Tensor Core | 映射到 Cube 单元 |
| `tl.trans` | 可能被优化掉 | **必须显式调用** |
| 共享内存 | CUDA Shared Memory | L1 Buffer |
| 流水线 | num_stages > 1 支持 | num_stages = 1（backward 限制） |
| Subprocess autotune | 支持 | 支持（通过 `CATLASSBenchmarkRequest` 或 `TritonBenchmarkRequest`） |

---

## 9. NPU 没有 CuteDSL 路径

### 9.1 为什么没有

NVIDIA 的 CuteDSL 路径需要：
1. **CuteDSL**（NVIDIA 独有的 Python AST → CUTLASS CuTe 编译层）
2. **flash_attn.cute**（FlashAttention-4 的 Cute 接口）
3. **CUTLASS 3.x**（NVIDIA 独有的 GPU 矩阵计算库）

这三者都是 NVIDIA 专有的，Ascend NPU 没有：
- **没有 CATLASS DSL**：CATLASS 是 C++ 模板库，没有 Python DSL 层
- **没有 FlashAttention-4 Cute**：Ascend 没有对应的 flash attention Cute 接口
- **没有 CuTe 抽象**：NPU 的 AICore 编程模型与 GPU 的 SIMT 模型完全不同

### 9.2 后果

NPU 上的 FlexAttention **只能走 Triton 路径**：
- 性能可能不如 GPU 上的 CuteDSL 路径（如果可用的话）
- 但 Triton 路径功能完整：支持 score_mod、mask_mod、backward、GQA、decode kernel
- 随着 NPU Triton 的持续优化，性能差距可能缩小

---

## 10. 关键发现与结论

### 10.1 NPU FlexAttention 接入的核心设计

1. **最大化代码复用**：从 PyTorch 上游导入所有可复用的工具函数（`build_subgraph_buffer`、`create_placeholder`、`flex_attention_grid` 等），仅重写 NPU 特有的部分

2. **NPUTritonTemplate 是关键适配层**：继承 `TritonTemplate`，仅覆盖 `generate()` 方法中的 kernel 实例化和 TF32 设置

3. **手动 K/V 转置修复**：由于 NPU Triton 不支持 `tl.make_block_ptr` 的转置加载，所有 K/V 加载改为 `load_checked_2d` + `tl.trans()`

### 10.2 mm vs FlexAttention 的后端选择逻辑对比

```
mm 算子:
  [CATLASS C++ 模板] ← 预编译，高性能
  [ATen 回退]         ← 兜底

FlexAttention:
  [NPU Triton]       ← 唯一后端，JIT 编译
  (无 CATLASS 选项)   ← CATLASS 无法 inline score_mod/mask_mod
```

### 10.3 Autotuning 差异反映了硬件特征

910B3 的 AICore 内存层次（DDR → L1 → UB）与 GPU（HBM → Shared Memory → Register）不同：
- 更小的 BLOCK_M/N 反映了 L1 Buffer 的容量限制
- backward 的 num_stages=1 反映了 NPU Triton 的流水线实现不完善
- 保守的配置确保正确性，后续可以逐步优化

### 10.4 与 GPU 管线对比总结

| 维度 | NVIDIA GPU | Ascend NPU |
|------|-----------|------------|
| **mm 后端** | CUTLASS / Triton / ATen | CATLASS / ATen |
| **FlexAttention 后端** | Triton / CuteDSL | **仅 Triton** |
| **JIT 编译** | Triton → LLVM → PTX | Triton → MLIR → AICore |
| **代码复用** | 上游 PyTorch | 上游 + NPU 适配层 |
| **动态 kernel 生成** | Triton + CuteDSL | 仅 Triton |
| **预编译模板** | CUTLASS (C++) | CATLASS (C++) |

---

*分析日期：2026-04-27 | 源码版本：torch_npu (Ascend 910B3) | 源码行数：flex_attention.py 1874 行*
