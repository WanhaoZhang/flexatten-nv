# Ascend NPU 矩阵乘法 (mm) 接入 PyTorch 与 CATLASS 后端分析报告

> 分析方法：torch_npu 源码静态分析  
> 分析文件：`torch_npu/_inductor/kernel/mm.py` (272 行)  
> 硬件环境：Ascend 910B3 (×4) | torch_npu  

---

## 目录

1. [概述](#1-概述)
2. [源码文件结构](#2-源码文件结构)
3. [mm 算子接入 PyTorch 的完整链路](#3-mm-算子接入-pytorch-的完整链路)
4. [CATLASS 后端接入机制](#4-catlass-后端接入机制)
5. [Autotuning 策略](#5-autotuning-策略)
6. [与 NVIDIA CUTLASS 对比](#6-与-nvidia-cutlass-对比)
7. [关键发现与结论](#7-关键发现与结论)

---

## 1. 概述

Ascend NPU 的 `torch_npu` 通过 Inductor 的 `register_lowering` 机制覆盖 PyTorch 原生的 `aten.mm` 和 `aten.addmm` 降级，将矩阵乘法路由到 CATLASS（CUTLASS 的 Ascend NPU 版本）后端。

核心设计原则：**在 Inductor 的多候选算法选择框架中注册 CATLASS 作为 mm/addmm 的一个候选**，通过 autotuning 选择最优后端。

---

## 2. 源码文件结构

```
torch_npu/_inductor/
├── kernel/mm.py                          # mm/addmm 降级注册（本报告主体）
├── utils.py                              # use_catlass_template() 判断函数
├── select_algorithm.py                   # NPUTritonTemplate 定义
├── config/catlass.py                     # CATLASS 配置（启用算子、最小 GEMM 尺寸）
└── codegen/catlass/
    ├── catlass_template.py               # CATLASSTemplate 基类（KernelTemplate 子类）
    ├── catlass_kernel.py                 # CATLASSTemplateKernel（C++ 代码生成）
    ├── catlass_python_evg.py             # EVG (Element-wise Vector Graph) 融合
    ├── catlass_utils.py                  # CATLASS 库导入与工具函数
    ├── catlass_scheduling.py             # CATLASS kernel 调度
    ├── gemm_template.py                  # CATLASSGemmTemplate + CATLASS1xGemmTemplate
    └── catlass_library/
        ├── gemm_autotune.py              # GEMM autotune 配置库
        ├── evg_extension.py              # EVG 融合扩展
        └── __init__.py
```

---

## 3. mm 算子接入 PyTorch 的完整链路

### 3.1 注册机制

```python
# torch_npu/_inductor/kernel/mm.py
def _register_npu_inductor_mm():
    @register_lowering(aten.mm, type_promotion_kind=None)
    def tuned_mm(mat1, mat2, *, layout=None):
        ...
```

`_register_npu_inductor_mm()` 在 torch_npu 初始化时被调用，**覆盖** PyTorch 原生的 `aten.mm` 降级函数。这意味着当 `torch.compile` 遇到 `torch.mm()` 时，会进入 NPU 自定义的降级逻辑。

### 3.2 降级流程

```
用户代码: C = torch.mm(A, B)
    ↓
Dynamo 追踪: FX Graph 中的 aten.mm 节点
    ↓
Inductor 降级: 调用 register_lowering 注册的 tuned_mm()
    ↓
tuned_mm() 收集候选后端:
    ├── aten_mm          (ATen 回退，use_aten_gemm_kernels() 时)
    ├── CATLASS1xGemmTemplate (满足条件时)
    ├── CKGemmTemplate    (ROCm 环境，NPU 不走)
    ├── CppGemmTemplate   (CPU 环境，NPU 不走)
    └── external_matmul   (用户自定义外部 kernel)
    ↓
autotune_select_algorithm() 选择最优后端
    ↓
生成 kernel 代码并执行
```

### 3.3 tuned_mm 核心逻辑

```python
def tuned_mm(mat1, mat2, *, layout=None):
    # Step 1: 解析 mm 参数（M, N, K, layout, 输入张量）
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    
    # Step 2: 构建候选列表
    choices = []
    
    # 候选 1: ATen 原生实现
    if use_aten_gemm_kernels():
        choices.append(aten_mm.bind((mat1, mat2), aten_layout))
    
    # 候选 2: CATLASS 模板（关键！）
    is_contiguous_input = (
        is_contiguous_striding(mat1.get_size(), mat1.get_stride())
        and is_contiguous_striding(mat2.get_size(), mat2.get_stride())
    )
    if (is_contiguous_input
        and is_nonzero
        and use_catlass_template("mm", layout, m, n, k)):
        CATLASS1xGemmTemplate.add_catlass_gemm_choices(
            choices, layout, [mat1, mat2]
        )
    
    # Step 3: Autotuning 选择最优
    return autotune_select_algorithm(name, choices, [mat1, mat2], layout)
```

### 3.4 使用 CATLASS 的条件

`use_catlass_template()` 函数定义在 `torch_npu/_inductor/utils.py` 中，需同时满足：

| 条件 | 含义 |
|------|------|
| `op_name.upper() in enabled_ops` | mm 在 CATLASS 启用列表中（默认 "ALL"） |
| `gemm_size >= catlass_backend_min_gemm_size` | GEMM 规模不小于阈值 |
| `not torch.version.hip` | 非 ROCm 环境 |
| 输入连续性检查 | mat1/mat2 的 stride 满足行优先或列优先 |

### 3.5 addmm（带偏置的矩阵乘法）

`_register_npu_inductor_addmm()` 逻辑类似，额外处理 bias 项：
- 调用 `mm_args(mat1, mat2, inp)` 解析 addmm 参数
- 额外检查 bias 的 stride 不为 0（避免不支持的广播）
- 传递 `alpha` 和 `beta` 参数

---

## 4. CATLASS 后端接入机制

### 4.1 CATLASS 模板体系

```
KernelTemplate (PyTorch Inductor)
  └── CATLASSTemplate (catlass_template.py)
        └── CATLASSGemmTemplate (gemm_template.py, 抽象类)
              └── CATLASS1xGemmTemplate (具体实现，1.x 版 CATLASS)
```

### 4.2 代码生成流程

```
CATLASS1xGemmTemplate.add_catlass_gemm_choices()
    ↓
创建 CATLASS1xGemmTemplate 实例
    ↓
调用 _add_catlass_gemm_choices()
    ↓
对每种 GEMM 配置调用 template.generate()
    ↓
CATLASSTemplate.generate():
    ├── 创建 CATLASSTemplateKernel
    ├── 渲染 C++ 模板 (CATLASS_TEMPLATE_1X)
    │     ├── gen_input_template()     → 输入布局描述
    │     ├── gen_kernel_template()    → GEMM kernel 定义
    │     ├── gen_layout_template()    → 数据布局
    │     └── gen_params_device()      → 运行时参数
    ├── 生成 C++ 源码
    └── 返回 CATLASSTemplateCaller
    ↓
CATLASSBenchmarkRequest 在子进程中编译和 benchmark
    ↓
autotune_select_algorithm 选择最优配置
```

### 4.3 生成的 C++ 代码结构

CATLASS 模板生成的 C++ 代码核心结构（`CATLASS_TEMPLATE_1X`）：

```cpp
// CATLASS GEMM kernel 入口
extern "C" PT_EXPORT kernel_call_signature {
    // 1. 获取设备指针
    uint8_t* deviceA = catlass_type_cast(X, kernel.ptr(X));
    uint8_t* deviceB = catlass_type_cast(W, kernel.ptr(W));
    uint8_t* deviceC = catlass_type_cast(Y, kernel.ptr(Y));
    
    // 2. 生成 GEMM kernel 类型
    // op.gen_kernel_template() → GemmKernel 定义
    // op.gen_layout_template() → 数据布局（RowMajor/ColumnMajor）
    using GemmAdapter = Gemm::Device::DeviceGemm<GemmKernel>;
    
    // 3. 获取 AICore 数量
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()
                        ->GetCoreNumAic();
    
    // 4. 初始化和执行
    typename GemmKernel::Arguments arguments{...};
    GemmAdapter gemm_op;
    gemm_op.Initialize(arguments, workspace);
    gemm_op(stream, aicCoreNum);  // 在 NPU AICore 上执行
}
```

### 4.4 与 CUDA GEMM 的关键差异

| 维度 | NVIDIA CUTLASS | Ascend CATLASS |
|------|---------------|----------------|
| 编程模型 | SIMT (Thread/Block/Grid) | AICore 矩阵单元 |
| Kernel 入口 | `__global__ void kernel()` | `extern "C" PT_EXPORT kernel()` |
| 并行单元 | CUDA Core / Tensor Core | AICore (Cube/Vector) |
| 数据加载 | Global Memory → Shared Memory → Register | DDR → L1 Buffer → UB |
| 矩阵乘 | `wmma::mma_sync()` / `wgmma` | Cube 单元自动执行 |
| Epilogue | 写回 Global Memory | EVG 融合 (Element-wise Vector Graph) |
| 核心数获取 | `blockIdx`, `threadIdx` | `GetCoreNumAic()` |

---

## 5. Autotuning 策略

### 5.1 CATLASS GEMM 配置选择

CATLASS 使用预编译的 C++ 模板库，每种配置对应不同的：
- **Tile 大小**：矩阵分块的 (M_tile, N_tile, K_tile)
- **流水线级数**：数据预取的深度
- **Epilogue 融合**：是否融合 ReLU/EVG 操作

### 5.2 回退策略

```python
# 如果 CATLASS 不可用，回退到 ATen
if len(choices) == 0 and inductor_config.autotune_fallback_to_aten:
    log.warning("No choices for GEMM, using ATen backend as fallback")
    return aten_mm.bind((mat1, mat2), aten_layout).output_node()
```

这确保了即使 CATLASS 不支持某些情况（非连续输入、极小 GEMM），mm 仍能正确执行。

---

## 6. 与 NVIDIA CUTLASS 对比

### 6.1 架构对比

```
NVIDIA:
  aten.mm → Inductor lowering → [TritonTemplate | CUTLASS | ATen] → PTX → GPU

Ascend NPU:
  aten.mm → Inductor lowering → [CATLASS | ATen] → C++ 编译 → NPU AICore
```

### 6.2 关键设计差异

| 维度 | NVIDIA | Ascend NPU |
|------|--------|------------|
| **JIT 语言** | Triton (Python→PTX) 或 CUTLASS (C++→PTX) | CATLASS (C++ 模板→二进制) |
| **动态编译** | Triton JIT 运行时编译 | 预编译 C++ 模板 + 子进程 benchmark |
| **融合能力** | Triton Pointwise 融合 | EVG (Element-wise Vector Graph) 融合 |
| **Autotuning** | Triton autotune (BLOCK_M/N, warps, stages) | GEMM autotune (tile sizes, pipeline depth) |
| **代码生成** | Python 模板 → Triton IR | Jinja2 → C++ 源码 → C++ 编译器 |
| **共享内存** | CUDA Shared Memory | L1 Buffer / Unified Buffer |

### 6.3 CATLASS 没有 CuteDSL 对应物

NVIDIA 有 CuteDSL 作为 Python AST 到 CUTLASS CuTe 的桥梁，Ascend NPU **没有类似物**：
- mm/addmm 直接使用预编译的 C++ CATLASS 模板
- 没有 Python-level 的 DSL 动态生成 kernel
- 这意味着 CATLASS 的灵活性不如 CuteDSL，但编译时间更短（无需运行时 JIT）

---

## 7. 关键发现与结论

### 7.1 mm 接入 PyTorch 的核心机制

1. **`register_lowering` 覆盖**：torch_npu 覆盖了 PyTorch 原生的 `aten.mm` 降级，加入 CATLASS 作为候选后端
2. **Autotuning 竞争**：CATLASS 不是唯一的后端，它与 ATen 回退竞争，由 autotune 选择最优
3. **条件性启用**：只有满足连续输入、足够大的 GEMM 尺寸、且在启用列表中时才使用 CATLASS

### 7.2 CATLASS 与 CUTLASS 的定位差异

| 方面 | CUTLASS (NVIDIA) | CATLASS (Ascend) |
|------|------------------|-------------------|
| 定位 | GPU 通用矩阵计算库 | NPU 专用矩阵计算库 |
| 在 PyTorch 中的角色 | SDPA 后端、CuteDSL FlexAttention 后端 | mm/addmm 后端 |
| 接入方式 | 预编译 + CuteDSL JIT | 预编译 C++ 模板 |
| FlexAttention 支持 | CuteDSL 路径 (PT 2.7+) | **不支持**（FlexAttention 在 NPU 上走 Triton） |

### 7.3 为什么 FlexAttention 不使用 CATLASS

FlexAttention 需要将用户的 Python `score_mod`/`mask_mod` 函数 inline 到 kernel 中。CATLASS 的 C++ 模板机制无法实现这种动态 inline——它只支持预定义的 GEMM 模式。因此 NPU 上的 FlexAttention 只能走 Triton 路径（`NPUTritonTemplate`），详见另一份报告。

---

*分析日期：2026-04-27 | 源码版本：torch_npu (Ascend 910B3) | 源码行数：mm.py 272 行 + gemm_template.py 907 行*
