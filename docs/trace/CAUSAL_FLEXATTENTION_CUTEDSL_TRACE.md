# Causal FlexAttention 到 GPU Kernel 的执行链路分析

> 实验位置：`causal_attention_trace/causal_flexattention_trace.py`  
> 运行产物：`causal_attention_trace/artifacts/`  
> 服务器：`instance-20260405-l4gpu`，NVIDIA L4，PyTorch 2.6.0+cu124，Triton 3.2.0

## 结论先行

本次新增了一个最小 causal attention 实现：用 `mask_mod(batch, head, q_idx, kv_idx): return q_idx >= kv_idx` 描述因果 mask，用 `create_block_mask` 转成 FlexAttention 的块稀疏元数据，再用 `torch.compile(flex_attention, fullgraph=True)` 触发 GPU 编译执行。

在当前 `flexatten` conda 环境中，真实跑通的链路是：

```text
Python causal_mask_mod
  -> create_block_mask 生成 BlockMask
  -> torch.compile(flex_attention)
  -> torch.ops.higher_order.flex_attention
  -> Inductor lowering: torch/_inductor/kernel/flex/flex_attention.py
  -> TritonTemplate: flex_attention.py.jinja
  -> 生成 triton_flex_attention / triton_tem_fused_0
  -> NVIDIA L4 CUDA kernel
```

数值验证通过：脚本输出 `[1, 2, 128, 64]` 的 `float16` attention 结果，对 dense PyTorch causal reference 的最大绝对误差是 `0.0009765625`，warm run 约 `0.00021s`。

关于用户关心的 CUTLASS/CuteDSL：`/home/zhangwh/pytorch` 源码已经有 `BACKEND="FLASH"` 接入，源码路径会进入 `torch/_inductor/kernel/flex/flex_flash_attention.py`，该文件用 `CuteDSLTemplate` 生成 `flash_attention_cutedsl`，再由 `torch/_inductor/codegen/cutedsl` import `cutlass` / `cutlass.cute`。但是当前可运行的 conda 环境是 PyTorch 2.6.0 wheel，它的 `flex_attention` Python 源码还没有 `BACKEND` 选择；`flash-attention` 源码虽在 `/home/zhangwh/flash-attention`，但没有编译出 `flash_attn_2_cuda`；`cutlass.cute` 也不可 import。因此本机当前不能真实执行到 CUTLASS/CuteDSL，只能真实执行到 Triton，并基于下载好的 PyTorch 源码分析 CuteDSL 接入点。

## 新增测试文件

`causal_attention_trace/causal_flexattention_trace.py` 做了四件事：

1. 构造 `q/k/v`: `B=1, H=2, S=128, D=64, dtype=float16, device=cuda`。
2. 定义 causal `mask_mod`：`q_idx >= kv_idx`。
3. 通过 `create_block_mask` 生成 `BlockMask`，再调用 `torch.compile(flex_attention, fullgraph=True)`。
4. 和 dense causal attention reference 比较，并保存环境、trace、Inductor cache 命中线索、FLASH/CuteDSL backend 探测结果。

运行命令：

```bash
source /home/zhangwh/miniconda3/etc/profile.d/conda.sh
conda activate flexatten
cd /home/zhangwh/flexatten-nv
TORCH_LOGS="+dynamo,+inductor" python causal_attention_trace/causal_flexattention_trace.py \
  > causal_attention_trace/artifacts/causal_flexattention_run.log 2>&1
```

关键产物：

| 文件 | 说明 |
|------|------|
| `causal_attention_trace/causal_flexattention_trace.py` | 最小 causal FlexAttention 代码 |
| `causal_attention_trace/artifacts/causal_flexattention_summary.json` | 结构化运行结果、环境和生成代码线索 |
| `causal_attention_trace/artifacts/trace_summary.md` | 可读版摘要 |
| `causal_attention_trace/artifacts/causal_flexattention_run.log` | Dynamo/Inductor 详细 trace 日志 |

## 实验结果

从 `causal_flexattention_summary.json` 读取到的核心结果：

| 项目 | 值 |
|------|----|
| PyTorch | `2.6.0+cu124` |
| GPU | `NVIDIA L4` |
| 输出 shape | `[1, 2, 128, 64]` |
| 输出 dtype | `torch.float16` |
| compile + first run | `4.4800s` |
| warm run | `0.0002146s` |
| max abs diff vs dense reference | `0.0009765625` |
| mean abs diff vs dense reference | `2.809e-05` |
| repeated compiled run diff | `0.0` |
| BlockMask BLOCK_SIZE | `[128, 128]` |

Inductor cache 里找到了生成 wrapper 和 Triton kernel 线索：

- `/tmp/torchinductor_zhangwh/.../cjobrshugelg4owymnpe4od6lvzvdyivqy7mg6zuidolbu3g2x5u.py` 包含 `torch._inductor.kernel.flex_attention` 和 `async_compile.triton(...)`。
- `/tmp/torchinductor_zhangwh/.../cldcbqjjeryjrad3mhbuwpayckayd4nxkhytxlvi54ktj7mvpt2m.py` 包含 `def triton_flex_attention(...)` 与多个 `@triton.jit`。
- trace log 中也能看到 `CachingAutotuner gets 1 configs for triton_flex_attention`，说明最终可执行 kernel 是 Triton FlexAttention。

## 源码链路拆解

### 1. Python API 层：描述 mask，而不是手写 attention matrix

PyTorch 源码 `/home/zhangwh/pytorch/torch/nn/attention/flex_attention.py` 中，`create_block_mask` 的接口在 1480 行附近。它接收 `mask_mod(b, h, q_idx, kv_idx)`，示例里也正是 causal mask：

```python
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx
```

`create_block_mask` 会先调用 `create_mask` materialize 一个 bool mask，再用 `_convert_mask_to_block_mask(..., separate_full_blocks=True)` 转成 partial/full block 结构，最后封装为 `BlockMask`。本实验的 `BlockMask` 里 `full_kv_num_blocks_is_none=False`，意味着它能区分完整块和需要 mask_mod 的边界块。

### 2. flex_attention API 层：进入 HigherOrderOperator

`flex_attention` 的主体在同一文件 1818 行附近。关键路径：

- 1897-1900：校验 q/k/v，并调整 memory layout。
- 1932-1936：没有 `score_mod` 就使用 identity；没有 `block_mask` 就创建 empty block mask。
- 1973-1974：默认 scale 是 `1/sqrt(head_dim)`。
- 1996-2003：补齐 kernel options。
- 2034-2048：如果处于 Dynamo compile 中，调用 `flex_attention_hop(...)`。
- 2053-2079：如果用户直接 eager 调用，也会内部用 `torch.compile` 包一层 HOP wrapper，避免 materialize 全量 score matrix。

所以本测试中，Python 的 causal mask 不会停在 Python 循环里逐元素执行，而是作为 `mask_graph` 被捕获并下沉给编译器。

### 3. HigherOrderOperator 层：把 score_mod/mask_mod 变成 FX 子图

`/home/zhangwh/pytorch/torch/_higher_order_ops/flex_attention.py` 中：

- 94-125：`FlexAttentionHOP` 定义了 `torch.ops.higher_order.flex_attention`。
- 406-470：`trace_flex_attention` 用 `reenter_make_fx` 分别 trace `score_mod` 和 `mask_mod`，注册成 `sdpa_score` / `sdpa_mask` 子模块。
- 601-635：fake impl 只推导输出、LSE、max_scores 的 shape/stride，让 Inductor 能继续 lowering。

本次 trace log 里能看到 FX graph 生成了 `score_mod_0` 和 `mask_fn_0`：`score_mod_0` 是 identity，`mask_fn_0` 的主体是 `child_2 >= child_3`，对应 `q_idx >= kv_idx`。

### 4. Inductor lowering 层：选择 Triton / Decode / FLASH(CuteDSL)

`/home/zhangwh/pytorch/torch/_inductor/kernel/flex/flex_attention.py` 中：

- 99-106：定义 `flex_attention_template = TritonTemplate(...)`，模板源是 `templates/flex_attention.py.jinja + utilities + common`。
- 109-126：注册 `torch.ops.higher_order.flex_attention` 的 lowering。
- 165：从 `kernel_options` 取出 `BACKEND`。
- 221-244：如果是短序列 decode 场景，可能走 flex decoding Triton template。
- 274-290：如果 `_use_flex_flash_attention(...)` 为真，则进入 `create_flex_flash_attention_kernel(...)`。
- 396-470：否则枚举 Triton configs，调用 `flex_attention_template.maybe_append_choice(...)`。
- 495-500：用 `autotune_select_algorithm("flex_attention", choices, ...)` 选择并编译 kernel。

当前安装版 PyTorch 2.6.0 没有源码里这些完整 `BACKEND` 逻辑，实际 trace 显示默认路径降到 `triton_flex_attention`。

### 5. Triton template 层：块稀疏 online softmax

`/home/zhangwh/pytorch/torch/_inductor/kernel/flex/templates/flex_attention.py.jinja` 是最终 Triton 模板。它的核心结构是：

1. grid 是 `(ceil_div(num_queries, BLOCK_M), batch, q_heads)`。
2. 每个 program 处理一个 query block。
3. 根据 `KV_NUM_BLKS/KV_IDX` 遍历 partial blocks，需要同时应用 `mask_mod` 和 `score_mod`。
4. 根据 `FULL_KV_NUM_BLKS/FULL_KV_IDX` 遍历 full blocks，只应用 `score_mod`，不用再跑 mask。
5. 用 online softmax 累积 `m_i/l_i/acc`，最后写 output 和可选 LSE。

这就是 FlexAttention 避免显式 materialize `[B,H,Q,K]` dense scores 的关键：mask 语义以 FX 子图形式被 inline 到 kernel 模板中，block mask 控制需要访问哪些 KV block。

## CuteDSL / CUTLASS 接入点

下载的 PyTorch 源码里，CuteDSL 路径在 `/home/zhangwh/pytorch/torch/_inductor/kernel/flex/flex_flash_attention.py`：

- 64-74：`ensure_flash_available()` 检查 `flash_attn.cute` 是否可 import。
- 77-85：创建 `CuteDSLTemplate`：`flash_attention_cutedsl` 和 `flash_attention_backward_cutedsl`。
- 290-319：`_use_flex_flash_attention` 只在 `backend == "FLASH"` 时返回 True，所以这是显式选择，不是 AUTO 默认路径。
- 322-350：`create_flex_flash_attention_kernel` 检查 dtype 和 `flash_attn.cute` 可用性。
- 421-438：调用 `flash_attention_cutedsl_template.maybe_append_choice(...)` 生成 CuteDSL choice。

再往下，`/home/zhangwh/pytorch/torch/_inductor/codegen/cutedsl/cutedsl_kernel.py` 会生成包含这些 import 的代码：

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
```

也就是说，真正的 CUTLASS/CuteDSL 执行链路是：

```text
flex_attention(..., kernel_options={"BACKEND": "FLASH"})
  -> HOP trace score_mod/mask_mod
  -> Inductor flex_attention lowering
  -> _use_flex_flash_attention(... backend="FLASH")
  -> create_flex_flash_attention_kernel
  -> CuteDSLTemplate("flash_attention_cutedsl")
  -> codegen/cutedsl imports cutlass.cute
  -> async_compile.cutedsl(...)
  -> CUTLASS/CuTe runtime kernel
```

但这条链路对环境有两个硬要求：

1. Python 侧的 PyTorch 必须是包含 `BACKEND` / `flex_flash_attention.py` / `codegen.cutedsl` 的构建版本。
2. `flash_attn.cute` 和 `cutlass.cute` 必须能 import，通常需要安装/编译对应的 FlashAttention 4 / nvidia-cutlass Python 包。

当前服务器状态：

| 检查项 | 结果 |
|--------|------|
| installed `torch.nn.attention.flex_attention` signature | 没有 `return_aux`，源码中不含 `BACKEND` 文本 |
| `/home/zhangwh/pytorch` 源码 | 有 `BACKEND="FLASH"` 和 CuteDSL lowering，但不是当前 import 的 torch |
| `/home/zhangwh/flash-attention/flash_attn/cute` | 源码存在 |
| `PYTHONPATH=/home/zhangwh/flash-attention` import `flash_attn.cute` | 失败：缺少 `flash_attn_2_cuda` |
| `cutlass.cute` | 不可 import |
| `PYTHONPATH=/home/zhangwh/pytorch` import torch | 失败：源码树未 build，缺少 `torch.version` |

因此，本次提交里的脚本不是伪造 CuteDSL 调用，而是明确探测 `kernel_options={"BACKEND":"FLASH"}` 并记录失败；报告用源码说明如果环境补齐，FlexAttention 会在哪里切入 CuteDSL/CUTLASS。

## 为什么 causal mask 能被编进 GPU kernel

这个例子里，`causal_mask_mod` 是一个普通 Python 函数，但它只依赖 `q_idx` 和 `kv_idx`，没有 Python side effect。Dynamo/HOP tracing 会用 scalar fake tensors 调一次函数，把 `q_idx >= kv_idx` 捕获成 FX graph。Inductor lowering 再把这个 `mask_graph_buffer` 传给 Triton template；模板在 partial KV blocks 里生成对应的 `tl.where` / predicate，而不是在 Python 里生成完整 mask 后每次读 dense matrix。

对于 causal attention，BlockMask 还能把大部分严格下三角区域识别为 full blocks：full blocks 不需要执行 mask_mod，只需要做 QK、softmax 和 PV；只有边界块需要逐元素判断 `q_idx >= kv_idx`。这也是 FlexAttention 在结构化稀疏 mask 上比 naive dense mask 更有价值的地方。

## 后续如果要真正跑到 CUTLASS/CuteDSL

建议路径如下：

1. 在服务器上 build `/home/zhangwh/pytorch`，确保 `import torch` 来自该源码构建，而不是 `miniconda3/envs/flexatten/site-packages` 的 2.6.0 wheel。
2. 编译 `/home/zhangwh/flash-attention`，生成 `flash_attn_2_cuda`，并让 `flash_attn.cute` 可 import。
3. 安装或配置 nvidia-cutlass Python 包，使 `import cutlass.cute as cute` 成功。
4. 重新运行脚本，并把 `kernel_options={"BACKEND":"FLASH"}` 作为正式路径，而不是 probe。
5. 预期 trace 中应出现 `flash_attention_cutedsl`、`async_compile.cutedsl(...)`、`cutlass.cute`，而不是 `triton_flex_attention`。

## 本次提交的边界

本次在现有环境中完成了：

- 一个真实可运行的 causal FlexAttention GPU 示例。
- 数值正确性校验。
- Dynamo/Inductor/Triton 执行链路抓取。
- 基于下载的 PyTorch 源码分析 FlexAttention 接入 GPU、以及新源码中接入 CuteDSL/CUTLASS 的位置。

未完成真实 CUTLASS/CuteDSL 执行，因为当前服务器环境缺少对应的 PyTorch 构建和 Python/CUDA 扩展。这个限制已经由脚本和日志直接记录，避免把 Triton 路径误报成 CuteDSL 路径。
