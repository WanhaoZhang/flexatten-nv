# PyTorch FlexAttention 到 CuteDSL/CUTLASS 的执行链路复测报告

日期：2026-04-27  
机器：GCE instance-20260405-l4gpu, NVIDIA L4, compute capability 8.9  
目标：在服务器重启后重新搭环境，验证 PyTorch FlexAttention 是否能走到 Cutlass/CuteDSL 路径，并记录当前 L4 上的最终限制。

## 结论

这次已经让 torch.nn.attention.flex_attention.flex_attention 进入了 PyTorch Inductor 的 FLASH 后端，并实际调用到生成的 CuteDSL wrapper：

1. 用户代码调用 torch.compile(flex_attention, fullgraph=True)。
2. 调用时传入 kernel_options={"BACKEND": "FLASH"}。
3. PyTorch nightly 的 FlexAttention lowering 选择 torch._inductor.kernel.flex.flex_flash_attention。
4. Inductor 生成 /tmp/torchinductor_zhangwh/.../c4uqww...py，其中调用 cutedsl_fused_flex_attention_859db5d0.run(...)。
5. 该 wrapper 进入 /tmp/torchinductor_zhangwh/.../cqa6zace...py 的 cutedsl_fused_flex_attention_859db5d0_main。
6. 生成代码继续调用 /home/zhangwh/flash-attention/flash_attn/cute/interface.py 中的 _flash_attn_fwd(...)。
7. _flash_attn_fwd 内部进入 FlashAttention-4 CuteDSL/CUTLASS 路径，并在架构检查处停止：assert arch // 10 in [9, 10, 11]。

所以，FlexAttention 到 CuteDSL/CUTLASS 的接入链路已经跑通到运行时入口；当前未能完成数值 kernel 的原因不是 Python 接入失败，而是 L4 是 sm89，FlashAttention-4 Cute 前向 kernel 当前只放行 9.x/10.x/11.x 架构。

## 新环境

为了和旧的 PyTorch 2.6.0 环境隔离，我新建了 conda 环境：

| 项目 | 值 |
|------|----|
| conda env | flexcute |
| Python | 3.11 |
| PyTorch | 2.13.0.dev20260427+cu126 |
| CUDA runtime in torch | 12.6 |
| GPU | NVIDIA L4 |
| compute capability | (8, 9) |
| Cutlass/CuteDSL | nvidia-cutlass-dsl==4.4.2 |
| FlashAttention Cute | editable install from /home/zhangwh/flash-attention/flash_attn/cute |

环境摘要已保存到：causal_attention_trace/artifacts/flexcute_environment_summary.json。

关键模块位置：

| 模块 | 路径 |
|------|------|
| cutlass.cute | /home/zhangwh/miniconda3/envs/flexcute/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/__init__.py |
| flash_attn.cute | /home/zhangwh/flash-attention/flash_attn/cute/__init__.py |
| torch._inductor.codegen.cutedsl.cutedsl_template | /home/zhangwh/miniconda3/envs/flexcute/lib/python3.11/site-packages/torch/_inductor/codegen/cutedsl/cutedsl_template.py |
| torch._inductor.kernel.flex.flex_flash_attention | /home/zhangwh/miniconda3/envs/flexcute/lib/python3.11/site-packages/torch/_inductor/kernel/flex/flex_flash_attention.py |

## 复现实验

新增脚本：causal_attention_trace/flexcute_flash_backend_probe.py

运行方式：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flexcute
cd /home/zhangwh/flexatten-nv
TORCH_LOGS="+dynamo,+inductor" python causal_attention_trace/flexcute_flash_backend_probe.py \
  > causal_attention_trace/artifacts/flexcute_flash_backend_probe.log 2>&1
```

测试内容是一个最小 causal attention：

- B=1, H=2, S=128, D=64
- q/k/v 为 CUDA FP16 tensor
- create_block_mask(causal_mask, ...) 构造 causal BlockMask
- torch.compile(flex_attention, fullgraph=True) 编译
- 调用时传入 kernel_options={"BACKEND": "FLASH"} 强制走 PyTorch 新版 FlexAttention 的 FLASH lowering

日志已保存到：causal_attention_trace/artifacts/flexcute_flash_backend_probe.log。

## 关键日志

日志中能看到环境和模块都已经就绪：

```text
torch 2.13.0.dev20260427+cu126 /home/zhangwh/miniconda3/envs/flexcute/lib/python3.11/site-packages/torch/__init__.py
cuda 12.6 True NVIDIA L4 (8, 9)
cutlass.cute ... /nvidia_cutlass_dsl/python_packages/cutlass/cute/__init__.py
flash_attn.cute ... /home/zhangwh/flash-attention/flash_attn/cute/__init__.py
```

最重要的调用栈如下：

```text
/tmp/torchinductor_zhangwh/4u/c4uqwwvlmgrykmi4ful5wky2evnwoeg23ttndkm54cmzmov7v6ew.py:161
    cutedsl_fused_flex_attention_859db5d0.run(...)

/tmp/torchinductor_zhangwh/qa/cqa6zaceijiweypytoe3azxseyscluo7j746clruld6eqocfg5xf.py:57
    _flash_attn_fwd(...)

/home/zhangwh/flash-attention/flash_attn/cute/interface.py:258
    assert arch // 10 in [9, 10, 11], "Unsupported compute capability. Supported: 9.x, 10.x, 11.x"
```

最终错误：

```text
FLASH_BACKEND_FAILED AssertionError Unsupported compute capability. Supported: 9.x, 10.x, 11.x
```

这说明执行已经越过了 PyTorch 前端、Dynamo、AOT/Inductor lowering 和 CuteDSL wrapper 生成阶段，进入到了 FlashAttention-4 Cute 实现的 runtime guard。

## Inductor 生成代码证据

为了避免 /tmp/torchinductor_zhangwh 被清理，我把相关生成代码的关键片段保存到了：

causal_attention_trace/artifacts/flexcute_generated_cutedsl_snippets.txt

其中能看到两段核心证据。

第一段，Inductor 外层 wrapper 调用 CuteDSL kernel：

```text
cutedsl_fused_flex_attention_859db5d0.run(arg0_1, arg1_1, arg2_1, buf1, arg4_1, arg3_1, arg5_1, arg6_1, buf2, stream=stream0)
```

第二段，CuteDSL kernel body 调 FlashAttention-4 Cute interface：

```text
from flash_attn.cute.interface import _flash_attn_fwd
_flash_attn_fwd(...)
```

这正是 PyTorch FlexAttention 接入 GPU CuteDSL/CUTLASS 的实际路径。

## 为什么旧环境只能看到 Triton

旧环境 flexatten 使用的是 torch 2.6.0+cu124。这个版本的 torch.nn.attention.flex_attention 主要落到 Triton flex attention template；在安装包里没有当前 nightly 的 BACKEND="FLASH" 入口，也没有新版 torch._inductor.kernel.flex.flex_flash_attention 这条 lowering 路径。

新版 nightly 中，kernel_options 的 BACKEND 可以选择 AUTO/TRITON/FLASH/TRITON_DECODE。指定 FLASH 后，FlexAttention lowering 会尝试走 FlashAttention-4 CuteDSL backend。也就是说，这次能进入 CuteDSL 的关键不是改业务 mask，而是换到包含该 backend 的 PyTorch nightly，并安装对应的 flash-attn-4 与 nvidia-cutlass-dsl。

## 当前 L4 的限制

L4 是 Ada Lovelace GPU，compute capability 为 8.9。FlashAttention-4 Cute interface 当前显式要求：

```python
assert arch // 10 in [9, 10, 11]
```

对 sm89 来说，arch // 10 == 8，所以会抛出 Unsupported compute capability。这不是 FlexAttention mask、BlockMask、Dynamo 或 Inductor 的问题，而是 FA4 Cute kernel 的硬件支持范围限制。

如果要看到完整数值输出 FLASH_BACKEND_OK，需要把同一环境和脚本放到支持 sm90+ 的 GPU 上，例如 H100/H200，或支持 10.x/11.x 的新架构设备。

## 磁盘清理记录

安装 nightly PyTorch、Cutlass DSL 和 FlashAttention-4 期间，根分区一度接近满盘。按用户授权，我删除了 TensorRT-LLM 下载模型候选目录：

- 删除路径：/home/zhangwh/qwen_model
- 删除前大小：约 15G
- 删除原因：释放根分区空间，继续安装/验证 FlexAttention CuteDSL 环境
- 详细记录：causal_attention_trace/artifacts/tensorrtllm_deleted_model_artifacts.txt

删除后根分区可用空间从约 22G 提升到约 36G；当前检查约为 31G 可用。

## 文件清单

| 文件 | 说明 |
|------|------|
| causal_attention_trace/flexcute_flash_backend_probe.py | 最小 causal FlexAttention + BACKEND="FLASH" 复现实验 |
| causal_attention_trace/artifacts/flexcute_flash_backend_probe.log | 含 Dynamo/Inductor trace 和失败栈的完整日志 |
| causal_attention_trace/artifacts/flexcute_environment_summary.json | 新环境和关键模块路径 |
| causal_attention_trace/artifacts/flexcute_generated_cutedsl_snippets.txt | Inductor 生成代码中 CuteDSL 调用片段 |
| causal_attention_trace/artifacts/tensorrtllm_deleted_model_artifacts.txt | 按授权删除 TensorRT-LLM 模型目录的记录 |

## 下一步建议

1. 在 H100/H200 或其他 sm90+ 机器上复用 flexcute 环境安装步骤和新增脚本，验证最终数值输出。
2. 如果必须在 L4 上跑完整 kernel，目前应继续使用 Triton FlexAttention backend，而不是 FA4 Cute backend。
3. 如果后续 FlashAttention-4 Cute 增加 sm89 支持，再重新运行同一脚本即可验证是否解除限制。
