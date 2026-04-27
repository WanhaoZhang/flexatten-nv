# Causal FlexAttention Trace Summary

- torch: `2.6.0+cu124` from `/home/zhangwh/miniconda3/envs/flexatten/lib/python3.11/site-packages/torch/__init__.py`
- GPU: `NVIDIA L4`
- compiled output: `[1, 2, 128, 64]` `torch.float16`
- first compile+run: `4.4800s`; warm run: `0.000215s`
- max abs diff vs dense causal reference: `0.0009765625`
- flash/CuteDSL probe status: `failed_or_unavailable`
- torchinductor cache root: `/tmp/torchinductor_zhangwh`

## Generated Code Hints

- `/tmp/torchinductor_zhangwh/tq/ctqwer2hwtflrvmbfufbidquuv3htvioqmaaaz4imzd2benrmswc.py`
  - L6: `from torch._inductor.runtime import triton_helpers, triton_heuristics`
  - L7: `from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math`
  - L10: `@triton_heuristics.template(`
  - L13: `    triton_meta={'signature': {'arg_Q': '*fp16', 'arg_K': '*fp16', 'arg_V': '*fp16', 'arg_LSE': '*fp32', 'arg_KV_NUM_BLKS': '*i32', 'arg_KV_IDX': '*i32', 'arg_FULL_KV_NUM_BLKS': '*i32', 'arg_FULL_KV_IDX': '*i32', 'out_ptr0': '*fp16'}, 'devi`
- `/tmp/torchinductor_zhangwh/jo/cjobrshugelg4owymnpe4od6lvzvdyivqy7mg6zuidolbu3g2x5u.py`
  - L16: `import torch._inductor.kernel.flex_attention`
  - L20: `from torch._inductor.runtime.triton_heuristics import (`
  - L44: `# Topologically Sorted Source Nodes: [flex_attention], Original ATen: []`
  - L46: `#   flex_attention => flex_attention`
- `/tmp/torchinductor_zhangwh/ld/cldcbqjjeryjrad3mhbuwpayckayd4nxkhytxlvi54ktj7mvpt2m.py`
  - L7: `from torch._inductor.runtime import triton_helpers, triton_heuristics`
  - L8: `from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math`
  - L11: `@triton_heuristics.template(`
  - L14: `    triton_meta={'signature': {'arg_Q': '*fp16', 'arg_K': '*fp16', 'arg_V': '*fp16', 'arg_LSE': '*fp32', 'arg_KV_NUM_BLKS': '*i32', 'arg_KV_IDX': '*i32', 'arg_FULL_KV_NUM_BLKS': '*i32', 'arg_FULL_KV_IDX': '*i32', 'out_ptr0': '*fp16'}, 'devi`
- `/tmp/torchinductor_zhangwh/hl/chl7wff6337a3pbyma5ri7xle63wh2yu3fxsepmqkeodjxg2cg5s.py`
  - L6: `from torch._inductor.runtime import triton_helpers, triton_heuristics`
  - L7: `from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math`
  - L10: `@triton_heuristics.template(`
  - L13: `    triton_meta={'signature': {'arg_Q': '*fp16', 'arg_K': '*fp16', 'arg_V': '*fp16', 'arg_LSE': '*fp32', 'arg_KV_NUM_BLKS': '*i32', 'arg_KV_IDX': '*i32', 'arg_FULL_KV_NUM_BLKS': '*i32', 'arg_FULL_KV_IDX': '*i32', 'out_ptr0': '*fp16'}, 'devi`
- `/tmp/torchinductor_zhangwh/ua/cuactwn7hdymygeojykqetse24ypy3q24sksnql6feuoy4iud675.py`
  - L16: `import torch._inductor.kernel.flex_attention`
  - L20: `from torch._inductor.runtime.triton_heuristics import (`
  - L44: `# Topologically Sorted Source Nodes: [flex_attention], Original ATen: []`
  - L46: `#   flex_attention => flex_attention`
