#!/usr/bin/env python3
# Probe PyTorch FlexAttention FLASH backend and record the CuteDSL/CUTLASS path.
# Run in flexcute:
#   TORCH_LOGS="+dynamo,+inductor" python causal_attention_trace/flexcute_flash_backend_probe.py

from __future__ import annotations

import importlib.util
import inspect
import math
import traceback

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


def causal_mask(batch, head, query_idx, kv_idx):
    return query_idx >= kv_idx


def module_origin(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    return str(spec.origin)


def main() -> None:
    print("torch", torch.__version__, torch.__file__)
    print(
        "cuda",
        torch.version.cuda,
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_capability(0),
    )
    print("flex_attention_signature", inspect.signature(flex_attention))
    for name in [
        "cutlass.cute",
        "flash_attn.cute",
        "torch._inductor.codegen.cutedsl.cutedsl_template",
        "torch._inductor.kernel.flex.flex_flash_attention",
    ]:
        print(name, module_origin(name))

    batch, heads, seq_len, head_dim = 1, 2, 128, 64
    q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    block_mask = create_block_mask(causal_mask, batch, heads, seq_len, seq_len, device="cuda")
    print("block_mask", block_mask)

    try:
        compiled = torch.compile(flex_attention, fullgraph=True)
        out = compiled(q, k, v, block_mask=block_mask, kernel_options={"BACKEND": "FLASH"})
        torch.cuda.synchronize()

        ref_scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(head_dim)
        mask = torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool).tril()
        ref = (torch.softmax(ref_scores.masked_fill(~mask, float("-inf")), dim=-1) @ v.float()).to(torch.float16)
        print("FLASH_BACKEND_OK", out.shape, out.dtype, float((out.float() - ref.float()).abs().max()))
    except Exception as exc:
        print("FLASH_BACKEND_FAILED", type(exc).__name__, str(exc))
        traceback.print_exc(limit=80)


if __name__ == "__main__":
    main()
