#!/usr/bin/env python3
"""Minimal causal attention implemented with torch.nn.attention.flex_attention.

This script is intentionally small and inspectable. It compiles a causal
FlexAttention call on CUDA, checks it against a dense PyTorch reference, records
runtime metadata, and probes whether the installed environment can request the
newer FlexAttention FLASH/CuteDSL backend.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


BATCH = 1
HEADS = 2
SEQ_LEN = 128
HEAD_DIM = 64
DTYPE = torch.float16
DEVICE = "cuda"


def causal_mask_mod(batch: torch.Tensor, head: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
    """Keep token j visible to token i iff j <= i."""
    return q_idx >= kv_idx


def dense_causal_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    causal = torch.ones((q.shape[-2], k.shape[-2]), dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


def module_spec(name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    if spec is None:
        return None
    return str(spec.origin or spec.submodule_search_locations)


def environment_summary() -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "driver_runtime_cuda": torch.version.cuda,
        }
    specs = {mod: module_spec(mod) for mod in ["flash_attn", "flash_attn.cute", "cutlass", "cutlass.cute", "triton"]}
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_file": torch.__file__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "module_specs": specs,
        "flex_attention_source": inspect.getsourcefile(flex_attention),
        "flex_attention_signature": str(inspect.signature(flex_attention)),
        "flex_attention_has_backend_option_in_python_source": "BACKEND" in inspect.getsource(flex_attention),
    }


def run_compiled_flexattention(kernel_options: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    torch.manual_seed(20260427)
    q = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    block_mask = create_block_mask(
        causal_mask_mod,
        BATCH,
        HEADS,
        SEQ_LEN,
        SEQ_LEN,
        device=DEVICE,
    )

    compiled = torch.compile(flex_attention, fullgraph=True)
    started = time.perf_counter()
    out = compiled(q, k, v, block_mask=block_mask, kernel_options=kernel_options)
    torch.cuda.synchronize()
    compile_and_first_run_s = time.perf_counter() - started

    started = time.perf_counter()
    out2 = compiled(q, k, v, block_mask=block_mask, kernel_options=kernel_options)
    torch.cuda.synchronize()
    warm_run_s = time.perf_counter() - started

    ref = dense_causal_reference(q, k, v)
    max_abs_diff = (out.float() - ref.float()).abs().max().item()
    mean_abs_diff = (out.float() - ref.float()).abs().mean().item()
    second_run_diff = (out.float() - out2.float()).abs().max().item()

    meta = {
        "shape": list(out.shape),
        "dtype": str(out.dtype),
        "kernel_options": kernel_options or {},
        "compile_and_first_run_s": compile_and_first_run_s,
        "warm_run_s": warm_run_s,
        "max_abs_diff_vs_dense_reference": max_abs_diff,
        "mean_abs_diff_vs_dense_reference": mean_abs_diff,
        "max_abs_diff_between_compiled_runs": second_run_diff,
        "block_mask": {
            "kv_num_blocks_shape": list(block_mask.kv_num_blocks.shape),
            "kv_indices_shape": list(block_mask.kv_indices.shape),
            "full_kv_num_blocks_is_none": block_mask.full_kv_num_blocks is None,
            "BLOCK_SIZE": list(block_mask.BLOCK_SIZE),
        },
    }
    return out, meta


def probe_flash_backend() -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested_kernel_options": {"BACKEND": "FLASH"},
        "flash_attn_cute_importable": module_spec("flash_attn.cute") is not None,
        "installed_flex_attention_signature": str(inspect.signature(flex_attention)),
        "installed_python_source_mentions_BACKEND": "BACKEND" in inspect.getsource(flex_attention),
    }
    try:
        _, meta = run_compiled_flexattention({"BACKEND": "FLASH"})
        result["status"] = "ran"
        result["run_meta"] = meta
    except Exception as exc:
        result["status"] = "failed_or_unavailable"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc).splitlines()[:8]
    return result


def collect_generated_code_hints(cache_root: Path) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    if not cache_root.exists():
        return hints
    needles = ["flex_attention", "higher_order.flex_attention", "triton_", "@triton"]
    candidates = sorted(cache_root.rglob("*.py"), key=lambda p: p.stat().st_mtime, reverse=True)[:200]
    for path in candidates:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if any(n in text for n in needles):
            lines = []
            for i, line in enumerate(text.splitlines(), start=1):
                if any(n in line for n in needles):
                    lines.append({"line": i, "text": line[:240]})
                if len(lines) >= 12:
                    break
            hints.append({"path": str(path), "matches": lines})
        if len(hints) >= 10:
            break
    return hints


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this FlexAttention GPU trace")

    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(exist_ok=True)
    cache_root = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_" + os.environ.get("USER", "unknown")))

    env = environment_summary()
    out, run_meta = run_compiled_flexattention()
    flash_probe = probe_flash_backend()
    generated_hints = collect_generated_code_hints(cache_root)

    summary = {
        "environment": env,
        "default_triton_flexattention_run": run_meta,
        "flash_cutedsl_probe": flash_probe,
        "torchinductor_cache_root": str(cache_root),
        "generated_code_hints": generated_hints,
        "output_checksum": float(out.float().sum().item()),
    }
    (out_dir / "causal_flexattention_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    md_lines = [
        "# Causal FlexAttention Trace Summary",
        "",
        f"- torch: `{env['torch_version']}` from `{env['torch_file']}`",
        f"- GPU: `{env['gpu']['name'] if env['gpu'] else 'none'}`",
        f"- compiled output: `{run_meta['shape']}` `{run_meta['dtype']}`",
        f"- first compile+run: `{run_meta['compile_and_first_run_s']:.4f}s`; warm run: `{run_meta['warm_run_s']:.6f}s`",
        f"- max abs diff vs dense causal reference: `{run_meta['max_abs_diff_vs_dense_reference']}`",
        f"- flash/CuteDSL probe status: `{flash_probe['status']}`",
        f"- torchinductor cache root: `{cache_root}`",
        "",
        "## Generated Code Hints",
        "",
    ]
    if generated_hints:
        for item in generated_hints[:5]:
            md_lines.append(f"- `{item['path']}`")
            for match in item["matches"][:4]:
                md_lines.append(f"  - L{match['line']}: `{match['text']}`")
    else:
        md_lines.append("No generated Python wrapper containing the FlexAttention markers was found in the inspected cache window.")
    md_lines.append("")
    (out_dir / "trace_summary.md").write_text("\n".join(md_lines))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
