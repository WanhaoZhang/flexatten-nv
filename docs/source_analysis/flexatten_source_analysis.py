#!/usr/bin/env python3
"""
FlexAttention Source Code Analysis: Three Attention Paths
=========================================================
Deep analysis of three attention computation paths:
  Path 1: Vanilla PyTorch (manual QK^T + softmax + V)
  Path 2: SDPA / FlashAttention2 (F.scaled_dot_product_attention)
  Path 3: FlexAttention (torch.compile + Triton)

Experiments trace each path, compare intermediate states,
analyze compilation overhead, and benchmark comprehensively.

All chart text is pure ASCII.
NVIDIA L4 (24GB) | PyTorch 2.6.0+cu124 | Triton 3.2.0
"""

import torch
import torch.nn.functional as F
import time
import json
import gc
import os
import sys
from collections import defaultdict

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

DEVICE = "cuda"
DTYPE = torch.float16
FIGURES_DIR = "figures_source"
os.makedirs(FIGURES_DIR, exist_ok=True)

torch.manual_seed(42)

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def measure(func, warmup=3, repeat=10):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return (sum(times) / len(times)) * 1000  # ms

def get_peak_memory():
    return torch.cuda.max_memory_allocated() / 1e9


# ============================================================
# Experiment F1: Three-Path Step-by-Step Trace
# ============================================================
def experiment_f1():
    """Trace each path step by step, measure per-step cost."""
    print("=" * 70)
    print("Experiment F1: Three-Path Step-by-Step Trace")
    print("=" * 70)

    B, H, S, D = 1, 8, 2048, 64
    q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    scale = 1.0 / (D ** 0.5)

    results = {"path_vanilla": {}, "path_sdpa": {}, "path_flex": {}}

    # --- Path 1: Vanilla PyTorch ---
    print("\n--- Vanilla Path ---")
    clear_gpu()
    steps_vanilla = {}

    # Step 1: QK^T
    t0 = time.perf_counter()
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    torch.cuda.synchronize()
    steps_vanilla["1_qkt"] = (time.perf_counter() - t0) * 1000
    steps_vanilla["1_qkt_mem"] = scores.nelement() * scores.element_size() / 1e6
    print(f"  1. QK^T: {steps_vanilla['1_qkt']:.3f}ms, tensor={steps_vanilla['1_qkt_mem']:.1f}MB")

    # Step 2: Causal mask
    t0 = time.perf_counter()
    causal_mask = torch.ones(S, S, device=DEVICE, dtype=torch.bool).tril_()
    torch.cuda.synchronize()
    steps_vanilla["2_causal"] = (time.perf_counter() - t0) * 1000
    steps_vanilla["2_causal_mem"] = causal_mask.nelement() * causal_mask.element_size() / 1e6
    print(f"  2. Causal mask: {steps_vanilla['2_causal']:.3f}ms, tensor={steps_vanilla['2_causal_mem']:.1f}MB")

    # Step 3: Apply mask
    t0 = time.perf_counter()
    scores = scores.masked_fill(~causal_mask, float('-inf'))
    torch.cuda.synchronize()
    steps_vanilla["3_mask_apply"] = (time.perf_counter() - t0) * 1000
    print(f"  3. Apply mask: {steps_vanilla['3_mask_apply']:.3f}ms")

    # Step 4: Softmax
    t0 = time.perf_counter()
    attn_weights = F.softmax(scores.float(), dim=-1).to(DTYPE)
    torch.cuda.synchronize()
    steps_vanilla["4_softmax"] = (time.perf_counter() - t0) * 1000
    steps_vanilla["4_softmax_mem"] = attn_weights.nelement() * attn_weights.element_size() / 1e6
    print(f"  4. Softmax: {steps_vanilla['4_softmax']:.3f}ms, tensor={steps_vanilla['4_softmax_mem']:.1f}MB")

    # Step 5: x V
    t0 = time.perf_counter()
    output = torch.matmul(attn_weights, v)
    torch.cuda.synchronize()
    steps_vanilla["5_xv"] = (time.perf_counter() - t0) * 1000
    print(f"  5. x V: {steps_vanilla['5_xv']:.3f}ms")

    steps_vanilla["total_mem_gb"] = round(get_peak_memory(), 3)
    steps_vanilla["kernel_launches"] = 5
    steps_vanilla["hbm_writes"] = "5+ (scores, mask, masked_scores, weights, output)"
    results["path_vanilla"] = steps_vanilla

    # --- Path 2: SDPA / FlashAttention2 ---
    print("\n--- SDPA Path ---")
    clear_gpu()
    steps_sdpa = {}

    # SDPA is a single fused kernel - no intermediate steps visible
    t0 = time.perf_counter()
    output_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    steps_sdpa["1_total"] = (time.perf_counter() - t0) * 1000
    steps_sdpa["kernel_launches"] = 1
    steps_sdpa["hbm_writes"] = "1 (output only)"
    steps_sdpa["total_mem_gb"] = round(get_peak_memory(), 3)
    steps_sdpa["intermediate_tensors"] = "0 (all in SRAM)"

    # Test SDPA backend info
    try:
        # Check which backend SDPA uses
        with torch.nn.attention.sdpa_kernel(
            [torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
            t_flash = measure(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        steps_sdpa["flash2_ms"] = round(t_flash, 3)
    except Exception:
        steps_sdpa["flash2_ms"] = -1

    try:
        with torch.nn.attention.sdpa_kernel(
            [torch.nn.attention.SDPBackend.MATH]):
            t_math = measure(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        steps_sdpa["math_ms"] = round(t_math, 3)
    except Exception:
        steps_sdpa["math_ms"] = -1

    print(f"  Flash2: {steps_sdpa.get('flash2_ms', '?')}ms")
    print(f"  Math: {steps_sdpa.get('math_ms', '?')}ms")
    results["path_sdpa"] = steps_sdpa

    # --- Path 3: FlexAttention ---
    print("\n--- FlexAttention Path ---")
    clear_gpu()
    steps_flex = {}

    # Step 1: Define mask function (no computation)
    t0 = time.perf_counter()
    def causal_mask_fn(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    steps_flex["0_mask_def"] = (time.perf_counter() - t0) * 1000
    print(f"  0. Define mask: {steps_flex['0_mask_def']:.6f}ms (trivial)")

    # Step 2: Create BlockMask (analysis + vmap)
    clear_gpu()
    t0 = time.perf_counter()
    block_mask = create_block_mask(causal_mask_fn, B, 1, S, S, device=DEVICE)
    torch.cuda.synchronize()
    steps_flex["1_blockmask"] = (time.perf_counter() - t0) * 1000

    # BlockMask structure analysis
    if hasattr(block_mask, 'kv_num_blocks'):
        steps_flex["blockmask_shape"] = str(list(block_mask.kv_num_blocks.shape))
        steps_flex["non_empty_blocks"] = block_mask.kv_num_blocks.sum().item()
        total_blocks = (S // 128) * (S // 128)
        steps_flex["total_blocks"] = total_blocks
        steps_flex["block_utilization_pct"] = round(
            block_mask.kv_num_blocks.sum().item() / total_blocks * 100, 1)
    print(f"  1. BlockMask: {steps_flex['1_blockmask']:.3f}ms")

    # Step 3: First call (includes JIT compilation)
    clear_gpu()
    t0 = time.perf_counter()
    output_flex = flex_attention(q, k, v, block_mask=block_mask)
    torch.cuda.synchronize()
    steps_flex["2_first_call_compile"] = (time.perf_counter() - t0) * 1000
    steps_flex["2_first_call_mem_gb"] = round(get_peak_memory(), 3)
    print(f"  2. First call (JIT): {steps_flex['2_first_call_compile']:.3f}ms")

    # Step 4: Subsequent calls (cached kernel)
    clear_gpu()
    steps_flex["3_cached_ms"] = round(measure(
        lambda: flex_attention(q, k, v, block_mask=block_mask)), 3)
    steps_flex["3_cached_mem_gb"] = round(get_peak_memory(), 3)
    print(f"  3. Cached calls: {steps_flex['3_cached_ms']:.3f}ms")

    steps_flex["kernel_launches"] = 1
    steps_flex["hbm_writes"] = "1 (output only, same as Flash2)"
    results["path_flex"] = steps_flex

    # --- Numerical accuracy ---
    print("\n--- Numerical Accuracy ---")
    max_diff_v_s = (output - output_sdpa).abs().max().item()
    max_diff_v_f = (output - output_flex).abs().max().item()
    max_diff_s_f = (output_sdpa - output_flex).abs().max().item()
    print(f"  Vanilla vs SDPA: {max_diff_v_s:.6f}")
    print(f"  Vanilla vs Flex: {max_diff_v_f:.6f}")
    print(f"  SDPA vs Flex: {max_diff_s_f:.6f}")
    results["accuracy"] = {
        "vanilla_vs_sdpa": max_diff_v_s,
        "vanilla_vs_flex": max_diff_v_f,
        "sdpa_vs_flex": max_diff_s_f,
    }

    return results


# ============================================================
# Experiment F2: Compilation Overhead Analysis
# ============================================================
def experiment_f2():
    """Analyze JIT compilation overhead of FlexAttention."""
    print("\n" + "=" * 70)
    print("Experiment F2: Compilation Overhead Analysis")
    print("=" * 70)

    B, H, D = 1, 8, 64
    results = []

    # Test different mask patterns and their compile times
    patterns = {
        "causal": lambda b, h, q, kv: q >= kv,
        "sliding_window": lambda b, h, q, kv: (q >= kv) & ((q - kv) <= 256),
        "doc_packing_4": lambda b, h, q, kv: (q >= kv) & ((torch.arange(2048, device=DEVICE) // 512)[q] == (torch.arange(2048, device=DEVICE) // 512)[kv]),
    }

    for S in [512, 1024, 2048, 4096]:
        for pname, mask_fn_factory in patterns.items():
            clear_gpu()
            q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)

            # Create fresh mask function for this S
            if pname == "causal":
                mask_fn = lambda b, h, q, kv: q >= kv
            elif pname == "sliding_window":
                mask_fn = lambda b, h, q, kv: (q >= kv) & ((q - kv) <= 256)
            elif pname == "doc_packing_4":
                doc_ids = torch.arange(S, device=DEVICE) // (S // 4)
                mask_fn = lambda b, h, q, kv: (q >= kv) & (doc_ids[q] == doc_ids[kv])

            # Measure BlockMask creation time
            t0 = time.perf_counter()
            bm = create_block_mask(mask_fn, B, 1, S, S, device=DEVICE)
            torch.cuda.synchronize()
            blockmask_time = (time.perf_counter() - t0) * 1000

            # First call (compile)
            # Use a fresh torch.compile cache by using different functions
            t0 = time.perf_counter()
            out = flex_attention(q, k, v, block_mask=bm)
            torch.cuda.synchronize()
            first_call_time = (time.perf_counter() - t0) * 1000

            # Cached calls
            cached_time = measure(lambda: flex_attention(q, k, v, block_mask=bm))

            # SDPA reference
            sdpa_time = measure(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))

            row = {
                "S": S, "pattern": pname,
                "blockmask_ms": round(blockmask_time, 3),
                "first_call_ms": round(first_call_time, 3),
                "compile_overhead_ms": round(first_call_time - cached_time, 3),
                "cached_ms": round(cached_time, 3),
                "sdpa_ms": round(sdpa_time, 3),
                "flex_vs_sdpa_cached": round(cached_time / sdpa_time, 1) if sdpa_time > 0 else 0,
            }
            results.append(row)
            print(f"  S={S} {pname}: compile={first_call_time - cached_time:.0f}ms "
                  f"cached={cached_time:.1f}ms sdpa={sdpa_time:.1f}ms "
                  f"ratio={cached_time/sdpa_time:.1f}x")

    return results


# ============================================================
# Experiment F3: BlockMask Internal Structure Deep Dive
# ============================================================
def experiment_f3():
    """Inspect BlockMask internal fields for different patterns."""
    print("\n" + "=" * 70)
    print("Experiment F3: BlockMask Internal Structure")
    print("=" * 70)

    S = 2048
    B = 1
    results = {}

    configs = []

    # Causal
    configs.append(("causal", lambda b, h, q, kv: q >= kv))

    # Sliding windows
    for w in [64, 128, 256, 512]:
        def make_sw(ww):
            return lambda b, h, q, kv: (q >= kv) & ((q - kv) <= ww)
        configs.append((f"sw_{w}", make_sw(w)))

    # Document packing
    for nd in [2, 4, 8, 16]:
        doc_ids = torch.arange(S, device=DEVICE) // (S // nd)
        def make_doc(did):
            return lambda b, h, q, kv: (q >= kv) & (did[q] == did[kv])
        configs.append((f"doc_{nd}", make_doc(doc_ids)))

    # Prefix LM
    for pr in [0.1, 0.25, 0.5]:
        pl = int(S * pr)
        def make_prefix(pp):
            return lambda b, h, q, kv: (q >= kv) | (kv < pp)
        configs.append((f"prefix_{int(pr*100)}pct", make_prefix(pl)))

    for name, mask_fn in configs:
        clear_gpu()
        try:
            bm = create_block_mask(mask_fn, B, 1, S, S, device=DEVICE)
            info = {"name": name}

            if hasattr(bm, 'kv_num_blocks'):
                info["kv_num_blocks_shape"] = list(bm.kv_num_blocks.shape)
                info["total_non_empty"] = int(bm.kv_num_blocks.sum().item())
                info["total_blocks"] = (S // 128) ** 2
                info["utilization_pct"] = round(
                    bm.kv_num_blocks.sum().item() / ((S // 128) ** 2) * 100, 1)
                # Per-row analysis
                per_row = bm.kv_num_blocks[0, 0].tolist()
                info["min_blocks_per_row"] = min(per_row)
                info["max_blocks_per_row"] = max(per_row)
                info["mean_blocks_per_row"] = round(sum(per_row) / len(per_row), 1)

            if hasattr(bm, 'kv_indices'):
                info["kv_indices_shape"] = list(bm.kv_indices.shape)
                info["kv_indices_dtype"] = str(bm.kv_indices.dtype)
                info["kv_indices_size_mb"] = round(
                    bm.kv_indices.nelement() * bm.kv_indices.element_size() / 1e6, 3)

            # BlockMask total memory
            total_mem = 0
            for attr_name in ['kv_num_blocks', 'kv_indices', 'full_kv_num_blocks',
                              'full_kv_indices', 'q_num_blocks', 'q_indices']:
                if hasattr(bm, attr_name):
                    tensor = getattr(bm, attr_name)
                    if tensor is not None:
                        total_mem += tensor.nelement() * tensor.element_size()
            info["total_blockmask_bytes"] = total_mem

            # Compare with full SxS mask
            full_mask_bytes = S * S  # bool
            info["compression_ratio"] = round(full_mask_bytes / total_mem, 1) if total_mem > 0 else 0

            results[name] = info
            print(f"  {name}: util={info['utilization_pct']}% "
                  f"blocks={info['total_non_empty']}/{info['total_blocks']} "
                  f"compression={info['compression_ratio']}x")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            results[name] = {"error": str(e)}

    return results


# ============================================================
# Experiment F4: Data Flow Memory Tracking
# ============================================================
def experiment_f4():
    """Track memory allocation at each step of all three paths."""
    print("\n" + "=" * 70)
    print("Experiment F4: Data Flow Memory Tracking")
    print("=" * 70)

    B, H, D = 1, 8, 64
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]
    results = {}

    for S in seq_lengths:
        row = {"S": S}
        q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
        v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)

        # Vanilla: track step by step
        clear_gpu()
        mem_steps = {}
        mem_steps["input"] = get_peak_memory()
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
        mem_steps["after_qkt"] = get_peak_memory()
        causal = torch.ones(S, S, device=DEVICE, dtype=torch.bool).tril_()
        mem_steps["after_causal"] = get_peak_memory()
        scores = scores.masked_fill(~causal, float('-inf'))
        mem_steps["after_mask"] = get_peak_memory()
        weights = F.softmax(scores.float(), dim=-1).to(DTYPE)
        mem_steps["after_softmax"] = get_peak_memory()
        out = torch.matmul(weights, v)
        mem_steps["after_xv"] = get_peak_memory()
        row["vanilla_steps"] = {k: round(v, 4) for k, v in mem_steps.items()}

        # SDPA
        clear_gpu()
        out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        row["sdpa_peak_gb"] = round(get_peak_memory(), 4)

        # Flex
        clear_gpu()
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        bm = create_block_mask(causal_mask, B, 1, S, S, device=DEVICE)
        row["flex_blockmask_gb"] = round(get_peak_memory(), 4)
        out_flex = flex_attention(q, k, v, block_mask=bm)
        row["flex_peak_gb"] = round(get_peak_memory(), 4)

        results[S] = row
        print(f"  S={S}: Vanilla peak={mem_steps['after_xv']:.3f}GB "
              f"SDPA={row['sdpa_peak_gb']:.3f}GB Flex={row['flex_peak_gb']:.3f}GB")

        del q, k, v
        clear_gpu()

    return results


# ============================================================
# Experiment F5: SDPA Backend Comparison
# ============================================================
def experiment_f5():
    """Compare all SDPA backends: Flash2, Math, Efficient."""
    print("\n" + "=" * 70)
    print("Experiment F5: SDPA Backend Comparison")
    print("=" * 70)

    B, H, D = 1, 8, 64
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    results = []

    backends = [
        ("flash2", torch.nn.attention.SDPBackend.FLASH_ATTENTION),
        ("math", torch.nn.attention.SDPBackend.MATH),
        ("efficient", torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION),
    ]

    for S in seq_lengths:
        clear_gpu()
        try:
            q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
        except Exception:
            break

        row = {"S": S}
        for bname, backend in backends:
            clear_gpu()
            try:
                with torch.nn.attention.sdpa_kernel([backend]):
                    t = measure(lambda: F.scaled_dot_product_attention(
                        q, k, v, is_causal=True))
                    m = get_peak_memory()
                    row[f"{bname}_ms"] = round(t, 3)
                    row[f"{bname}_gb"] = round(m, 3)
            except Exception as e:
                row[f"{bname}_ms"] = -1
                row[f"{bname}_gb"] = -1
                print(f"  {bname} failed at S={S}: {e}")

        results.append(row)
        parts = [f"{bn}={row.get(f'{bn}_ms', '?')}ms" for bn, _ in backends]
        print(f"  S={S}: {', '.join(parts)}")

        del q, k, v
        clear_gpu()

    return results


# ============================================================
# Experiment F6: Comprehensive Three-Way Benchmark
# ============================================================
def experiment_f6():
    """Full benchmark: Vanilla vs SDPA vs Flex for multiple patterns."""
    print("\n" + "=" * 70)
    print("Experiment F6: Comprehensive Three-Way Benchmark")
    print("=" * 70)

    B, H, D = 1, 8, 64
    seq_lengths = [512, 1024, 2048, 4096]
    results = []

    patterns = {
        "causal": {
            "vanilla": lambda q, k, v: (
                torch.matmul(
                    F.softmax(
                        (torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)).masked_fill(
                            ~torch.ones(q.shape[-2], q.shape[-2], device=DEVICE, dtype=torch.bool).tril_(),
                            float('-inf')
                        ).float(), dim=-1
                    ).to(q.dtype), v
                )
            ),
            "sdpa": lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
            "flex_mask": lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
        },
    }

    # Add document packing
    for nd in [4, 8]:
        def make_doc_vanilla(n_docs):
            def fn(q, k, v):
                S = q.shape[-2]
                doc_ids = torch.arange(S, device=DEVICE) // (S // n_docs)
                scores = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
                causal = torch.ones(S, S, device=DEVICE, dtype=torch.bool).tril_()
                doc = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
                scores = scores.masked_fill(~(causal & doc), float('-inf'))
                return torch.matmul(F.softmax(scores.float(), dim=-1).to(q.dtype), v)
            return fn
        def make_doc_flex_mask(n_docs):
            def fn(b, h, q_idx, kv_idx):
                S_val = 2048
                doc_ids = torch.arange(S_val, device=DEVICE) // (S_val // n_docs)
                return (q_idx >= kv_idx) & (doc_ids[q_idx] == doc_ids[kv_idx])
            return fn

        patterns[f"doc_{nd}"] = {
            "vanilla": make_doc_vanilla(nd),
            "sdpa": None,  # SDPA cannot do doc packing
            "flex_mask": make_doc_flex_mask(nd),
        }

    for S in seq_lengths:
        for pname, pconfig in patterns.items():
            clear_gpu()
            q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)

            row = {"S": S, "pattern": pname}

            # Vanilla
            clear_gpu()
            try:
                t = measure(lambda: pconfig["vanilla"](q, k, v))
                m = get_peak_memory()
                row["vanilla_ms"] = round(t, 3)
                row["vanilla_gb"] = round(m, 3)
            except Exception:
                row["vanilla_ms"] = -1
                row["vanilla_gb"] = -1

            # SDPA
            if pconfig["sdpa"] is not None:
                clear_gpu()
                try:
                    t = measure(lambda: pconfig["sdpa"](q, k, v))
                    m = get_peak_memory()
                    row["sdpa_ms"] = round(t, 3)
                    row["sdpa_gb"] = round(m, 3)
                except Exception:
                    row["sdpa_ms"] = -1
                    row["sdpa_gb"] = -1
            else:
                row["sdpa_ms"] = -2  # Not supported
                row["sdpa_gb"] = -2

            # Flex
            clear_gpu()
            try:
                bm = create_block_mask(pconfig["flex_mask"], B, 1, S, S, device=DEVICE)
                _ = flex_attention(q, k, v, block_mask=bm)
                torch.cuda.synchronize()
                t = measure(lambda: flex_attention(q, k, v, block_mask=bm))
                m = get_peak_memory()
                row["flex_ms"] = round(t, 3)
                row["flex_gb"] = round(m, 3)
            except Exception:
                row["flex_ms"] = -1
                row["flex_gb"] = -1

            results.append(row)
            sdpa_str = f"SDPA={row.get('sdpa_ms', '?')}ms" if row.get("sdpa_ms", -2) != -2 else "SDPA=N/A"
            print(f"  S={S} {pname}: Vanilla={row.get('vanilla_ms', '?')}ms "
                  f"{sdpa_str} Flex={row.get('flex_ms', '?')}ms")

            del q, k, v
            clear_gpu()

    return results


# ============================================================
# Experiment F7: Autograd Profiling
# ============================================================
def experiment_f7():
    """Profile forward and backward passes for all three paths."""
    print("\n" + "=" * 70)
    print("Experiment F7: Autograd Profiling")
    print("=" * 70)

    B, H, S, D = 1, 8, 2048, 64
    results = {}

    # Causal for all three paths
    for path_name in ["vanilla", "sdpa", "flex"]:
        clear_gpu()
        q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
        k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
        v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE, requires_grad=True)

        info = {}

        if path_name == "vanilla":
            def fwd():
                scores = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
                causal = torch.ones(S, S, device=DEVICE, dtype=torch.bool).tril_()
                scores = scores.masked_fill(~causal, float('-inf'))
                return torch.matmul(F.softmax(scores.float(), dim=-1).to(DTYPE), v)
        elif path_name == "sdpa":
            def fwd():
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            def causal_mask(b, h, qi, kvi):
                return qi >= kvi
            bm = create_block_mask(causal_mask, B, 1, S, S, device=DEVICE)
            def fwd():
                return flex_attention(q, k, v, block_mask=bm)
            # Warmup for compile
            _ = fwd()
            torch.cuda.synchronize()

        # Forward
        clear_gpu()
        fwd_time = measure(fwd)
        fwd_mem = get_peak_memory()
        info["fwd_ms"] = round(fwd_time, 3)
        info["fwd_gb"] = round(fwd_mem, 3)

        # Backward
        out = fwd()
        clear_gpu()
        bwd_time = measure(lambda: fwd().sum().backward())
        bwd_mem = get_peak_memory()
        info["bwd_ms"] = round(bwd_time, 3)
        info["bwd_gb"] = round(bwd_mem, 3)
        info["fwd_bwd_ratio"] = round(bwd_time / fwd_time, 2) if fwd_time > 0 else 0

        # Total
        info["total_ms"] = round(fwd_time + bwd_time, 3)

        results[path_name] = info
        print(f"  {path_name}: fwd={fwd_time:.1f}ms bwd={bwd_time:.1f}ms "
              f"total={fwd_time+bwd_time:.1f}ms ratio={bwd_time/fwd_time:.2f}x")

    return results


# ============================================================
# Experiment F8: Numerical Precision Deep Dive
# ============================================================
def experiment_f8():
    """Compare numerical precision across paths and data types."""
    print("\n" + "=" * 70)
    print("Experiment F8: Numerical Precision Deep Dive")
    print("=" * 70)

    B, H, S = 1, 8, 2048
    results = []

    for D in [32, 64, 128]:
        for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32),
                                   ("bf16", torch.bfloat16)]:
            clear_gpu()
            try:
                q = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
                k = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
                v = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
            except Exception:
                continue

            # Vanilla (in fp32 for reference)
            q32 = q.float()
            k32 = k.float()
            v32 = v.float()
            scores = torch.matmul(q32, k32.transpose(-2, -1)) / (D**0.5)
            causal = torch.ones(S, S, device=DEVICE, dtype=torch.bool).tril_()
            scores = scores.masked_fill(~causal, float('-inf'))
            ref = torch.matmul(F.softmax(scores.float(), dim=-1), v32)

            # Vanilla in native dtype
            scores2 = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
            scores2 = scores2.masked_fill(~causal, float('-inf'))
            vanilla_out = torch.matmul(F.softmax(scores2.float(), dim=-1).to(dtype), v)

            # SDPA
            sdpa_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

            # Flex
            def causal_mask(b, h, qi, kvi):
                return qi >= kvi
            bm = create_block_mask(causal_mask, B, 1, S, S, device=DEVICE)
            flex_out = flex_attention(q, k, v, block_mask=bm)

            # Compare against fp32 reference
            row = {
                "D": D, "dtype": dtype_name,
                "vanilla_vs_ref": round((ref - vanilla_out.float()).abs().max().item(), 6),
                "sdpa_vs_ref": round((ref - sdpa_out.float()).abs().max().item(), 6),
                "flex_vs_ref": round((ref - flex_out.float()).abs().max().item(), 6),
                "vanilla_vs_flex": round((vanilla_out - flex_out).abs().max().item(), 6),
                "sdpa_vs_flex": round((sdpa_out - flex_out).abs().max().item(), 6),
            }
            results.append(row)
            print(f"  D={D} {dtype_name}: VvsRef={row['vanilla_vs_ref']:.6f} "
                  f"SvsRef={row['sdpa_vs_ref']:.6f} FvsRef={row['flex_vs_ref']:.6f}")

    return results


# ============================================================
# Chart Generation
# ============================================================
def generate_charts(f1, f2, f3, f4, f5, f6, f7, f8):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

    # ---- Chart F1: Three-Path Trace ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Vanilla steps
    steps = f1["path_vanilla"]
    step_names = ["QK^T", "Causal Mask", "Apply Mask", "Softmax", "x V"]
    step_times = [steps["1_qkt"], steps["2_causal"], steps["3_mask_apply"],
                  steps["4_softmax"], steps["5_xv"]]
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']
    axes[0].bar(step_names, step_times, color=colors)
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Vanilla Path: Per-Step Latency (S=2048)')
    axes[0].grid(axis='y', alpha=0.3)
    for i, t in enumerate(step_times):
        axes[0].text(i, t + 0.01, f'{t:.3f}', ha='center', fontsize=8)

    # Three paths comparison
    paths = ["Vanilla\n(5 kernels)", "SDPA/Flash2\n(1 kernel)", "Flex\n(1 kernel)"]
    vanilla_total = sum(step_times)
    sdpa_time = f1["path_sdpa"].get("flash2_ms", f1["path_sdpa"]["1_total"])
    flex_time = f1["path_flex"]["3_cached_ms"]
    path_times = [vanilla_total, sdpa_time, flex_time]
    colors2 = ['#e74c3c', '#2ecc71', '#3498db']
    axes[1].bar(paths, path_times, color=colors2)
    axes[1].set_ylabel('Total Time (ms)')
    axes[1].set_title('Three Paths: Total Latency (S=2048)')
    axes[1].grid(axis='y', alpha=0.3)
    for i, t in enumerate(path_times):
        axes[1].text(i, t + 0.01, f'{t:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/F1_three_path_trace.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/F1_three_path_trace.png")

    # ---- Chart F2: Compilation Overhead ----
    fig, ax = plt.subplots(figsize=(12, 6))
    patterns = list(set(r["pattern"] for r in f2))
    for pi, pname in enumerate(patterns):
        sub = [r for r in f2 if r["pattern"] == pname and r.get("first_call_ms", 0) > 0]
        if not sub:
            continue
        s_vals = [r["S"] for r in sub]
        compile_overhead = [r.get("compile_overhead_ms", 0) for r in sub]
        cached = [r.get("cached_ms", 0) for r in sub]
        ax.plot(s_vals, compile_overhead, marker='o', label=f'{pname} (compile overhead)')
        ax.plot(s_vals, cached, marker='s', linestyle='--', label=f'{pname} (cached)')
    ax.set_xlabel('Sequence Length (S)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('FlexAttention: Compilation Overhead vs Cached Performance')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xscale('log', base=2)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/F2_compile_overhead.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/F2_compile_overhead.png")

    # ---- Chart F3: BlockMask Structure ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    names = [n for n in f3 if "error" not in f3[n]]
    utils = [f3[n]["utilization_pct"] for n in names]
    compressions = [f3[n]["compression_ratio"] for n in names]

    colors = plt.cm.RdYlGn_r([u / 100 for u in utils])
    axes[0].bar(range(len(names)), utils, color=colors)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    axes[0].set_ylabel('Block Utilization (%)')
    axes[0].set_title('BlockMask Utilization by Pattern (S=2048)')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(range(len(names)), compressions,
                color=['#2ecc71' if c > 100 else '#3498db' for c in compressions])
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    axes[1].set_ylabel('Compression Ratio (x)')
    axes[1].set_title('BlockMask Compression vs Full SxS Mask')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/F3_blockmask_structure.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/F3_blockmask_structure.png")

    # ---- Chart F4: Memory Waterfall ----
    fig, ax = plt.subplots(figsize=(12, 6))
    for S_str, data in f4.items():
        steps_data = data.get("vanilla_steps", {})
        if not steps_data:
            continue
        step_labels = list(steps_data.keys())
        step_mems = list(steps_data.values())
        ax.plot(step_labels, step_mems, marker='o', label=f'S={S_str}')
    ax.set_ylabel('Cumulative Peak Memory (GB)')
    ax.set_title('Vanilla Path: Memory Build-Up at Each Step')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/F4_memory_waterfall.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/F4_memory_waterfall.png")

    # ---- Chart F5: SDPA Backends ----
    if f5:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for bname in ["flash2", "math", "efficient"]:
            s_vals = [r["S"] for r in f5 if r.get(f"{bname}_ms", -1) > 0]
            times = [r[f"{bname}_ms"] for r in f5 if r.get(f"{bname}_ms", -1) > 0]
            mems = [r[f"{bname}_gb"] for r in f5 if r.get(f"{bname}_ms", -1) > 0]
            if times:
                axes[0].plot(s_vals, times, marker='o', label=bname)
                axes[1].plot(s_vals, mems, marker='o', label=bname)
        axes[0].set_xlabel('Sequence Length (S)')
        axes[0].set_ylabel('Latency (ms)')
        axes[0].set_title('SDPA Backend Latency')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_xscale('log', base=2)
        axes[0].set_yscale('log')
        axes[1].set_xlabel('Sequence Length (S)')
        axes[1].set_ylabel('Peak Memory (GB)')
        axes[1].set_title('SDPA Backend Memory')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_xscale('log', base=2)
        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/F5_sdpa_backends.png', bbox_inches='tight')
        plt.close()
        print(f"Saved {FIGURES_DIR}/F5_sdpa_backends.png")

    # ---- Chart F6: Three-Way Benchmark ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for pname in set(r["pattern"] for r in f6):
        for impl, color, marker in [("vanilla", "#e74c3c", "o"),
                                     ("sdpa", "#2ecc71", "s"),
                                     ("flex", "#3498db", "^")]:
            sub = [r for r in f6 if r["pattern"] == pname and r.get(f"{impl}_ms", -2) > 0]
            if not sub:
                continue
            s_vals = [r["S"] for r in sub]
            times = [r[f"{impl}_ms"] for r in sub]
            label = f'{pname} ({impl})'
            axes[0].plot(s_vals, times, marker=marker, color=color, label=label, alpha=0.7)
    axes[0].set_xlabel('Sequence Length (S)')
    axes[0].set_ylabel('Latency (ms)')
    axes[0].set_title('Three-Way Latency Comparison')
    axes[0].legend(fontsize=7, loc='upper left')
    axes[0].grid(alpha=0.3)
    axes[0].set_xscale('log', base=2)
    axes[0].set_yscale('log')

    for pname in set(r["pattern"] for r in f6):
        for impl, color, marker in [("vanilla", "#e74c3c", "o"),
                                     ("sdpa", "#2ecc71", "s"),
                                     ("flex", "#3498db", "^")]:
            sub = [r for r in f6 if r["pattern"] == pname and r.get(f"{impl}_gb", -2) > 0]
            if not sub:
                continue
            s_vals = [r["S"] for r in sub]
            mems = [r[f"{impl}_gb"] for r in sub]
            label = f'{pname} ({impl})'
            axes[1].plot(s_vals, mems, marker=marker, color=color, label=label, alpha=0.7)
    axes[1].set_xlabel('Sequence Length (S)')
    axes[1].set_ylabel('Peak Memory (GB)')
    axes[1].set_title('Three-Way Memory Comparison')
    axes[1].legend(fontsize=7, loc='upper left')
    axes[1].grid(alpha=0.3)
    axes[1].set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/F6_three_way_benchmark.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/F6_three_way_benchmark.png")

    # ---- Chart F7: Autograd ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    path_names = list(f7.keys())
    fwd_times = [f7[p]["fwd_ms"] for p in path_names]
    bwd_times = [f7[p]["bwd_ms"] for p in path_names]
    totals = [f7[p]["total_ms"] for p in path_names]
    colors3 = ['#e74c3c', '#2ecc71', '#3498db']

    axes[0].bar(path_names, fwd_times, color=colors3)
    axes[0].set_title('Forward Pass Time')
    axes[0].set_ylabel('Time (ms)')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(path_names, bwd_times, color=colors3)
    axes[1].set_title('Backward Pass Time')
    axes[1].set_ylabel('Time (ms)')
    axes[1].grid(axis='y', alpha=0.3)

    # Stacked bar
    x = range(len(path_names))
    axes[2].bar(x, fwd_times, color=colors3, label='Forward')
    axes[2].bar(x, bwd_times, bottom=fwd_times, color=[c + '80' for c in ['#e74c3c', '#2ecc71', '#3498db']], label='Backward')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(path_names)
    axes[2].set_title('Forward + Backward (stacked)')
    axes[2].set_ylabel('Time (ms)')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/F7_autograd.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/F7_autograd.png")

    # ---- Chart F8: Numerical Precision ----
    fig, ax = plt.subplots(figsize=(12, 6))
    if f8:
        labels = [f"D={r['D']},{r['dtype']}" for r in f8]
        x = range(len(labels))
        width = 0.25
        ax.bar([i - width for i in x], [r["vanilla_vs_ref"] for r in f8],
               width, label='Vanilla vs FP32 ref', color='#e74c3c')
        ax.bar([i for i in x], [r["sdpa_vs_ref"] for r in f8],
               width, label='SDPA vs FP32 ref', color='#2ecc71')
        ax.bar([i + width for i in x], [r["flex_vs_ref"] for r in f8],
               width, label='Flex vs FP32 ref', color='#3498db')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Max Absolute Difference')
        ax.set_title('Numerical Precision: All Paths vs FP32 Reference')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/F8_precision.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/F8_precision.png")

    print("\nAll source analysis charts saved!")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("FlexAttention Source Code Analysis: Three Attention Paths")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    all_results = {}

    all_results["F1"] = experiment_f1()
    all_results["F2"] = experiment_f2()
    all_results["F3"] = experiment_f3()
    all_results["F4"] = experiment_f4()
    all_results["F5"] = experiment_f5()
    all_results["F6"] = experiment_f6()
    all_results["F7"] = experiment_f7()
    all_results["F8"] = experiment_f8()

    generate_charts(
        all_results["F1"], all_results["F2"], all_results["F3"],
        all_results["F4"], all_results["F5"], all_results["F6"],
        all_results["F7"], all_results["F8"]
    )

    with open("source_analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to source_analysis_results.json")
    print("DONE!")
