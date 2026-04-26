#!/usr/bin/env python3
"""
Comprehensive Attention Pattern Analysis: PyTorch Vanilla vs FlexAttention
==========================================================================
Tests 8 attention patterns from attention-gym with detailed comparison.
Each pattern: Vanilla PyTorch implementation vs FlexAttention implementation.
Metrics: latency, memory, numerical accuracy, sparsity.

All chart text is pure ASCII. Reports will be in Chinese.

NVIDIA L4 (24GB) | PyTorch 2.6.0+cu124 | Triton 3.2.0
"""

import torch
import torch.nn.functional as F
import time
import json
import gc
import os
import math
from collections import defaultdict

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# ============================================================
# Configuration
# ============================================================
DEVICE = "cuda"
DTYPE = torch.float16
FIGURES_DIR = "figures_patterns"
os.makedirs(FIGURES_DIR, exist_ok=True)

torch.manual_seed(42)

# ============================================================
# Utility Functions
# ============================================================
def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def measure(func, warmup=3, repeat=10):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        func()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeat
    return elapsed * 1000  # ms

def get_peak_memory():
    return torch.cuda.max_memory_allocated() / 1e9  # GB

# ============================================================
# Pattern 1: Standard Causal Attention
# ============================================================
def vanilla_causal(q, k, v, **kwargs):
    S = q.shape[-2]
    D = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
    causal = torch.ones(S, S, device=q.device, dtype=torch.bool).tril_()
    scores = scores.masked_fill(~causal, float('-inf'))
    weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(weights, v)

def flex_causal(q, k, v, **kwargs):
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    S = q.shape[-2]
    B = q.shape[0]
    block_mask = create_block_mask(causal_mask, B, 1, S, S, device=q.device)
    return flex_attention(q, k, v, block_mask=block_mask)

# ============================================================
# Pattern 2: Sliding Window Attention
# ============================================================
def vanilla_sliding_window(q, k, v, window_size=256, **kwargs):
    S = q.shape[-2]
    D = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
    pos = torch.arange(S, device=q.device)
    causal = pos.unsqueeze(0) >= pos.unsqueeze(1)
    window = (pos.unsqueeze(0) - pos.unsqueeze(1)) <= window_size
    scores = scores.masked_fill(~(causal & window), float('-inf'))
    weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(weights, v)

def flex_sliding_window(q, k, v, window_size=256, **kwargs):
    def sw_mask(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        window = (q_idx - kv_idx) <= window_size
        return causal & window
    B, _, S, _ = q.shape
    block_mask = create_block_mask(sw_mask, B, 1, S, S, device=q.device)
    return flex_attention(q, k, v, block_mask=block_mask)

# ============================================================
# Pattern 3: Prefix LM
# ============================================================
def vanilla_prefix_lm(q, k, v, prefix_ratio=0.25, **kwargs):
    S = q.shape[-2]
    D = q.shape[-1]
    prefix_len = int(S * prefix_ratio)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
    pos = torch.arange(S, device=q.device)
    causal = pos.unsqueeze(0) >= pos.unsqueeze(1)
    prefix = pos.unsqueeze(1) < prefix_len  # kv in prefix -> all rows can see
    prefix_rows = pos.unsqueeze(0) < prefix_len  # q in prefix -> see all kv
    mask = causal | prefix | prefix_rows
    # Correct: prefix region is bidirectional, non-prefix is causal
    # All tokens can attend to prefix tokens; non-prefix tokens are causal among themselves
    prefix_mask = pos.unsqueeze(1) < prefix_len  # kv_idx < prefix_len
    mask = causal | prefix_mask
    scores = scores.masked_fill(~mask, float('-inf'))
    weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(weights, v)

def flex_prefix_lm(q, k, v, prefix_ratio=0.25, **kwargs):
    S = q.shape[-2]
    prefix_len = int(S * prefix_ratio)
    def prefix_mask_fn(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        prefix = kv_idx < prefix_len
        return causal | prefix
    B = q.shape[0]
    block_mask = create_block_mask(prefix_mask_fn, B, 1, S, S, device=q.device)
    return flex_attention(q, k, v, block_mask=block_mask)

# ============================================================
# Pattern 4: ALiBi (Attention with Linear Biases)
# ============================================================
def vanilla_alibi(q, k, v, **kwargs):
    B, H, S, D = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
    # ALiBi slopes per head
    slopes = torch.tensor([2 ** (-8 * (h + 1) / H) for h in range(H)],
                          device=q.device, dtype=q.dtype)
    pos = torch.arange(S, device=q.device)
    dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().float()
    causal = pos.unsqueeze(0) >= pos.unsqueeze(1)
    for h in range(H):
        scores[:, h] -= slopes[h] * dist
    scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))
    weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(weights, v)

def flex_alibi(q, k, v, **kwargs):
    B, H, S, D = q.shape
    slopes = torch.tensor([2 ** (-8 * (h + 1) / H) for h in range(H)],
                          device=q.device, dtype=q.dtype)
    def alibi_score(score, b, h, q_idx, kv_idx):
        return score - slopes[h] * (q_idx - kv_idx).abs()
    # For ALiBi we use score_mod, not mask_mod
    # Still need causal mask
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    block_mask = create_block_mask(causal_mask, B, 1, S, S, device=q.device)
    return flex_attention(q, k, v, score_mod=alibi_score, block_mask=block_mask)

# ============================================================
# Pattern 5: Tanh Softcapping
# ============================================================
def vanilla_softcap(q, k, v, soft_cap=50.0, **kwargs):
    S = q.shape[-2]
    D = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
    causal = torch.ones(S, S, device=q.device, dtype=torch.bool).tril_()
    scores = scores.masked_fill(~causal, float('-inf'))
    # Apply tanh softcapping BEFORE softmax
    scores = soft_cap * torch.tanh(scores / soft_cap)
    weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(weights, v)

def flex_softcap(q, k, v, soft_cap=50.0, **kwargs):
    B, _, S, _ = q.shape
    def softcap_score(score, b, h, q_idx, kv_idx):
        return soft_cap * torch.tanh(score / soft_cap)
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    block_mask = create_block_mask(causal_mask, B, 1, S, S, device=q.device)
    return flex_attention(q, k, v, score_mod=softcap_score, block_mask=block_mask)

# ============================================================
# Pattern 6: Document Packing + Causal
# ============================================================
def vanilla_doc_packing(q, k, v, doc_ids, **kwargs):
    S = q.shape[-2]
    D = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
    causal = torch.ones(S, S, device=q.device, dtype=torch.bool).tril_()
    doc = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
    scores = scores.masked_fill(~(causal & doc), float('-inf'))
    weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(weights, v)

def flex_doc_packing(q, k, v, doc_ids, **kwargs):
    B, _, S, _ = q.shape
    def doc_causal(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (doc_ids[q_idx] == doc_ids[kv_idx])
    block_mask = create_block_mask(doc_causal, B, 1, S, S, device=q.device)
    return flex_attention(q, k, v, block_mask=block_mask)

# ============================================================
# Pattern 7: Dilated Sliding Window
# ============================================================
def vanilla_dilated_sw(q, k, v, window_size=256, dilation=4, **kwargs):
    S = q.shape[-2]
    D = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
    pos = torch.arange(S, device=q.device)
    causal = pos.unsqueeze(0) >= pos.unsqueeze(1)
    window = (pos.unsqueeze(0) - pos.unsqueeze(1)) <= window_size
    dilated = ((pos.unsqueeze(0) - pos.unsqueeze(1)) % dilation) == 0
    scores = scores.masked_fill(~(causal & window & dilated), float('-inf'))
    weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(weights, v)

def flex_dilated_sw(q, k, v, window_size=256, dilation=4, **kwargs):
    B, _, S, _ = q.shape
    def dilated_sw_mask(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        window = (q_idx - kv_idx) <= window_size
        dilated = ((q_idx - kv_idx) % dilation) == 0
        return causal & window & dilated
    block_mask = create_block_mask(dilated_sw_mask, B, 1, S, S, device=q.device)
    return flex_attention(q, k, v, block_mask=block_mask)

# ============================================================
# Pattern 8: Combined (Document + Sliding Window + ALiBi)
# ============================================================
def vanilla_combined(q, k, v, doc_ids, window_size=256, **kwargs):
    B, H, S, D = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
    pos = torch.arange(S, device=q.device)
    causal = pos.unsqueeze(0) >= pos.unsqueeze(1)
    window = (pos.unsqueeze(0) - pos.unsqueeze(1)) <= window_size
    doc = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
    mask = causal & window & doc
    slopes = torch.tensor([2 ** (-8 * (h + 1) / H) for h in range(H)],
                          device=q.device, dtype=q.dtype)
    dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().float()
    for h in range(H):
        scores[:, h] -= slopes[h] * dist
    scores = scores.masked_fill(~mask, float('-inf'))
    weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(weights, v)

def flex_combined(q, k, v, doc_ids, window_size=256, **kwargs):
    B, H, S, D = q.shape
    slopes = torch.tensor([2 ** (-8 * (h + 1) / H) for h in range(H)],
                          device=q.device, dtype=q.dtype)
    def combined_mask(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        window = (q_idx - kv_idx) <= window_size
        doc = doc_ids[q_idx] == doc_ids[kv_idx]
        return causal & window & doc
    def alibi_score(score, b, h, q_idx, kv_idx):
        return score - slopes[h] * (q_idx - kv_idx).abs()
    block_mask = create_block_mask(combined_mask, B, 1, S, S, device=q.device)
    return flex_attention(q, k, v, score_mod=alibi_score, block_mask=block_mask)


# ============================================================
# Pattern Registry
# ============================================================
PATTERNS = {
    "causal": {
        "name": "Causal Attention",
        "vanilla_fn": vanilla_causal,
        "flex_fn": flex_causal,
        "needs_doc_ids": False,
        "has_score_mod": False,
    },
    "sliding_window": {
        "name": "Sliding Window (256)",
        "vanilla_fn": vanilla_sliding_window,
        "flex_fn": flex_sliding_window,
        "needs_doc_ids": False,
        "has_score_mod": False,
    },
    "prefix_lm": {
        "name": "Prefix LM (25%)",
        "vanilla_fn": vanilla_prefix_lm,
        "flex_fn": flex_prefix_lm,
        "needs_doc_ids": False,
        "has_score_mod": False,
    },
    "alibi": {
        "name": "ALiBi + Causal",
        "vanilla_fn": vanilla_alibi,
        "flex_fn": flex_alibi,
        "needs_doc_ids": False,
        "has_score_mod": True,
    },
    "softcap": {
        "name": "Tanh Softcapping",
        "vanilla_fn": vanilla_softcap,
        "flex_fn": flex_softcap,
        "needs_doc_ids": False,
        "has_score_mod": True,
    },
    "doc_packing": {
        "name": "Document Packing (4 docs)",
        "vanilla_fn": vanilla_doc_packing,
        "flex_fn": flex_doc_packing,
        "needs_doc_ids": True,
        "has_score_mod": False,
    },
    "dilated_sw": {
        "name": "Dilated SW (w=256,d=4)",
        "vanilla_fn": vanilla_dilated_sw,
        "flex_fn": flex_dilated_sw,
        "needs_doc_ids": False,
        "has_score_mod": False,
    },
    "combined": {
        "name": "Doc+SW+ALiBi (Ultimate)",
        "vanilla_fn": vanilla_combined,
        "flex_fn": flex_combined,
        "needs_doc_ids": True,
        "has_score_mod": True,
    },
}

# ============================================================
# Experiment E1: Full Pattern Comparison (Vanilla vs Flex)
# ============================================================
def experiment_e1():
    print("=" * 70)
    print("Experiment E1: Full Pattern Comparison (Vanilla vs Flex)")
    print("=" * 70)

    B, H, D = 1, 8, 64
    seq_lengths = [256, 512, 1024, 2048, 4096]
    results = {}

    for pattern_key, pattern_info in PATTERNS.items():
        print(f"\n--- {pattern_info['name']} ---")
        results[pattern_key] = []

        for S in seq_lengths:
            clear_gpu()
            q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
            k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
            v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE, requires_grad=True)

            kwargs = {}
            if pattern_info["needs_doc_ids"]:
                num_docs = 4
                kwargs["doc_ids"] = torch.arange(S, device=DEVICE) // (S // num_docs)

            # Vanilla
            clear_gpu()
            try:
                q_v = q.clone().detach().requires_grad_(True)
                k_v = k.clone().detach().requires_grad_(True)
                v_v = v.clone().detach().requires_grad_(True)
                out_v = pattern_info["vanilla_fn"](q_v, k_v, v_v, **kwargs)
                vanilla_mem = get_peak_memory()
                vanilla_time_fwd = measure(lambda: pattern_info["vanilla_fn"](
                    q_v, k_v, v_v, **kwargs))
                # Backward
                loss_v = out_v.sum()
                loss_v.backward()
                vanilla_time_bwd = measure(lambda: (
                    pattern_info["vanilla_fn"](q_v, k_v, v_v, **kwargs).sum().backward()))
                vanilla_success = True
            except Exception as e:
                print(f"  Vanilla OOM at S={S}: {e}")
                vanilla_mem = vanilla_time_fwd = vanilla_time_bwd = -1
                vanilla_success = False

            # Flex
            clear_gpu()
            try:
                q_f = q.clone().detach().requires_grad_(True)
                k_f = k.clone().detach().requires_grad_(True)
                v_f = v.clone().detach().requires_grad_(True)
                out_f = pattern_info["flex_fn"](q_f, k_f, v_f, **kwargs)
                flex_mem = get_peak_memory()
                flex_time_fwd = measure(lambda: pattern_info["flex_fn"](
                    q_f, k_f, v_f, **kwargs))
                # Backward
                loss_f = out_f.sum()
                loss_f.backward()
                flex_time_bwd = measure(lambda: (
                    pattern_info["flex_fn"](q_f, k_f, v_f, **kwargs).sum().backward()))
                flex_success = True
            except Exception as e:
                print(f"  Flex failed at S={S}: {e}")
                flex_mem = flex_time_fwd = flex_time_bwd = -1
                flex_success = False

            # Numerical accuracy
            max_diff = -1
            if vanilla_success and flex_success:
                clear_gpu()
                q_t = q.clone().detach().requires_grad_(True)
                k_t = k.clone().detach().requires_grad_(True)
                v_t = v.clone().detach().requires_grad_(True)
                out_v2 = pattern_info["vanilla_fn"](q_t, k_t, v_t, **kwargs)
                q_t2 = q.clone().detach().requires_grad_(True)
                k_t2 = k.clone().detach().requires_grad_(True)
                v_t2 = v.clone().detach().requires_grad_(True)
                out_f2 = pattern_info["flex_fn"](q_t2, k_t2, v_t2, **kwargs)
                max_diff = (out_v2 - out_f2).abs().max().item()

            row = {
                "S": S, "B": B, "H": H, "D": D,
                "vanilla_fwd_ms": round(vanilla_time_fwd, 3) if vanilla_success else -1,
                "vanilla_bwd_ms": round(vanilla_time_bwd, 3) if vanilla_success else -1,
                "vanilla_mem_gb": round(vanilla_mem, 3) if vanilla_success else -1,
                "flex_fwd_ms": round(flex_time_fwd, 3) if flex_success else -1,
                "flex_bwd_ms": round(flex_time_bwd, 3) if flex_success else -1,
                "flex_mem_gb": round(flex_mem, 3) if flex_success else -1,
                "max_diff": round(max_diff, 6) if max_diff >= 0 else -1,
            }
            results[pattern_key].append(row)
            print(f"  S={S}: Vanilla({vanilla_time_fwd:.1f}ms, {vanilla_mem:.3f}GB) "
                  f"Flex({flex_time_fwd:.1f}ms, {flex_mem:.3f}GB) diff={max_diff:.6f}")

            del q, k, v
            clear_gpu()

    return results


# ============================================================
# Experiment E2: Scalability Deep Dive (Sweep S up to 16384)
# ============================================================
def experiment_e2():
    print("\n" + "=" * 70)
    print("Experiment E2: Scalability Deep Dive")
    print("=" * 70)

    B, H, D = 1, 8, 64
    seq_lengths = [512, 1024, 2048, 4096, 8192, 12288, 16384]
    patterns_to_test = ["causal", "sliding_window", "doc_packing", "alibi", "combined"]
    results = {}

    for pattern_key in patterns_to_test:
        pattern_info = PATTERNS[pattern_key]
        print(f"\n--- {pattern_info['name']} ---")
        results[pattern_key] = []

        for S in seq_lengths:
            clear_gpu()
            try:
                q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
                k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
                v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            except Exception:
                break

            kwargs = {}
            if pattern_info["needs_doc_ids"]:
                num_docs = max(2, S // 1024)
                kwargs["doc_ids"] = torch.arange(S, device=DEVICE) // (S // num_docs)

            # Vanilla
            clear_gpu()
            try:
                t0 = time.perf_counter()
                out_v = pattern_info["vanilla_fn"](q, k, v, **kwargs)
                torch.cuda.synchronize()
                vanilla_time = (time.perf_counter() - t0) * 1000
                vanilla_mem = get_peak_memory()
                vanilla_ok = True
            except Exception as e:
                print(f"  Vanilla OOM at S={S}")
                vanilla_time = vanilla_mem = -1
                vanilla_ok = False

            # Flex
            clear_gpu()
            try:
                # Warmup for compile
                _ = pattern_info["flex_fn"](q, k, v, **kwargs)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out_f = pattern_info["flex_fn"](q, k, v, **kwargs)
                torch.cuda.synchronize()
                flex_time = (time.perf_counter() - t0) * 1000
                flex_mem = get_peak_memory()
                flex_ok = True
            except Exception as e:
                print(f"  Flex OOM at S={S}: {e}")
                flex_time = flex_mem = -1
                flex_ok = False

            row = {
                "S": S,
                "vanilla_ms": round(vanilla_time, 3) if vanilla_ok else -1,
                "vanilla_gb": round(vanilla_mem, 3) if vanilla_ok else -1,
                "flex_ms": round(flex_time, 3) if flex_ok else -1,
                "flex_gb": round(flex_mem, 3) if flex_ok else -1,
            }
            results[pattern_key].append(row)
            print(f"  S={S}: Vanilla({vanilla_time:.1f}ms, {vanilla_mem:.3f}GB) "
                  f"Flex({flex_time:.1f}ms, {flex_mem:.3f}GB)")

            del q, k, v
            clear_gpu()

    return results


# ============================================================
# Experiment E3: SDPA Baseline for Standard Patterns
# ============================================================
def experiment_e3():
    print("\n" + "=" * 70)
    print("Experiment E3: SDPA (FlashAttention2) Baseline")
    print("=" * 70)

    B, H, D = 1, 8, 64
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    results = []

    for S in seq_lengths:
        clear_gpu()
        try:
            q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
        except Exception:
            break

        # SDPA with is_causal=True (uses FlashAttention2 backend)
        clear_gpu()
        try:
            t0 = time.perf_counter()
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()
            sdpa_time = (time.perf_counter() - t0) * 1000
            sdpa_mem = get_peak_memory()
        except Exception:
            sdpa_time = sdpa_mem = -1

        row = {
            "S": S,
            "sdpa_ms": round(sdpa_time, 3),
            "sdpa_gb": round(sdpa_mem, 3),
        }
        results.append(row)
        print(f"  S={S}: SDPA({sdpa_time:.3f}ms, {sdpa_mem:.3f}GB)")

        del q, k, v
        clear_gpu()

    return results


# ============================================================
# Experiment E4: Head Dimension and Batch Size Sweep
# ============================================================
def experiment_e4():
    print("\n" + "=" * 70)
    print("Experiment E4: Head Dimension and Batch Size Sweep")
    print("=" * 70)

    S = 2048
    results = {}

    # Head dimension sweep
    print("\n--- Head Dimension Sweep (B=1, H=8, S=2048) ---")
    results["head_dim"] = []
    for D in [32, 64, 96, 128]:
        for impl_name, fn in [("vanilla", vanilla_causal), ("flex", flex_causal)]:
            clear_gpu()
            try:
                q = torch.randn(1, 8, S, D, device=DEVICE, dtype=DTYPE)
                k = torch.randn(1, 8, S, D, device=DEVICE, dtype=DTYPE)
                v = torch.randn(1, 8, S, D, device=DEVICE, dtype=DTYPE)
                if impl_name == "flex":
                    _ = fn(q, k, v)
                    torch.cuda.synchronize()
                t = measure(lambda: fn(q, k, v))
                mem = get_peak_memory()
                results["head_dim"].append({
                    "D": D, "impl": impl_name, "time_ms": round(t, 3), "mem_gb": round(mem, 3)
                })
                print(f"  D={D} {impl_name}: {t:.3f}ms, {mem:.3f}GB")
            except Exception:
                print(f"  D={D} {impl_name}: FAILED")
            clear_gpu()

    # Batch size sweep
    print("\n--- Batch Size Sweep (H=8, D=64, S=2048) ---")
    results["batch_size"] = []
    for B in [1, 2, 4, 8]:
        for impl_name, fn in [("vanilla", vanilla_causal), ("flex", flex_causal)]:
            clear_gpu()
            try:
                q = torch.randn(B, 8, S, 64, device=DEVICE, dtype=DTYPE)
                k = torch.randn(B, 8, S, 64, device=DEVICE, dtype=DTYPE)
                v = torch.randn(B, 8, S, 64, device=DEVICE, dtype=DTYPE)
                if impl_name == "flex":
                    _ = fn(q, k, v)
                    torch.cuda.synchronize()
                t = measure(lambda: fn(q, k, v))
                mem = get_peak_memory()
                results["batch_size"].append({
                    "B": B, "impl": impl_name, "time_ms": round(t, 3), "mem_gb": round(mem, 3)
                })
                print(f"  B={B} {impl_name}: {t:.3f}ms, {mem:.3f}GB")
            except Exception:
                print(f"  B={B} {impl_name}: FAILED")
            clear_gpu()

    return results


# ============================================================
# Experiment E5: BlockMask Sparsity Analysis per Pattern
# ============================================================
def experiment_e5():
    print("\n" + "=" * 70)
    print("Experiment E5: BlockMask Sparsity Analysis")
    print("=" * 70)

    S = 2048
    B = 1
    results = {}

    sparsity_configs = {
        "causal": lambda: (lambda b, h, q, kv: q >= kv, {}),
        "sliding_window_64": lambda: (lambda b, h, q, kv: (q >= kv) & ((q - kv) <= 64), {}),
        "sliding_window_256": lambda: (lambda b, h, q, kv: (q >= kv) & ((q - kv) <= 256), {}),
        "prefix_lm_25": lambda: (lambda b, h, q, kv: (q >= kv) | (kv < int(S * 0.25)), {}),
        "prefix_lm_50": lambda: (lambda b, h, q, kv: (q >= kv) | (kv < int(S * 0.50)), {}),
        "doc_2": lambda: (
            lambda b, h, q, kv: (q >= kv) & (
                (torch.arange(S, device=DEVICE) // (S // 2))[q] ==
                (torch.arange(S, device=DEVICE) // (S // 2))[kv]
            ),
            {}
        ),
        "doc_4": lambda: (
            lambda b, h, q, kv: (q >= kv) & (
                (torch.arange(S, device=DEVICE) // (S // 4))[q] ==
                (torch.arange(S, device=DEVICE) // (S // 4))[kv]
            ),
            {}
        ),
        "doc_8": lambda: (
            lambda b, h, q, kv: (q >= kv) & (
                (torch.arange(S, device=DEVICE) // (S // 8))[q] ==
                (torch.arange(S, device=DEVICE) // (S // 8))[kv]
            ),
            {}
        ),
        "dilated_sw_128_2": lambda: (
            lambda b, h, q, kv: (q >= kv) & ((q - kv) <= 128) & (((q - kv) % 2) == 0),
            {}
        ),
        "dilated_sw_256_4": lambda: (
            lambda b, h, q, kv: (q >= kv) & ((q - kv) <= 256) & (((q - kv) % 4) == 0),
            {}
        ),
    }

    for name, config_fn in sparsity_configs.items():
        clear_gpu()
        mask_fn, kwargs = config_fn()
        try:
            # Compute actual sparsity
            pos = torch.arange(S)
            i = pos.unsqueeze(0).expand(S, S)
            j = pos.unsqueeze(1).expand(S, S)
            if "doc_2" in name:
                doc_ids = torch.arange(S) // (S // 2)
                actual_mask = (i >= j) & (doc_ids[i] == doc_ids[j])
            elif "doc_4" in name:
                doc_ids = torch.arange(S) // (S // 4)
                actual_mask = (i >= j) & (doc_ids[i] == doc_ids[j])
            elif "doc_8" in name:
                doc_ids = torch.arange(S) // (S // 8)
                actual_mask = (i >= j) & (doc_ids[i] == doc_ids[j])
            elif "prefix_lm_25" in name:
                actual_mask = (i >= j) | (j < int(S * 0.25))
            elif "prefix_lm_50" in name:
                actual_mask = (i >= j) | (j < int(S * 0.50))
            elif "sliding_window_64" in name:
                actual_mask = (i >= j) & ((i - j) <= 64)
            elif "sliding_window_256" in name:
                actual_mask = (i >= j) & ((i - j) <= 256)
            elif "dilated_sw_128_2" in name:
                actual_mask = (i >= j) & ((i - j) <= 128) & (((i - j) % 2) == 0)
            elif "dilated_sw_256_4" in name:
                actual_mask = (i >= j) & ((i - j) <= 256) & (((i - j) % 4) == 0)
            else:
                actual_mask = i >= j

            total = S * S
            num_true = actual_mask.sum().item()
            sparsity = 1.0 - num_true / total

            # BlockMask info
            block_mask = create_block_mask(mask_fn, B, 1, S, S, device=DEVICE)
            # Count non-empty blocks
            if hasattr(block_mask, 'kv_num_blocks'):
                total_blocks = (S // 128) ** 2
                non_empty = block_mask.kv_num_blocks.sum().item()
                block_sparsity = 1.0 - non_empty / total_blocks if total_blocks > 0 else 0
            else:
                block_sparsity = 0

            results[name] = {
                "pixel_sparsity": round(sparsity * 100, 1),
                "block_sparsity": round(block_sparsity * 100, 1) if block_sparsity else 0,
                "num_true": num_true,
                "total": total,
            }
            print(f"  {name}: pixel={sparsity*100:.1f}% block_skip={block_sparsity*100:.1f}%")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            results[name] = {"error": str(e)}

    return results


# ============================================================
# Experiment E6: Gradient Flow and Backward Pass Analysis
# ============================================================
def experiment_e6():
    print("\n" + "=" * 70)
    print("Experiment E6: Gradient Flow (Forward + Backward Timing)")
    print("=" * 70)

    B, H, D = 1, 8, 64
    S = 2048
    patterns_to_test = ["causal", "sliding_window", "doc_packing", "alibi", "combined"]
    results = {}

    for pattern_key in patterns_to_test:
        pattern_info = PATTERNS[pattern_key]
        print(f"\n--- {pattern_info['name']} ---")
        results[pattern_key] = {}

        for impl_name, fn in [("vanilla", pattern_info["vanilla_fn"]),
                               ("flex", pattern_info["flex_fn"])]:
            clear_gpu()
            try:
                q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
                k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
                v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE, requires_grad=True)

                kwargs = {}
                if pattern_info["needs_doc_ids"]:
                    kwargs["doc_ids"] = torch.arange(S, device=DEVICE) // (S // 4)

                # Warmup
                if impl_name == "flex":
                    _ = fn(q, k, v, **kwargs)
                    torch.cuda.synchronize()

                # Forward
                clear_gpu()
                fwd_time = measure(lambda: fn(q, k, v, **kwargs))
                fwd_mem = get_peak_memory()

                # Backward
                out = fn(q, k, v, **kwargs)
                loss = out.sum()
                bwd_time = measure(lambda: (
                    fn(q, k, v, **kwargs).sum().backward()))
                bwd_mem = get_peak_memory()

                results[pattern_key][impl_name] = {
                    "fwd_ms": round(fwd_time, 3),
                    "bwd_ms": round(bwd_time, 3),
                    "fwd_mem_gb": round(fwd_mem, 3),
                    "bwd_mem_gb": round(bwd_mem, 3),
                    "fwd_bwd_ratio": round(bwd_time / fwd_time, 2) if fwd_time > 0 else 0,
                }
                print(f"  {impl_name}: fwd={fwd_time:.1f}ms bwd={bwd_time:.1f}ms "
                      f"ratio={bwd_time/fwd_time:.2f}x")
            except Exception as e:
                print(f"  {impl_name}: FAILED ({e})")
                results[pattern_key][impl_name] = {"error": str(e)}

            clear_gpu()

    return results


# ============================================================
# Experiment E7: Long Context Stress Test
# ============================================================
def experiment_e7():
    print("\n" + "=" * 70)
    print("Experiment E7: Long Context Stress Test")
    print("=" * 70)

    B, H, D = 1, 8, 64
    seq_lengths = [2048, 4096, 8192, 12288, 16384]
    num_docs = 8
    results = []

    for S in seq_lengths:
        clear_gpu()
        doc_ids = torch.arange(S, device=DEVICE) // (S // num_docs)
        row = {"S": S, "num_docs": num_docs}

        # Vanilla Doc Packing
        try:
            q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            clear_gpu()
            t = measure(lambda: vanilla_doc_packing(q, k, v, doc_ids=doc_ids))
            mem = get_peak_memory()
            row["vanilla_ms"] = round(t, 3)
            row["vanilla_gb"] = round(mem, 3)
            print(f"  S={S} Vanilla: {t:.1f}ms, {mem:.3f}GB")
            del q, k, v
        except Exception as e:
            row["vanilla_ms"] = -1
            row["vanilla_gb"] = -1
            print(f"  S={S} Vanilla: OOM")

        # Flex Doc Packing
        try:
            q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            clear_gpu()
            _ = flex_doc_packing(q, k, v, doc_ids=doc_ids)
            torch.cuda.synchronize()
            t = measure(lambda: flex_doc_packing(q, k, v, doc_ids=doc_ids))
            mem = get_peak_memory()
            row["flex_ms"] = round(t, 3)
            row["flex_gb"] = round(mem, 3)
            print(f"  S={S} Flex: {t:.1f}ms, {mem:.3f}GB")
        except Exception as e:
            row["flex_ms"] = -1
            row["flex_gb"] = -1
            print(f"  S={S} Flex: OOM")

        # SDPA (standard causal - for reference)
        try:
            clear_gpu()
            q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            clear_gpu()
            t = measure(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
            mem = get_peak_memory()
            row["sdpa_ms"] = round(t, 3)
            row["sdpa_gb"] = round(mem, 3)
            print(f"  S={S} SDPA: {t:.1f}ms, {mem:.3f}GB")
        except Exception:
            row["sdpa_ms"] = -1
            row["sdpa_gb"] = -1
            print(f"  S={S} SDPA: OOM")

        results.append(row)
        clear_gpu()

    return results


# ============================================================
# Experiment E8: BlockMask Structure Deep Dive
# ============================================================
def experiment_e8():
    print("\n" + "=" * 70)
    print("Experiment E8: BlockMask Structure Analysis")
    print("=" * 70)

    S = 2048
    B = 1
    results = {}

    def analyze_blockmask(name, mask_fn):
        try:
            bm = create_block_mask(mask_fn, B, 1, S, S, device=DEVICE)
            info = {"exists": True}
            if hasattr(bm, 'kv_num_blocks'):
                info["kv_num_blocks_shape"] = list(bm.kv_num_blocks.shape)
                info["total_non_empty_blocks"] = bm.kv_num_blocks.sum().item()
                info["total_possible_blocks"] = (S // 128) * (S // 128)
                info["block_utilization"] = round(
                    bm.kv_num_blocks.sum().item() / ((S // 128) * (S // 128)) * 100, 1)
            if hasattr(bm, 'kv_indices'):
                info["kv_indices_shape"] = list(bm.kv_indices.shape)
            # Creation time
            t0 = time.perf_counter()
            bm = create_block_mask(mask_fn, B, 1, S, S, device=DEVICE)
            torch.cuda.synchronize()
            info["creation_time_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            # Memory
            info["blockmask_memory_mb"] = round(
                sum(p.nelement() * p.element_size() for p in [
                    bm.kv_num_blocks, bm.kv_indices
                ] if hasattr(bm, attr := 'kv_num_blocks')) / 1e6, 3)
            print(f"  {name}: blocks={info.get('total_non_empty_blocks', '?')}"
                  f"/{info.get('total_possible_blocks', '?')} "
                  f"util={info.get('block_utilization', '?')}%")
            return info
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            return {"error": str(e)}

    # Causal
    results["causal"] = analyze_blockmask("causal",
        lambda b, h, q, kv: q >= kv)

    # Sliding windows
    for w in [64, 128, 256, 512]:
        results[f"sw_{w}"] = analyze_blockmask(f"sw_{w}",
            lambda b, h, q, kv, ww=w: (q >= kv) & ((q - kv) <= ww))

    # Document packing
    for nd in [2, 4, 8, 16]:
        doc_ids = torch.arange(S, device=DEVICE) // (S // nd)
        results[f"doc_{nd}"] = analyze_blockmask(f"doc_{nd}",
            lambda b, h, q, kv, did=doc_ids: (q >= kv) & (did[q] == did[kv]))

    # Prefix LM
    for pr in [0.1, 0.25, 0.5]:
        pl = int(S * pr)
        results[f"prefix_{int(pr*100)}"] = analyze_blockmask(f"prefix_{int(pr*100)}",
            lambda b, h, q, kv, p=pl: (q >= kv) | (kv < p))

    # Dilated SW
    for w, d in [(128, 2), (256, 4), (512, 8)]:
        results[f"dilated_{w}_{d}"] = analyze_blockmask(f"dilated_{w}_{d}",
            lambda b, h, q, kv, ww=w, dd=d: (q >= kv) & ((q - kv) <= ww) & (((q - kv) % dd) == 0))

    return results


# ============================================================
# Chart Generation
# ============================================================
def generate_charts(e1, e2, e3, e4, e5, e6, e7, e8):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'figure.dpi': 150,
    })

    # ---- Chart E1a: Pattern Latency Comparison ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    S_plot = 2048  # Use S=2048 data

    pattern_names = []
    vanilla_fwd = []
    flex_fwd = []
    vanilla_mem = []
    flex_mem = []

    for pk, rows in e1.items():
        for row in rows:
            if row["S"] == S_plot and row["vanilla_fwd_ms"] > 0 and row["flex_fwd_ms"] > 0:
                pattern_names.append(PATTERNS[pk]["name"])
                vanilla_fwd.append(row["vanilla_fwd_ms"])
                flex_fwd.append(row["flex_fwd_ms"])
                vanilla_mem.append(row["vanilla_mem_gb"])
                flex_mem.append(row["flex_mem_gb"])
                break

    x = range(len(pattern_names))
    width = 0.35

    ax = axes[0, 0]
    bars1 = ax.bar([i - width/2 for i in x], vanilla_fwd, width, label='Vanilla', color='#ff6b6b')
    bars2 = ax.bar([i + width/2 for i in x], flex_fwd, width, label='FlexAttention', color='#4ecdc4')
    ax.set_ylabel('Forward Latency (ms)')
    ax.set_title(f'Forward Latency at S={S_plot}')
    ax.set_xticks(x)
    ax.set_xticklabels(pattern_names, rotation=25, ha='right', fontsize=7)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 1]
    bars1 = ax.bar([i - width/2 for i in x], vanilla_mem, width, label='Vanilla', color='#ff6b6b')
    bars2 = ax.bar([i + width/2 for i in x], flex_mem, width, label='FlexAttention', color='#4ecdc4')
    ax.set_ylabel('Peak Memory (GB)')
    ax.set_title(f'Peak Memory at S={S_plot}')
    ax.set_xticks(x)
    ax.set_xticklabels(pattern_names, rotation=25, ha='right', fontsize=7)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Numerical accuracy
    ax = axes[1, 0]
    max_diffs = []
    for pk, rows in e1.items():
        for row in rows:
            if row["S"] == S_plot and row["max_diff"] >= 0:
                max_diffs.append((PATTERNS[pk]["name"], row["max_diff"]))
                break
    names, diffs = zip(*max_diffs) if max_diffs else ([], [])
    colors = ['#2ecc71' if d < 0.01 else '#e74c3c' for d in diffs]
    ax.barh(range(len(names)), diffs, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Max Absolute Difference')
    ax.set_title('Numerical Accuracy (Vanilla vs Flex)')
    ax.set_xscale('log')
    ax.axvline(x=0.01, color='orange', linestyle='--', alpha=0.5, label='0.01 threshold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # Speed ratio
    ax = axes[1, 1]
    speed_ratios = []
    ratio_names = []
    for pk, rows in e1.items():
        for row in rows:
            if row["S"] == S_plot and row["vanilla_fwd_ms"] > 0 and row["flex_fwd_ms"] > 0:
                ratio_names.append(PATTERNS[pk]["name"])
                speed_ratios.append(row["flex_fwd_ms"] / row["vanilla_fwd_ms"])
                break
    colors = ['#e74c3c' if r > 3 else '#f39c12' if r > 1.5 else '#2ecc71' for r in speed_ratios]
    ax.barh(range(len(ratio_names)), speed_ratios, color=colors)
    ax.set_yticks(range(len(ratio_names)))
    ax.set_yticklabels(ratio_names, fontsize=8)
    ax.set_xlabel('Flex / Vanilla Speed Ratio')
    ax.set_title('Speed Ratio (higher = Flex slower)')
    ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/E1_pattern_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/E1_pattern_comparison.png")

    # ---- Chart E2: Scalability ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for pk in e2:
        if not e2[pk]:
            continue
        name = PATTERNS[pk]["name"]
        s_vals = [r["S"] for r in e2[pk] if r["vanilla_ms"] > 0 or r["flex_ms"] > 0]
        v_times = [r["vanilla_ms"] if r["vanilla_ms"] > 0 else None for r in e2[pk]]
        f_times = [r["flex_ms"] if r["flex_ms"] > 0 else None for r in e2[pk]]

        v_plot = [(s, t) for s, t in zip(s_vals, v_times) if t is not None]
        f_plot = [(s, t) for s, t in zip(s_vals, f_times) if t is not None]

        if v_plot:
            axes[0].plot([p[0] for p in v_plot], [p[1] for p in v_plot],
                        marker='o', linestyle='-', alpha=0.7, label=f'{name} (Vanilla)')
        if f_plot:
            axes[0].plot([p[0] for p in f_plot], [p[1] for p in f_plot],
                        marker='s', linestyle='--', alpha=0.7, label=f'{name} (Flex)')

    axes[0].set_xlabel('Sequence Length (S)')
    axes[0].set_ylabel('Forward Latency (ms)')
    axes[0].set_title('Latency Scalability')
    axes[0].legend(fontsize=7, loc='upper left')
    axes[0].grid(alpha=0.3)
    axes[0].set_xscale('log', base=2)
    axes[0].set_yscale('log')

    for pk in e2:
        if not e2[pk]:
            continue
        name = PATTERNS[pk]["name"]
        s_vals = [r["S"] for r in e2[pk] if r["vanilla_gb"] > 0 or r["flex_gb"] > 0]
        v_mems = [r["vanilla_gb"] if r["vanilla_gb"] > 0 else None for r in e2[pk]]
        f_mems = [r["flex_gb"] if r["flex_gb"] > 0 else None for r in e2[pk]]

        v_plot = [(s, m) for s, m in zip(s_vals, v_mems) if m is not None]
        f_plot = [(s, m) for s, m in zip(s_vals, f_mems) if m is not None]

        if v_plot:
            axes[1].plot([p[0] for p in v_plot], [p[1] for p in v_plot],
                        marker='o', linestyle='-', alpha=0.7, label=f'{name} (Vanilla)')
        if f_plot:
            axes[1].plot([p[0] for p in f_plot], [p[1] for p in f_plot],
                        marker='s', linestyle='--', alpha=0.7, label=f'{name} (Flex)')

    axes[1].set_xlabel('Sequence Length (S)')
    axes[1].set_ylabel('Peak Memory (GB)')
    axes[1].set_title('Memory Scalability')
    axes[1].legend(fontsize=7, loc='upper left')
    axes[1].grid(alpha=0.3)
    axes[1].set_xscale('log', base=2)
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/E2_scalability.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/E2_scalability.png")

    # ---- Chart E3: SDPA Baseline ----
    fig, ax = plt.subplots(figsize=(10, 5))
    s_vals = [r["S"] for r in e3]
    times = [r["sdpa_ms"] for r in e3]
    mems = [r["sdpa_gb"] for r in e3]

    ax2 = ax.twinx()
    l1 = ax.plot(s_vals, times, 'b-o', label='Latency (ms)')
    l2 = ax2.plot(s_vals, mems, 'r-s', label='Memory (GB)')

    ax.set_xlabel('Sequence Length (S)')
    ax.set_ylabel('Latency (ms)', color='b')
    ax2.set_ylabel('Peak Memory (GB)', color='r')
    ax.set_title('SDPA (FlashAttention2) Baseline Performance')
    ax.set_xscale('log', base=2)
    ax.grid(alpha=0.3)
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/E3_sdpa_baseline.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/E3_sdpa_baseline.png")

    # ---- Chart E4a: Head Dimension ----
    if e4.get("head_dim"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        d_vals = sorted(set(r["D"] for r in e4["head_dim"]))
        for impl in ["vanilla", "flex"]:
            sub = [r for r in e4["head_dim"] if r["impl"] == impl]
            d_sub = [r["D"] for r in sub]
            t_sub = [r["time_ms"] for r in sub]
            m_sub = [r["mem_gb"] for r in sub]
            axes[0].plot(d_sub, t_sub, marker='o', label=impl)
            axes[1].plot(d_sub, m_sub, marker='o', label=impl)
        axes[0].set_xlabel('Head Dimension (D)')
        axes[0].set_ylabel('Latency (ms)')
        axes[0].set_title('Latency vs Head Dimension (S=2048)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].set_xlabel('Head Dimension (D)')
        axes[1].set_ylabel('Peak Memory (GB)')
        axes[1].set_title('Memory vs Head Dimension (S=2048)')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/E4_head_dim.png', bbox_inches='tight')
        plt.close()
        print(f"Saved {FIGURES_DIR}/E4_head_dim.png")

    # ---- Chart E4b: Batch Size ----
    if e4.get("batch_size"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for impl in ["vanilla", "flex"]:
            sub = [r for r in e4["batch_size"] if r["impl"] == impl]
            b_sub = [r["B"] for r in sub]
            t_sub = [r["time_ms"] for r in sub]
            m_sub = [r["mem_gb"] for r in sub]
            axes[0].plot(b_sub, t_sub, marker='o', label=impl)
            axes[1].plot(b_sub, m_sub, marker='o', label=impl)
        axes[0].set_xlabel('Batch Size (B)')
        axes[0].set_ylabel('Latency (ms)')
        axes[0].set_title('Latency vs Batch Size (S=2048, D=64)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].set_xlabel('Batch Size (B)')
        axes[1].set_ylabel('Peak Memory (GB)')
        axes[1].set_title('Memory vs Batch Size (S=2048, D=64)')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/E4_batch_size.png', bbox_inches='tight')
        plt.close()
        print(f"Saved {FIGURES_DIR}/E4_batch_size.png")

    # ---- Chart E5: Sparsity ----
    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(e5.keys())
    pixel_sp = [e5[n].get("pixel_sparsity", 0) for n in names]
    block_sp = [e5[n].get("block_sparsity", 0) for n in names]

    x = range(len(names))
    width = 0.35
    ax.bar([i - width/2 for i in x], pixel_sp, width, label='Pixel Sparsity', color='#e74c3c')
    ax.bar([i + width/2 for i in x], block_sp, width, label='Block Skip Ratio', color='#3498db')
    ax.set_ylabel('Sparsity (%)')
    ax.set_title('Sparsity Analysis Across Attention Patterns (S=2048)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/E5_sparsity.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/E5_sparsity.png")

    # ---- Chart E6: Gradient Flow ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for pk in e6:
        name = PATTERNS[pk]["name"]
        for impl in ["vanilla", "flex"]:
            if impl in e6[pk] and "error" not in e6[pk][impl]:
                fwd = e6[pk][impl]["fwd_ms"]
                bwd = e6[pk][impl]["bwd_ms"]
                axes[0].bar(f"{name}\n({impl})", fwd, color='#ff6b6b' if impl == 'vanilla' else '#4ecdc4', alpha=0.8)
                axes[0].bar(f"{name}\n({impl})", bwd, bottom=fwd, color='#c0392b' if impl == 'vanilla' else '#16a085', alpha=0.6)

    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Forward + Backward Time')
    axes[0].tick_params(axis='x', rotation=45, labelsize=7)
    axes[0].grid(axis='y', alpha=0.3)

    # FWD/BWD ratio
    for pk in e6:
        name = PATTERNS[pk]["name"]
        for impl in ["vanilla", "flex"]:
            if impl in e6[pk] and "error" not in e6[pk][impl]:
                ratio = e6[pk][impl]["fwd_bwd_ratio"]
                axes[1].barh(f"{name} ({impl})", ratio,
                            color='#e74c3c' if impl == 'vanilla' else '#4ecdc4')

    axes[1].set_xlabel('BWD / FWD Ratio')
    axes[1].set_title('Backward-to-Forward Ratio')
    axes[1].axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/E6_gradient_flow.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/E6_gradient_flow.png")

    # ---- Chart E7: Long Context Stress ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    s_vals = [r["S"] for r in e7]

    for method, color, marker, ls in [
        ("vanilla", "#ff6b6b", "o", "-"),
        ("flex", "#4ecdc4", "s", "--"),
        ("sdpa", "#3498db", "^", "-."),
    ]:
        times = [r.get(f"{method}_ms", -1) for r in e7]
        mems = [r.get(f"{method}_gb", -1) for r in e7]
        t_plot = [(s, t) for s, t in zip(s_vals, times) if t > 0]
        m_plot = [(s, m) for s, m in zip(s_vals, mems) if m > 0]
        label = method.upper() if method != "sdpa" else "SDPA (Flash2)"
        if t_plot:
            axes[0].plot([p[0] for p in t_plot], [p[1] for p in t_plot],
                        marker=marker, linestyle=ls, color=color, label=label)
        if m_plot:
            axes[1].plot([p[0] for p in m_plot], [p[1] for p in m_plot],
                        marker=marker, linestyle=ls, color=color, label=label)

    axes[0].set_xlabel('Sequence Length (S)')
    axes[0].set_ylabel('Latency (ms)')
    axes[0].set_title('Long Context Latency (Doc Packing, 8 docs)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xscale('log', base=2)
    axes[0].set_yscale('log')

    axes[1].set_xlabel('Sequence Length (S)')
    axes[1].set_ylabel('Peak Memory (GB)')
    axes[1].set_title('Long Context Memory')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xscale('log', base=2)
    axes[1].axhline(y=24, color='red', linestyle=':', alpha=0.5, label='L4 VRAM Limit')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/E7_long_context.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/E7_long_context.png")

    # ---- Chart E8: BlockMask Analysis ----
    fig, ax = plt.subplots(figsize=(14, 6))
    names = list(e8.keys())
    utils = [e8[n].get("block_utilization", 0) for n in names]

    colors = plt.cm.RdYlGn_r([u / 100 for u in utils])
    bars = ax.bar(range(len(names)), utils, color=colors)
    ax.set_ylabel('Block Utilization (%)')
    ax.set_title('BlockMask Utilization Across Patterns (S=2048)')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    for bar, util in zip(bars, utils):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{util:.1f}%', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/E8_blockmask_analysis.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/E8_blockmask_analysis.png")

    print("\nAll charts saved!")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("FlexAttention Comprehensive Pattern Analysis")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    all_results = {}

    # Run all experiments
    all_results["E1"] = experiment_e1()
    all_results["E2"] = experiment_e2()
    all_results["E3"] = experiment_e3()
    all_results["E4"] = experiment_e4()
    all_results["E5"] = experiment_e5()
    all_results["E6"] = experiment_e6()
    all_results["E7"] = experiment_e7()
    all_results["E8"] = experiment_e8()

    # Generate charts
    generate_charts(
        all_results["E1"], all_results["E2"], all_results["E3"],
        all_results["E4"], all_results["E5"], all_results["E6"],
        all_results["E7"], all_results["E8"]
    )

    # Save results
    with open("pattern_analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to pattern_analysis_results.json")
    print("DONE!")
