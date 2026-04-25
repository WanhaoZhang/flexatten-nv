#!/usr/bin/env python3
"""FlexAttention Comprehensive Experiments on NVIDIA L4
Based on: FlexAttention 基础知识与机制深度解析
Reference: https://github.com/meta-pytorch/attention-gym
"""

import torch
import torch.nn.functional as F
import json
import time
import sys
import gc
import traceback
from pathlib import Path

device = "cuda"
dtype = torch.float16

results = {}

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def benchmark(fn, warmup=3, runs=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return {"mean_ms": round(sum(times)/len(times), 3), "min_ms": round(min(times), 3), "max_ms": round(max(times), 3)}

def peak_mem(fn):
    clear_cache()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    mem_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    del out
    return mem_gb

# ================================================================
# Exp1: Attention Evolution (Standard vs SDPA vs FlexAttention)
# ================================================================
def exp1_attention_evolution():
    print("\n" + "="*70)
    print("  Experiment 1: Attention Evolution Comparison")
    print("="*70)
    H, D, B = 8, 64, 1
    seq_lengths = [512, 1024, 2048, 4096, 8192]
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    rows = []
    for S in seq_lengths:
        print(f"\n  [S={S}] B={B} H={H} D={D}")
        clear_cache()
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)

        # Standard Attention
        std_ok = True
        try:
            def std_fn():
                sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
                mask = torch.ones(S, S, device=device, dtype=torch.bool).tril_()
                sc = sc.masked_fill(~mask, float('-inf'))
                return torch.matmul(F.softmax(sc.float(), dim=-1).to(dtype), v)
            mem_std = peak_mem(std_fn)
            t_std = benchmark(std_fn, warmup=2, runs=5)
            print(f"    Standard: {t_std['mean_ms']:>8.3f} ms | {mem_std:>6.3f} GB")
        except RuntimeError as e:
            print(f"    Standard: OOM")
            std_ok = False; mem_std = None; t_std = None
            clear_cache()

        # SDPA
        sdpa_ok = True
        try:
            def sdpa_fn():
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
            mem_sdpa = peak_mem(sdpa_fn)
            t_sdpa = benchmark(sdpa_fn)
            print(f"    SDPA:     {t_sdpa['mean_ms']:>8.3f} ms | {mem_sdpa:>6.3f} GB")
        except RuntimeError:
            print(f"    SDPA:     OOM")
            sdpa_ok = False; mem_sdpa = None; t_sdpa = None
            clear_cache()

        # FlexAttention
        flex_ok = True
        try:
            def causal_mod(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx
            bm = create_block_mask(causal_mod, B, 1, S, S, device=device)
            # compile warmup
            _ = flex_attention(q, k, v, block_mask=bm)
            torch.cuda.synchronize()
            def flex_fn():
                return flex_attention(q, k, v, block_mask=bm)
            mem_flex = peak_mem(flex_fn)
            t_flex = benchmark(flex_fn)
            print(f"    Flex:     {t_flex['mean_ms']:>8.3f} ms | {mem_flex:>6.3f} GB")
        except RuntimeError:
            print(f"    Flex:     OOM")
            flex_ok = False; mem_flex = None; t_flex = None
            clear_cache()

        # Numerical diff
        diff = None
        if sdpa_ok and flex_ok:
            o1 = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            o2 = flex_attention(q, k, v, block_mask=bm)
            diff = round((o1.float() - o2.float()).abs().max().item(), 6)
            print(f"    Max diff (SDPA vs Flex): {diff}")

        rows.append({"S": S, "standard": {"time": t_std, "mem": mem_std},
                      "sdpa": {"time": t_sdpa, "mem": mem_sdpa},
                      "flex": {"time": t_flex, "mem": mem_flex}, "diff_sdpa_flex": diff})
        del q, k, v; clear_cache()
    return rows

# ================================================================
# Exp2: Document Packing + Causal (Dense vs Flex)
# ================================================================
def exp2_document_packing():
    print("\n" + "="*70)
    print("  Experiment 2: Document Packing + Causal")
    print("="*70)
    H, D, B = 8, 64, 1
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    rows = []
    for S in [1024, 2048, 4096, 8192]:
        for num_docs in [2, 4, 8, 16]:
            if S // num_docs < 128:
                continue
            print(f"\n  [S={S}, docs={num_docs}]")
            clear_cache()
            doc_ids = (torch.arange(S, device=device) // (S // num_docs)).clamp(max=num_docs-1)
            q = torch.randn(B, H, S, D, device=device, dtype=dtype)
            k = torch.randn(B, H, S, D, device=device, dtype=dtype)
            v = torch.randn(B, H, S, D, device=device, dtype=dtype)

            # Dense
            dense_ok = True
            try:
                def dense_fn():
                    sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
                    cm = torch.ones(S, S, device=device, dtype=torch.bool).tril_()
                    dm = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
                    sc = sc.masked_fill(~(cm & dm), float('-inf'))
                    return torch.matmul(F.softmax(sc.float(), dim=-1).to(dtype), v)
                mem_d = peak_mem(dense_fn)
                t_d = benchmark(dense_fn, warmup=1, runs=3)
                print(f"    Dense: {t_d['mean_ms']:>8.3f} ms | {mem_d:>6.3f} GB")
            except RuntimeError:
                print(f"    Dense: OOM"); dense_ok = False; mem_d = None; t_d = None
                clear_cache()

            # Flex
            flex_ok = True
            sparsity = None
            try:
                def packed_mod(score, b, h, q_idx, kv_idx):
                    return torch.where((q_idx >= kv_idx) & (doc_ids[q_idx] == doc_ids[kv_idx]),
                                       score, float('-inf'))
                bm = create_block_mask(packed_mod, B, 1, S, S, device=device)
                total_blks = bm.kv_num_blocks.sum().item()
                max_blks = B * (S // 128) * (S // 128)
                sparsity = round(1.0 - total_blks / max_blks, 4) if max_blks > 0 else 0
                _ = flex_attention(q, k, v, block_mask=bm)
                torch.cuda.synchronize()
                def flex_fn():
                    return flex_attention(q, k, v, block_mask=bm)
                mem_f = peak_mem(flex_fn)
                t_f = benchmark(flex_fn, warmup=2, runs=5)
                print(f"    Flex:  {t_f['mean_ms']:>8.3f} ms | {mem_f:>6.3f} GB | sparse={sparsity:.1%}")
            except RuntimeError:
                print(f"    Flex:  OOM"); flex_ok = False; mem_f = None; t_f = None
                clear_cache()

            diff = None
            if dense_ok and flex_ok:
                o1 = dense_fn()
                o2 = flex_attention(q, k, v, block_mask=bm)
                diff = round((o1.float() - o2.float()).abs().max().item(), 6)
                print(f"    Max diff: {diff}")

            rows.append({"S": S, "num_docs": num_docs,
                          "dense": {"time": t_d, "mem": mem_d},
                          "flex": {"time": t_f, "mem": mem_f},
                          "sparsity": sparsity, "max_diff": diff})
            del q, k, v; clear_cache()
    return rows

# ================================================================
# Exp3: score_mod Mechanisms
# ================================================================
def exp3_score_mods():
    print("\n" + "="*70)
    print("  Experiment 3: score_mod Mechanisms")
    print("="*70)
    H, D, B, S = 8, 64, 1, 2048
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    rows = []
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    # --- ALiBi ---
    print("\n  [ALiBi Score Mod]")
    slopes = 1.0 / (2.0 ** (8 + torch.arange(H, device=device, dtype=dtype)))
    def alibi_ref():
        sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
        pos = torch.arange(S, device=device, dtype=dtype)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        for h in range(H):
            sc[:, h] -= slopes[h] * dist
        return torch.matmul(F.softmax(sc.float(), dim=-1).to(dtype), v)

    def alibi_mod(score, b, h, q_idx, kv_idx):
        return score - slopes[h] * (q_idx - kv_idx).abs().to(dtype)

    clear_cache()
    mem_r = peak_mem(alibi_ref)
    t_r = benchmark(alibi_ref, warmup=1, runs=5)
    print(f"    Reference: {t_r['mean_ms']:>8.3f} ms | {mem_r:>6.3f} GB")

    _ = flex_attention(q, k, v, score_mod=alibi_mod); torch.cuda.synchronize()
    mem_f = peak_mem(lambda: flex_attention(q, k, v, score_mod=alibi_mod))
    t_f = benchmark(lambda: flex_attention(q, k, v, score_mod=alibi_mod), warmup=2, runs=5)
    print(f"    Flex:      {t_f['mean_ms']:>8.3f} ms | {mem_f:>6.3f} GB")
    o1 = alibi_ref(); o2 = flex_attention(q, k, v, score_mod=alibi_mod)
    diff = round((o1.float() - o2.float()).abs().max().item(), 6)
    print(f"    Max diff: {diff}")
    rows.append({"name": "ALiBi", "ref": {"time": t_r, "mem": mem_r},
                  "flex": {"time": t_f, "mem": mem_f}, "diff": diff})

    # --- Softcapping ---
    print("\n  [Softcapping Score Mod]")
    CAP = 50.0
    def softcap_ref():
        sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
        sc = CAP * torch.tanh(sc / CAP)
        return torch.matmul(F.softmax(sc.float(), dim=-1).to(dtype), v)
    def softcap_mod(score, b, h, q_idx, kv_idx):
        return CAP * torch.tanh(score / CAP)

    clear_cache()
    mem_r = peak_mem(softcap_ref)
    t_r = benchmark(softcap_ref, warmup=1, runs=5)
    print(f"    Reference: {t_r['mean_ms']:>8.3f} ms | {mem_r:>6.3f} GB")

    _ = flex_attention(q, k, v, score_mod=softcap_mod); torch.cuda.synchronize()
    mem_f = peak_mem(lambda: flex_attention(q, k, v, score_mod=softcap_mod))
    t_f = benchmark(lambda: flex_attention(q, k, v, score_mod=softcap_mod), warmup=2, runs=5)
    print(f"    Flex:      {t_f['mean_ms']:>8.3f} ms | {mem_f:>6.3f} GB")
    o1 = softcap_ref(); o2 = flex_attention(q, k, v, score_mod=softcap_mod)
    diff = round((o1.float() - o2.float()).abs().max().item(), 6)
    print(f"    Max diff: {diff}")
    rows.append({"name": "Softcapping", "ref": {"time": t_r, "mem": mem_r},
                  "flex": {"time": t_f, "mem": mem_f}, "diff": diff})

    # --- Relative Position Bias ---
    print("\n  [Relative Position Bias]")
    MAX_DIST = 128
    def relpos_ref():
        sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
        pos = torch.arange(S, device=device, dtype=dtype)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp(-MAX_DIST, MAX_DIST)
        sc = sc + dist * 0.01
        cm = torch.ones(S, S, device=device, dtype=torch.bool).tril_()
        sc = sc.masked_fill(~cm, float('-inf'))
        return torch.matmul(F.softmax(sc.float(), dim=-1).to(dtype), v)
    def relpos_mod(score, b, h, q_idx, kv_idx):
        dist = (q_idx - kv_idx).clamp(-MAX_DIST, MAX_DIST).to(dtype)
        return score + dist * 0.01
    def causal_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    bm = create_block_mask(causal_mod, B, 1, S, S, device=device)

    clear_cache()
    mem_r = peak_mem(relpos_ref)
    t_r = benchmark(relpos_ref, warmup=1, runs=5)
    print(f"    Reference: {t_r['mean_ms']:>8.3f} ms | {mem_r:>6.3f} GB")

    _ = flex_attention(q, k, v, score_mod=relpos_mod, block_mask=bm); torch.cuda.synchronize()
    mem_f = peak_mem(lambda: flex_attention(q, k, v, score_mod=relpos_mod, block_mask=bm))
    t_f = benchmark(lambda: flex_attention(q, k, v, score_mod=relpos_mod, block_mask=bm), warmup=2, runs=5)
    print(f"    Flex:      {t_f['mean_ms']:>8.3f} ms | {mem_f:>6.3f} GB")
    o1 = relpos_ref(); o2 = flex_attention(q, k, v, score_mod=relpos_mod, block_mask=bm)
    diff = round((o1.float() - o2.float()).abs().max().item(), 6)
    print(f"    Max diff: {diff}")
    rows.append({"name": "RelativePositionBias", "ref": {"time": t_r, "mem": mem_r},
                  "flex": {"time": t_f, "mem": mem_f}, "diff": diff})

    del q, k, v; clear_cache()
    return rows

# ================================================================
# Exp4: BlockMask Sparsity Analysis
# ================================================================
def exp4_sparsity():
    print("\n" + "="*70)
    print("  Experiment 4: BlockMask Sparsity vs Speed")
    print("="*70)
    H, D, B = 8, 64, 1
    S = 2048
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    max_blks = B * (S // 128) ** 2

    masks = {}

    # Causal
    masks["Causal"] = lambda: create_block_mask(
        lambda b, h, q_idx, kv_idx: q_idx >= kv_idx, B, 1, S, S, device=device)

    # Sliding Window 128
    masks["SlidingWindow(128)"] = lambda: create_block_mask(
        lambda b, h, q_idx, kv_idx: (q_idx >= kv_idx) & ((q_idx - kv_idx) <= 128),
        B, 1, S, S, device=device)

    # Sliding Window 256
    masks["SlidingWindow(256)"] = lambda: create_block_mask(
        lambda b, h, q_idx, kv_idx: (q_idx >= kv_idx) & ((q_idx - kv_idx) <= 256),
        B, 1, S, S, device=device)

    # Sliding Window 512
    masks["SlidingWindow(512)"] = lambda: create_block_mask(
        lambda b, h, q_idx, kv_idx: (q_idx >= kv_idx) & ((q_idx - kv_idx) <= 512),
        B, 1, S, S, device=device)

    # Document 4
    doc4 = (torch.arange(S, device=device) // (S//4)).clamp(max=3)
    masks["Document(4)"] = lambda: create_block_mask(
        lambda score, b, h, q_idx, kv_idx: torch.where(
            (q_idx >= kv_idx) & (doc4[q_idx] == doc4[kv_idx]), score, float('-inf')),
        B, 1, S, S, device=device)

    # Document 8
    doc8 = (torch.arange(S, device=device) // (S//8)).clamp(max=7)
    masks["Document(8)"] = lambda: create_block_mask(
        lambda score, b, h, q_idx, kv_idx: torch.where(
            (q_idx >= kv_idx) & (doc8[q_idx] == doc8[kv_idx]), score, float('-inf')),
        B, 1, S, S, device=device)

    # Document 16
    doc16 = (torch.arange(S, device=device) // (S//16)).clamp(max=15)
    masks["Document(16)"] = lambda: create_block_mask(
        lambda score, b, h, q_idx, kv_idx: torch.where(
            (q_idx >= kv_idx) & (doc16[q_idx] == doc16[kv_idx]), score, float('-inf')),
        B, 1, S, S, device=device)

    # Prefix LM 25%
    masks["PrefixLM(25%)"] = lambda: create_block_mask(
        lambda b, h, q_idx, kv_idx: (kv_idx < S//4) | (q_idx >= kv_idx),
        B, 1, S, S, device=device)

    # Prefix LM 50%
    masks["PrefixLM(50%)"] = lambda: create_block_mask(
        lambda b, h, q_idx, kv_idx: (kv_idx < S//2) | (q_idx >= kv_idx),
        B, 1, S, S, device=device)

    rows = []
    for name, mk_mask in masks.items():
        print(f"\n  [{name}]")
        bm = mk_mask()
        sparsity = round(1.0 - bm.kv_num_blocks.sum().item() / max_blks, 4)
        _ = flex_attention(q, k, v, block_mask=bm); torch.cuda.synchronize()
        t = benchmark(lambda: flex_attention(q, k, v, block_mask=bm), warmup=3, runs=10)
        print(f"    Sparsity: {sparsity:.1%} | Time: {t['mean_ms']:.3f} ms")
        rows.append({"mask": name, "sparsity": sparsity, "time_ms": t['mean_ms']})

    # SDPA baseline
    t_sdpa = benchmark(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    rows.append({"mask": "SDPA_Causal_Baseline", "sparsity": 0.5, "time_ms": t_sdpa['mean_ms']})
    print(f"\n  [SDPA Causal Baseline] {t_sdpa['mean_ms']:.3f} ms")

    del q, k, v; clear_cache()
    return rows

# ================================================================
# Exp5: Stress Test
# ================================================================
def exp5_stress():
    print("\n" + "="*70)
    print("  Experiment 5: Stress Test - Max Sequence Length")
    print("="*70)
    H, D, B = 8, 64, 1
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    rows = []
    for S in [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768]:
        print(f"\n  [S={S}]")
        # SDPA
        sdpa_ok = True
        try:
            clear_cache()
            q = torch.randn(B, H, S, D, device=device, dtype=dtype)
            k = torch.randn(B, H, S, D, device=device, dtype=dtype)
            v = torch.randn(B, H, S, D, device=device, dtype=dtype)
            torch.cuda.reset_peak_memory_stats()
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()
            mem = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            t = benchmark(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), warmup=1, runs=3)
            print(f"    SDPA: {t['mean_ms']:>8.3f} ms | {mem:>6.3f} GB")
        except RuntimeError:
            print(f"    SDPA: OOM"); sdpa_ok = False; mem = None; t = None
            clear_cache()

        # Flex
        flex_ok = True
        try:
            clear_cache()
            q = torch.randn(B, H, S, D, device=device, dtype=dtype)
            k = torch.randn(B, H, S, D, device=device, dtype=dtype)
            v = torch.randn(B, H, S, D, device=device, dtype=dtype)
            bm = create_block_mask(lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
                                   B, 1, S, S, device=device)
            _ = flex_attention(q, k, v, block_mask=bm); torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            _ = flex_attention(q, k, v, block_mask=bm)
            torch.cuda.synchronize()
            mem_f = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            t_f = benchmark(lambda: flex_attention(q, k, v, block_mask=bm), warmup=1, runs=3)
            print(f"    Flex: {t_f['mean_ms']:>8.3f} ms | {mem_f:>6.3f} GB")
        except RuntimeError:
            print(f"    Flex: OOM"); flex_ok = False; mem_f = None; t_f = None
            clear_cache()

        if not sdpa_ok and not flex_ok:
            rows.append({"S": S, "note": "Both OOM"})
            break
        rows.append({"S": S,
                      "sdpa": {"time": t, "mem": mem} if sdpa_ok else None,
                      "flex": {"time": t_f, "mem": mem_f} if flex_ok else None})
        del q, k, v; clear_cache()
    return rows

# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  FlexAttention Comprehensive Experiments on NVIDIA L4")
    print("=" * 70)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  dtype: {dtype}")

    all_results = {}
    experiments = [
        ("exp1_attention_evolution", exp1_attention_evolution),
        ("exp2_document_packing", exp2_document_packing),
        ("exp3_score_mods", exp3_score_mods),
        ("exp4_sparsity_analysis", exp4_sparsity),
        ("exp5_stress_test", exp5_stress),
    ]

    for name, fn in experiments:
        try:
            all_results[name] = fn()
        except Exception as e:
            print(f"\n  {name} FAILED: {e}")
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

    out = Path(__file__).parent / "experiment_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out}")
    print("  DONE!")
