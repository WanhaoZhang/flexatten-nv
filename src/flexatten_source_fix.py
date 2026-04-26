#!/usr/bin/env python3
"""
Fix + Rerun: F6-F8 Source Analysis + Generate All Charts
=========================================================
"""

import torch
import torch.nn.functional as F
import time
import json
import gc
import os

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
    return (sum(times) / len(times)) * 1000

def get_peak_memory():
    return torch.cuda.max_memory_allocated() / 1e9


# ============================================================
# Experiment F6: Comprehensive Three-Way Benchmark (FIXED)
# ============================================================
def experiment_f6():
    print("=" * 70)
    print("Experiment F6: Comprehensive Three-Way Benchmark (FIXED)")
    print("=" * 70)

    B, H, D = 1, 8, 64
    seq_lengths = [512, 1024, 2048, 4096]
    results = []

    for S in seq_lengths:
        clear_gpu()
        q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
        v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
        scale = D ** 0.5

        # --- Causal ---
        row = {"S": S, "pattern": "causal"}

        # Vanilla causal
        clear_gpu()
        def vanilla_causal():
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            causal = torch.ones(S, S, device=DEVICE, dtype=torch.bool).tril_()
            scores = scores.masked_fill(~causal, float('-inf'))
            return torch.matmul(F.softmax(scores.float(), dim=-1).to(DTYPE), v)
        row["vanilla_ms"] = round(measure(vanilla_causal), 3)
        row["vanilla_gb"] = round(get_peak_memory(), 3)

        # SDPA causal
        clear_gpu()
        row["sdpa_ms"] = round(measure(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)), 3)
        row["sdpa_gb"] = round(get_peak_memory(), 3)

        # Flex causal
        clear_gpu()
        bm = create_block_mask(lambda b, h, qi, kvi: qi >= kvi, B, 1, S, S, device=DEVICE)
        _ = flex_attention(q, k, v, block_mask=bm)
        torch.cuda.synchronize()
        row["flex_ms"] = round(measure(lambda: flex_attention(q, k, v, block_mask=bm)), 3)
        row["flex_gb"] = round(get_peak_memory(), 3)

        results.append(row)
        print(f"  S={S} causal: V={row['vanilla_ms']}ms SDPA={row['sdpa_ms']}ms F={row['flex_ms']}ms")

        # --- Doc Packing (4 docs) ---
        clear_gpu()
        doc_ids = torch.arange(S, device=DEVICE) // (S // 4)
        row2 = {"S": S, "pattern": "doc_4"}

        # Vanilla doc
        clear_gpu()
        def vanilla_doc():
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            causal = torch.ones(S, S, device=DEVICE, dtype=torch.bool).tril_()
            doc = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
            scores = scores.masked_fill(~(causal & doc), float('-inf'))
            return torch.matmul(F.softmax(scores.float(), dim=-1).to(DTYPE), v)
        row2["vanilla_ms"] = round(measure(vanilla_doc), 3)
        row2["vanilla_gb"] = round(get_peak_memory(), 3)

        row2["sdpa_ms"] = -2  # Not supported
        row2["sdpa_gb"] = -2

        # Flex doc
        clear_gpu()
        def doc_mask(b, h, qi, kvi):
            return (qi >= kvi) & (doc_ids[qi] == doc_ids[kvi])
        bm2 = create_block_mask(doc_mask, B, 1, S, S, device=DEVICE)
        _ = flex_attention(q, k, v, block_mask=bm2)
        torch.cuda.synchronize()
        row2["flex_ms"] = round(measure(lambda: flex_attention(q, k, v, block_mask=bm2)), 3)
        row2["flex_gb"] = round(get_peak_memory(), 3)

        results.append(row2)
        print(f"  S={S} doc_4: V={row2['vanilla_ms']}ms SDPA=N/A F={row2['flex_ms']}ms")

        # --- Doc Packing (8 docs) ---
        clear_gpu()
        doc_ids8 = torch.arange(S, device=DEVICE) // (S // 8)
        row3 = {"S": S, "pattern": "doc_8"}

        clear_gpu()
        def vanilla_doc8():
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            causal = torch.ones(S, S, device=DEVICE, dtype=torch.bool).tril_()
            doc = doc_ids8.unsqueeze(0) == doc_ids8.unsqueeze(1)
            scores = scores.masked_fill(~(causal & doc), float('-inf'))
            return torch.matmul(F.softmax(scores.float(), dim=-1).to(DTYPE), v)
        row3["vanilla_ms"] = round(measure(vanilla_doc8), 3)
        row3["vanilla_gb"] = round(get_peak_memory(), 3)
        row3["sdpa_ms"] = -2
        row3["sdpa_gb"] = -2

        clear_gpu()
        def doc_mask8(b, h, qi, kvi):
            return (qi >= kvi) & (doc_ids8[qi] == doc_ids8[kvi])
        bm3 = create_block_mask(doc_mask8, B, 1, S, S, device=DEVICE)
        _ = flex_attention(q, k, v, block_mask=bm3)
        torch.cuda.synchronize()
        row3["flex_ms"] = round(measure(lambda: flex_attention(q, k, v, block_mask=bm3)), 3)
        row3["flex_gb"] = round(get_peak_memory(), 3)

        results.append(row3)
        print(f"  S={S} doc_8: V={row3['vanilla_ms']}ms SDPA=N/A F={row3['flex_ms']}ms")

        # --- Sliding Window ---
        clear_gpu()
        row4 = {"S": S, "pattern": "sliding_window"}

        clear_gpu()
        def vanilla_sw():
            pos = torch.arange(S, device=DEVICE)
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            causal = pos.unsqueeze(0) >= pos.unsqueeze(1)
            window = (pos.unsqueeze(0) - pos.unsqueeze(1)) <= 256
            scores = scores.masked_fill(~(causal & window), float('-inf'))
            return torch.matmul(F.softmax(scores.float(), dim=-1).to(DTYPE), v)
        row4["vanilla_ms"] = round(measure(vanilla_sw), 3)
        row4["vanilla_gb"] = round(get_peak_memory(), 3)
        row4["sdpa_ms"] = -2
        row4["sdpa_gb"] = -2

        clear_gpu()
        def sw_mask(b, h, qi, kvi):
            return (qi >= kvi) & ((qi - kvi) <= 256)
        bm4 = create_block_mask(sw_mask, B, 1, S, S, device=DEVICE)
        _ = flex_attention(q, k, v, block_mask=bm4)
        torch.cuda.synchronize()
        row4["flex_ms"] = round(measure(lambda: flex_attention(q, k, v, block_mask=bm4)), 3)
        row4["flex_gb"] = round(get_peak_memory(), 3)

        results.append(row4)
        print(f"  S={S} sw_256: V={row4['vanilla_ms']}ms SDPA=N/A F={row4['flex_ms']}ms")

        del q, k, v, bm, bm2, bm3, bm4
        clear_gpu()

    return results


# ============================================================
# Experiment F7: Autograd Profiling
# ============================================================
def experiment_f7():
    print("\n" + "=" * 70)
    print("Experiment F7: Autograd Profiling")
    print("=" * 70)

    B, H, S, D = 1, 8, 2048, 64
    results = {}

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
            def causal_mask_fn(b, h, qi, kvi):
                return qi >= kvi
            bm = create_block_mask(causal_mask_fn, B, 1, S, S, device=DEVICE)
            def fwd():
                return flex_attention(q, k, v, block_mask=bm)
            _ = fwd()
            torch.cuda.synchronize()

        clear_gpu()
        fwd_time = measure(fwd)
        fwd_mem = get_peak_memory()
        info["fwd_ms"] = round(fwd_time, 3)
        info["fwd_gb"] = round(fwd_mem, 3)

        clear_gpu()
        bwd_time = measure(lambda: fwd().sum().backward())
        bwd_mem = get_peak_memory()
        info["bwd_ms"] = round(bwd_time, 3)
        info["bwd_gb"] = round(bwd_mem, 3)
        info["fwd_bwd_ratio"] = round(bwd_time / fwd_time, 2) if fwd_time > 0 else 0
        info["total_ms"] = round(fwd_time + bwd_time, 3)

        results[path_name] = info
        print(f"  {path_name}: fwd={fwd_time:.1f}ms bwd={bwd_time:.1f}ms total={fwd_time+bwd_time:.1f}ms")

    return results


# ============================================================
# Experiment F8: Numerical Precision Deep Dive
# ============================================================
def experiment_f8():
    print("\n" + "=" * 70)
    print("Experiment F8: Numerical Precision Deep Dive")
    print("=" * 70)

    B, H, S = 1, 8, 2048
    results = []

    for D in [32, 64, 128]:
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
            clear_gpu()
            try:
                q = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
                k = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
                v = torch.randn(B, H, S, D, device=DEVICE, dtype=dtype)
            except Exception:
                continue

            # FP32 reference
            q32, k32, v32 = q.float(), k.float(), v.float()
            scores32 = torch.matmul(q32, k32.transpose(-2, -1)) / (D**0.5)
            causal = torch.ones(S, S, device=DEVICE, dtype=torch.bool).tril_()
            scores32 = scores32.masked_fill(~causal, float('-inf'))
            ref = torch.matmul(F.softmax(scores32.float(), dim=-1), v32)

            # Vanilla
            scores = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
            scores = scores.masked_fill(~causal, float('-inf'))
            vanilla_out = torch.matmul(F.softmax(scores.float(), dim=-1).to(dtype), v)

            # SDPA
            sdpa_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

            # Flex
            def causal_mask_fn(b, h, qi, kvi):
                return qi >= kvi
            bm = create_block_mask(causal_mask_fn, B, 1, S, S, device=DEVICE)
            flex_out = flex_attention(q, k, v, block_mask=bm)

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
# Chart Generation (all 8 charts)
# ============================================================
def generate_all_charts():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

    # Read existing data from log - reconstruct key results
    # F1 data from log
    f1_vanilla_steps = {
        "1_qkt": 174.342, "2_causal": 8.195, "3_mask_apply": 73.329,
        "4_softmax": 24.191, "5_xv": 0.555
    }
    f1_sdpa_flash2 = 0.115
    f1_flex_cached = 12.575
    f1_flex_compile = 351.575
    f1_accuracy = {
        "vanilla_vs_sdpa": 0.001953,
        "vanilla_vs_flex": 0.000000,
        "sdpa_vs_flex": 0.001953,
    }

    # F2 data from log
    f2_data = [
        {"S": 512, "pattern": "causal", "compile_overhead_ms": 297, "cached_ms": 2.2, "sdpa_ms": 0.04},
        {"S": 512, "pattern": "sliding_window", "compile_overhead_ms": 205, "cached_ms": 2.3, "sdpa_ms": 0.1},
        {"S": 512, "pattern": "doc_packing_4", "compile_overhead_ms": 210, "cached_ms": 5.9, "sdpa_ms": 0.04},
        {"S": 1024, "pattern": "causal", "compile_overhead_ms": 1, "cached_ms": 3.1, "sdpa_ms": 0.07},
        {"S": 1024, "pattern": "sliding_window", "compile_overhead_ms": 1, "cached_ms": 3.3, "sdpa_ms": 0.07},
        {"S": 1024, "pattern": "doc_packing_4", "compile_overhead_ms": 222, "cached_ms": 6.8, "sdpa_ms": 0.07},
        {"S": 2048, "pattern": "causal", "compile_overhead_ms": 0, "cached_ms": 12.6, "sdpa_ms": 0.11},
        {"S": 2048, "pattern": "sliding_window", "compile_overhead_ms": 0, "cached_ms": 12.5, "sdpa_ms": 0.11},
        {"S": 2048, "pattern": "doc_packing_4", "compile_overhead_ms": 1, "cached_ms": 14.4, "sdpa_ms": 0.11},
        {"S": 4096, "pattern": "causal", "compile_overhead_ms": 0, "cached_ms": 48.1, "sdpa_ms": 0.32},
        {"S": 4096, "pattern": "sliding_window", "compile_overhead_ms": 6, "cached_ms": 48.3, "sdpa_ms": 0.32},
        {"S": 4096, "pattern": "doc_packing_4", "compile_overhead_ms": 7, "cached_ms": 47.2, "sdpa_ms": 0.32},
    ]

    # F3 data
    f3_data = {
        "causal": {"utilization_pct": 6.2, "compression_ratio": 1285.0},
        "sw_64": {"utilization_pct": 12.1, "compression_ratio": 1285.0},
        "sw_128": {"utilization_pct": 12.1, "compression_ratio": 1285.0},
        "sw_256": {"utilization_pct": 11.7, "compression_ratio": 1285.0},
        "sw_512": {"utilization_pct": 10.9, "compression_ratio": 1285.0},
        "doc_2": {"utilization_pct": 6.2, "compression_ratio": 1285.0},
        "doc_4": {"utilization_pct": 6.2, "compression_ratio": 1285.0},
        "doc_8": {"utilization_pct": 6.2, "compression_ratio": 1285.0},
        "doc_16": {"utilization_pct": 6.2, "compression_ratio": 1285.0},
        "prefix_10pct": {"utilization_pct": 6.2, "compression_ratio": 1285.0},
        "prefix_25pct": {"utilization_pct": 4.7, "compression_ratio": 1285.0},
        "prefix_50pct": {"utilization_pct": 3.1, "compression_ratio": 1285.0},
    }

    # F4 data
    f4_data = {
        256: {"vanilla_steps": {"input": 0, "after_qkt": 0.008, "after_causal": 0.013, "after_mask": 0.014, "after_softmax": 0.015, "after_xv": 0.015}, "sdpa_peak_gb": 0.012, "flex_peak_gb": 0.019},
        512: {"vanilla_steps": {"input": 0, "after_qkt": 0.021, "after_causal": 0.027, "after_mask": 0.030, "after_softmax": 0.033, "after_xv": 0.033}, "sdpa_peak_gb": 0.020, "flex_peak_gb": 0.048},
        1024: {"vanilla_steps": {"input": 0, "after_qkt": 0.056, "after_causal": 0.065, "after_mask": 0.083, "after_softmax": 0.102, "after_xv": 0.102}, "sdpa_peak_gb": 0.050, "flex_peak_gb": 0.160},
        2048: {"vanilla_steps": {"input": 0, "after_qkt": 0.194, "after_causal": 0.210, "after_mask": 0.297, "after_softmax": 0.374, "after_xv": 0.374}, "sdpa_peak_gb": 0.160, "flex_peak_gb": 0.599},
        4096: {"vanilla_steps": {"input": 0, "after_qkt": 0.753, "after_causal": 0.779, "after_mask": 1.091, "after_softmax": 1.453, "after_xv": 1.453}, "sdpa_peak_gb": 0.588, "flex_peak_gb": 2.339},
        8192: {"vanilla_steps": {"input": 0, "after_qkt": 2.975, "after_causal": 3.027, "after_mask": 4.236, "after_softmax": 5.751, "after_xv": 5.751}, "sdpa_peak_gb": 2.274, "flex_peak_gb": 9.266},
    }

    # F5 data
    f5_data = [
        {"S": 256, "flash2_ms": 0.039, "math_ms": 0.322, "efficient_ms": 0.037},
        {"S": 512, "flash2_ms": 0.040, "math_ms": 0.336, "efficient_ms": 0.048},
        {"S": 1024, "flash2_ms": 0.067, "math_ms": 1.509, "efficient_ms": 0.073},
        {"S": 2048, "flash2_ms": 0.112, "math_ms": 6.771, "efficient_ms": 0.171},
        {"S": 4096, "flash2_ms": 0.324, "math_ms": 28.866, "efficient_ms": 0.525},
        {"S": 8192, "flash2_ms": 1.056, "math_ms": 113.624, "efficient_ms": 1.837},
        {"S": 16384, "flash2_ms": 4.271, "math_ms": 463.754, "efficient_ms": 8.696},
        {"S": 32768, "flash2_ms": 19.885, "math_ms": -1, "efficient_ms": 33.433},
    ]

    # ---- Chart F1 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    step_names = ["QK^T", "Causal Mask", "Apply Mask", "Softmax", "x V"]
    step_times = [f1_vanilla_steps["1_qkt"], f1_vanilla_steps["2_causal"],
                  f1_vanilla_steps["3_mask_apply"], f1_vanilla_steps["4_softmax"],
                  f1_vanilla_steps["5_xv"]]
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']
    axes[0].bar(step_names, step_times, color=colors)
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Vanilla Path: Per-Step Latency (S=2048)')
    axes[0].grid(axis='y', alpha=0.3)
    for i, t in enumerate(step_times):
        axes[0].text(i, t + 0.5, f'{t:.1f}', ha='center', fontsize=8)

    vanilla_total = sum(step_times)
    paths = ["Vanilla\n(5 kernels)", "SDPA/Flash2\n(1 kernel)", "Flex\n(1 kernel)"]
    path_times = [vanilla_total, f1_sdpa_flash2, f1_flex_cached]
    colors2 = ['#e74c3c', '#2ecc71', '#3498db']
    axes[1].bar(paths, path_times, color=colors2)
    axes[1].set_ylabel('Total Time (ms)')
    axes[1].set_title('Three Paths: Total Latency (S=2048)')
    axes[1].grid(axis='y', alpha=0.3)
    for i, t in enumerate(path_times):
        axes[1].text(i, t + 1, f'{t:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/F1_three_path_trace.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/F1_three_path_trace.png")

    # ---- Chart F2 ----
    fig, ax = plt.subplots(figsize=(12, 6))
    for pname in ["causal", "sliding_window", "doc_packing_4"]:
        sub = [r for r in f2_data if r["pattern"] == pname]
        s_vals = [r["S"] for r in sub]
        cached = [r["cached_ms"] for r in sub]
        sdpa = [r["sdpa_ms"] for r in sub]
        ax.plot(s_vals, cached, marker='o', label=f'{pname} (Flex cached)')
        ax.plot(s_vals, sdpa, marker='s', linestyle='--', label=f'{pname} (SDPA)')
    ax.set_xlabel('Sequence Length (S)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Flex vs SDPA: Cached Performance')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/F2_compile_overhead.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/F2_compile_overhead.png")

    # ---- Chart F3 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    names = list(f3_data.keys())
    utils = [f3_data[n]["utilization_pct"] for n in names]
    compressions = [f3_data[n]["compression_ratio"] for n in names]

    colors = plt.cm.RdYlGn_r([u / 15 for u in utils])
    axes[0].bar(range(len(names)), utils, color=colors)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    axes[0].set_ylabel('Block Utilization (%)')
    axes[0].set_title('BlockMask Utilization by Pattern (S=2048)')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(range(len(names)), compressions, color='#3498db')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    axes[1].set_ylabel('Compression Ratio (x)')
    axes[1].set_title('BlockMask Compression vs Full SxS Mask')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/F3_blockmask_structure.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/F3_blockmask_structure.png")

    # ---- Chart F4 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for S, data in f4_data.items():
        steps = data["vanilla_steps"]
        labels = list(steps.keys())
        mems = list(steps.values())
        axes[0].plot(labels, mems, marker='o', label=f'S={S}')
    axes[0].set_ylabel('Cumulative Peak Memory (GB)')
    axes[0].set_title('Vanilla: Memory Build-Up at Each Step')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].tick_params(axis='x', rotation=30)

    s_vals = list(f4_data.keys())
    v_peaks = [f4_data[s]["vanilla_steps"]["after_xv"] for s in s_vals]
    s_peaks = [f4_data[s]["sdpa_peak_gb"] for s in s_vals]
    f_peaks = [f4_data[s]["flex_peak_gb"] for s in s_vals]
    axes[1].plot(s_vals, v_peaks, 'o-', label='Vanilla', color='#e74c3c')
    axes[1].plot(s_vals, s_peaks, 's--', label='SDPA', color='#2ecc71')
    axes[1].plot(s_vals, f_peaks, '^-.', label='Flex', color='#3498db')
    axes[1].set_xlabel('Sequence Length (S)')
    axes[1].set_ylabel('Peak Memory (GB)')
    axes[1].set_title('Peak Memory: Three Paths')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xscale('log', base=2)
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/F4_memory_waterfall.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/F4_memory_waterfall.png")

    # ---- Chart F5 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for bname, color, marker in [("flash2", "#2ecc71", "o"),
                                  ("math", "#e74c3c", "s"),
                                  ("efficient", "#3498db", "^")]:
        s_vals = [r["S"] for r in f5_data if r.get(f"{bname}_ms", -1) > 0]
        times = [r[f"{bname}_ms"] for r in f5_data if r.get(f"{bname}_ms", -1) > 0]
        if times:
            axes[0].plot(s_vals, times, marker=marker, color=color, label=bname)
    axes[0].set_xlabel('Sequence Length (S)')
    axes[0].set_ylabel('Latency (ms)')
    axes[0].set_title('SDPA Backend Latency')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_xscale('log', base=2)
    axes[0].set_yscale('log')

    # Bar chart at S=2048
    f5_2048 = [r for r in f5_data if r["S"] == 2048][0]
    backends = ["Flash2", "Math", "Efficient"]
    times_2048 = [f5_2048["flash2_ms"], f5_2048["math_ms"], f5_2048["efficient_ms"]]
    colors3 = ['#2ecc71', '#e74c3c', '#3498db']
    axes[1].bar(backends, times_2048, color=colors3)
    axes[1].set_ylabel('Latency (ms)')
    axes[1].set_title('SDPA Backend at S=2048')
    axes[1].grid(axis='y', alpha=0.3)
    for i, t in enumerate(times_2048):
        axes[1].text(i, t + 0.1, f'{t:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/F5_sdpa_backends.png', bbox_inches='tight')
    plt.close()
    print(f"Saved {FIGURES_DIR}/F5_sdpa_backends.png")

    print("\nCharts F1-F5 generated! F6-F8 will be generated after running experiments.")


# ============================================================
# Generate F6-F8 charts from experiment results
# ============================================================
def generate_f6_f8_charts(f6, f7, f8):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

    # ---- Chart F6 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for pname in set(r["pattern"] for r in f6):
        for impl, color, marker, ls in [("vanilla", "#e74c3c", "o", "-"),
                                         ("sdpa", "#2ecc71", "s", "--"),
                                         ("flex", "#3498db", "^", "-.")]:
            sub = [r for r in f6 if r["pattern"] == pname and r.get(f"{impl}_ms", -2) > 0]
            if not sub:
                continue
            s_vals = [r["S"] for r in sub]
            times = [r[f"{impl}_ms"] for r in sub]
            label = f'{pname} ({impl})'
            axes[0].plot(s_vals, times, marker=marker, linestyle=ls, color=color, label=label, alpha=0.7)
    axes[0].set_xlabel('Sequence Length (S)')
    axes[0].set_ylabel('Latency (ms)')
    axes[0].set_title('Three-Way Latency Comparison')
    axes[0].legend(fontsize=7, loc='upper left')
    axes[0].grid(alpha=0.3)
    axes[0].set_xscale('log', base=2)
    axes[0].set_yscale('log')

    for pname in set(r["pattern"] for r in f6):
        for impl, color, marker, ls in [("vanilla", "#e74c3c", "o", "-"),
                                         ("sdpa", "#2ecc71", "s", "--"),
                                         ("flex", "#3498db", "^", "-.")]:
            sub = [r for r in f6 if r["pattern"] == pname and r.get(f"{impl}_gb", -2) > 0]
            if not sub:
                continue
            s_vals = [r["S"] for r in sub]
            mems = [r[f"{impl}_gb"] for r in sub]
            label = f'{pname} ({impl})'
            axes[1].plot(s_vals, mems, marker=marker, linestyle=ls, color=color, label=label, alpha=0.7)
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

    # ---- Chart F7 ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    path_names = list(f7.keys())
    fwd_times = [f7[p]["fwd_ms"] for p in path_names]
    bwd_times = [f7[p]["bwd_ms"] for p in path_names]
    colors3 = ['#e74c3c', '#2ecc71', '#3498db']

    axes[0].bar(path_names, fwd_times, color=colors3)
    axes[0].set_title('Forward Pass Time (ms)')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(path_names, bwd_times, color=colors3)
    axes[1].set_title('Backward Pass Time (ms)')
    axes[1].grid(axis='y', alpha=0.3)

    x = range(len(path_names))
    axes[2].bar(x, fwd_times, color=colors3, label='Forward')
    axes[2].bar(x, bwd_times, bottom=fwd_times, color=['#c0392b', '#16a085', '#2980b9'], label='Backward')
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

    # ---- Chart F8 ----
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

    print("\nAll charts generated!")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Source Analysis: F6-F8 + Chart Generation")
    print("=" * 70)

    # Generate F1-F5 charts from log data
    generate_all_charts()

    # Run F6-F8
    f6 = experiment_f6()
    f7 = experiment_f7()
    f8 = experiment_f8()

    # Generate F6-F8 charts
    generate_f6_f8_charts(f6, f7, f8)

    # Save results
    results = {"F6": f6, "F7": f7, "F8": f8}
    with open("source_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to source_analysis_results.json")
    print("DONE!")
