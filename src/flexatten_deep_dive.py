#!/usr/bin/env python3
"""
FlexAttention 
===================================
 A:  FlexAttention  ---- 
 B: FlexAttention 
 C: FlexAttention 
"""

import torch
import torch.nn.functional as F
import json
import time
import gc
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

device = "cuda"
dtype = torch.float16
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

DATA = {}

def clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def bench(fn, warmup=3, runs=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        torch.cuda.synchronize()
        s = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - s) * 1000)
    return round(sum(ts)/len(ts), 3)

def peak_mem(fn):
    clear()
    torch.cuda.reset_peak_memory_stats()
    r = fn()
    torch.cuda.synchronize()
    m = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
    del r
    return m

# ================================================================
# A1:  ---- S^2 
# ================================================================
def exp_A1_memory_explosion():
    print("\n" + "="*70)
    print("  A1:  ---- Standard Attention  O(S^2) ")
    print("="*70)
    H, D, B = 8, 64, 1

    #  SxS 
    theory = {}
    for S in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        # : S*S*2bytes(fp16), 5(scores, causal_mask, combined, softmax, output)
        bytes_per_matrix = S * S * 2
        total_bytes = bytes_per_matrix * 5  # scores + masks + softmax + etc
        theory[S] = round(total_bytes / 1024**3, 3)

    #  Standard / SDPA / Flex 
    actual_std, actual_sdpa, actual_flex = {}, {}, {}
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    for S in [512, 1024, 2048, 4096, 8192]:
        print(f"  [S={S}]")
        clear()
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)

        # Standard
        try:
            def std_fn():
                sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
                mask = torch.ones(S, S, device=device, dtype=torch.bool).tril_()
                sc = sc.masked_fill(~mask, float('-inf'))
                w = F.softmax(sc.float(), dim=-1).to(dtype)
                return torch.matmul(w, v)
            actual_std[S] = peak_mem(std_fn)
            print(f"    Standard: {actual_std[S]:.3f} GB")
        except RuntimeError:
            actual_std[S] = None
            print(f"    Standard: OOM")
            clear()

        # SDPA
        try:
            actual_sdpa[S] = peak_mem(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
            print(f"    SDPA:     {actual_sdpa[S]:.3f} GB")
        except RuntimeError:
            actual_sdpa[S] = None
            clear()

        # Flex
        try:
            def cm(b,h,qi,ki): return qi >= ki
            bm = create_block_mask(cm, B, 1, S, S, device=device)
            _ = flex_attention(q, k, v, block_mask=bm); torch.cuda.synchronize()
            actual_flex[S] = peak_mem(lambda: flex_attention(q, k, v, block_mask=bm))
            print(f"    Flex:     {actual_flex[S]:.3f} GB")
        except RuntimeError:
            actual_flex[S] = None
            clear()

        del q, k, v; clear()

    # ===  A1:  ===
    fig, ax = plt.subplots(figsize=(10, 6))
    seqs = [512, 1024, 2048, 4096, 8192]
    x = np.arange(len(seqs))
    w = 0.25

    vals_std = [actual_std.get(s) for s in seqs]
    vals_sdpa = [actual_sdpa.get(s) for s in seqs]
    vals_flex = [actual_flex.get(s) for s in seqs]
    vals_theory = [theory[s] for s in seqs]

    ax.bar(x - 1.5*w, vals_theory, w, label='Theory (5x S^2 mats, fp16)', color='#ff6b6b', alpha=0.6, hatch='//')
    ax.bar(x - 0.5*w, [v if v else 0 for v in vals_std], w, label='Standard Attention', color='#e74c3c')
    ax.bar(x + 0.5*w, [v if v else 0 for v in vals_sdpa], w, label='SDPA / FlashAttention2', color='#2ecc71')
    ax.bar(x + 1.5*w, [v if v else 0 for v in vals_flex], w, label='FlexAttention', color='#3498db')

    ax.axhline(y=22.0, color='red', linestyle='--', linewidth=2, label='L4 VRAM Limit (22 GB)')

    ax.set_xlabel('Sequence Length S')
    ax.set_ylabel('Peak Memory (GB)')
    ax.set_title('A1: Memory Explosion - Standard O(S^2) vs SDPA/Flex')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seqs])
    ax.legend(loc='upper left')
    ax.set_ylim(0, 12)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'A1_memory_explosion.png')
    plt.close()
    print("  [] A1_memory_explosion.png ")

    DATA["A1"] = {"theory": theory, "std": actual_std, "sdpa": actual_sdpa, "flex": actual_flex}

# ================================================================
# A2:  ---- HBM 
# ================================================================
def exp_A2_bandwidth_starvation():
    print("\n" + "="*70)
    print("  A2: Bandwidth Starvation - Standard vs SDPA")
    print("="*70)
    H, D, B = 8, 64, 1

    std_times, sdpa_times, flex_times = {}, {}, {}
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    for S in [256, 512, 1024, 2048, 4096, 8192]:
        print(f"  [S={S}]")
        clear()
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)

        # Standard: QK^T -> mask -> softmax -> xV (4HBM)
        try:
            def std_fn():
                sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
                mask = torch.ones(S, S, device=device, dtype=torch.bool).tril_()
                sc = sc.masked_fill(~mask, float('-inf'))
                w = F.softmax(sc.float(), dim=-1).to(dtype)
                return torch.matmul(w, v)
            std_times[S] = bench(std_fn, warmup=2, runs=5)
        except: std_times[S] = None

        # SDPA:  Fused kernel
        try:
            sdpa_times[S] = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        except: sdpa_times[S] = None

        # Flex: Triton Fused kernel
        try:
            def cm(b,h,qi,ki): return qi >= ki
            bm = create_block_mask(cm, B, 1, S, S, device=device)
            _ = flex_attention(q, k, v, block_mask=bm); torch.cuda.synchronize()
            flex_times[S] = bench(lambda: flex_attention(q, k, v, block_mask=bm))
        except: flex_times[S] = None

        print(f"    Standard: {std_times.get(S,'OOM')}ms | SDPA: {sdpa_times.get(S,'OOM')}ms | Flex: {flex_times.get(S,'OOM')}ms")
        del q, k, v; clear()

    # ===  A2:  ===
    fig, ax = plt.subplots(figsize=(10, 6))
    seqs = sorted(std_times.keys())
    ax.plot(seqs, [std_times[s] for s in seqs], 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Standard (multi-kernel chain)')
    ax.plot(seqs, [sdpa_times[s] for s in seqs], 's-', color='#2ecc71', linewidth=2, markersize=8, label='SDPA (single Fused kernel)')
    ax.plot(seqs, [flex_times[s] for s in seqs], '^-', color='#3498db', linewidth=2, markersize=8, label='FlexAttention (Triton Fused)')

    ax.set_xlabel('Sequence Length S')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('A2: Bandwidth Starvation - Multi HBM Trips vs Fused Kernel')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'A2_bandwidth_starvation.png')
    plt.close()
    print("  [] A2_bandwidth_starvation.png ")

    DATA["A2"] = {"std": std_times, "sdpa": sdpa_times, "flex": flex_times}

# ================================================================
# A3:  ---- 
# ================================================================
def exp_A3_engineering_nightmare():
    print("\n" + "="*70)
    print("  A3:  ----  Document + SlidingWindow + ALiBi ")
    print("="*70)
    S, H, D, B = 2048, 8, 64, 1
    WINDOW = 256
    num_docs = 4
    doc_ids = (torch.arange(S, device=device) // (S // num_docs)).clamp(max=num_docs - 1)
    alibi_slopes = 1.0 / (2.0 ** (8 + torch.arange(H, device=device, dtype=dtype)))

    clear()
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    # === Vanilla  ===
    print("  [Vanilla ] 7...")
    def vanilla_combined():
        # Step 1: QK^T
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
        # Step 2: Causal mask
        causal = torch.ones(S, S, device=device, dtype=torch.bool).tril_()
        # Step 3: Sliding window mask
        pos = torch.arange(S, device=device)
        sw = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs() <= WINDOW
        # Step 4: Document mask
        dm = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
        # Step 5: Combine masks
        combined = causal & sw & dm
        # Step 6: Apply ALiBi
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().to(dtype)
        for h in range(H):
            scores[:, h] -= alibi_slopes[h] * dist
        # Step 7: Mask + softmax + output
        scores = scores.masked_fill(~combined, float('-inf'))
        w = F.softmax(scores.float(), dim=-1).to(dtype)
        return torch.matmul(w, v)

    mem_vanilla = peak_mem(vanilla_combined)
    time_vanilla = bench(vanilla_combined, warmup=1, runs=3)
    print(f"    : {time_vanilla:.3f} ms | : {mem_vanilla:.3f} GB")

    # === Flex  ===
    print("  [Flex ] 3...")
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    # mask_mod:  bool
    def combined_mask(b, h, q_idx, kv_idx):
        causal_ok = q_idx >= kv_idx
        sw_ok = (q_idx - kv_idx) <= WINDOW
        doc_ok = doc_ids[q_idx] == doc_ids[kv_idx]
        return causal_ok & sw_ok & doc_ok

    # score_mod: 
    def alibi_mod(score, b, h, q_idx, kv_idx):
        return score - alibi_slopes[h] * (q_idx - kv_idx).abs().to(dtype)

    bm = create_block_mask(combined_mask, B, 1, S, S, device=device)
    _ = flex_attention(q, k, v, score_mod=alibi_mod, block_mask=bm)
    torch.cuda.synchronize()

    mem_flex = peak_mem(lambda: flex_attention(q, k, v, score_mod=alibi_mod, block_mask=bm))
    time_flex = bench(lambda: flex_attention(q, k, v, score_mod=alibi_mod, block_mask=bm), warmup=2, runs=5)
    print(f"    : {time_flex:.3f} ms | : {mem_flex:.3f} GB")

    # 
    out_v = vanilla_combined()
    out_f = flex_attention(q, k, v, score_mod=alibi_mod, block_mask=bm)
    diff = round((out_v.float() - out_f.float()).abs().max().item(), 6)
    print(f"    : {diff}")

    # ===  A3:  vs  ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # :
    categories = ['Vanilla\n()', 'Flex\n()']
    steps = [7, 3]
    colors = ['#e74c3c', '#3498db']
    bars = ax1.bar(categories, steps, color=colors, width=0.5, edgecolor='black')
    ax1.set_ylabel('Implementation Steps')
    ax1.set_title('A3-1: Code Complexity - Doc+SW+ALiBi Combo')
    for bar, s in zip(bars, steps):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, str(s), ha='center', fontsize=16, fontweight='bold')
    ax1.set_ylim(0, 9)
    ax1.grid(axis='y', alpha=0.3)

    # :
    metrics = ['Latency (ms)', 'Memory (GB)']
    vanilla_vals = [time_vanilla, mem_vanilla * 1000]  # scale for visibility
    flex_vals = [time_flex, mem_flex * 1000]
    x = np.arange(len(metrics))
    w = 0.3
    ax2_twin = ax2.twinx()
    b1 = ax2.bar(x - w/2, [time_vanilla, mem_vanilla], w, label='Vanilla', color='#e74c3c')
    b2 = ax2.bar(x + w/2, [time_flex, mem_flex], w, label='Flex', color='#3498db')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_title('A3-2: Performance - Combined Attention Mode')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    for bar in b1:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f'{bar.get_height():.1f}', ha='center', fontsize=10)
    for bar in b2:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f'{bar.get_height():.1f}', ha='center', fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'A3_engineering_nightmare.png')
    plt.close()
    print("  [] A3_engineering_nightmare.png ")

    DATA["A3"] = {"vanilla": {"time": time_vanilla, "mem": mem_vanilla},
                  "flex": {"time": time_flex, "mem": mem_flex}, "diff": diff}

    del q, k, v; clear()

# ================================================================
# B1: score_mod 
# ================================================================
def exp_B1_score_mod_fusion():
    print("\n" + "="*70)
    print("  B1: score_mod Fusion - Write-Modify vs Fused")
    print("="*70)
    H, D, B = 8, 64, 1
    S = 2048
    alibi_slopes = 1.0 / (2.0 ** (8 + torch.arange(H, device=device, dtype=dtype)))

    clear()
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    # 1: (Standard)----  QK^T HBM,  bias HBM
    def write_then_modify():
        sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)  # HBM
        pos = torch.arange(S, device=device, dtype=dtype)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        for h in range(H):
            sc[:, h] -= alibi_slopes[h] * dist  # HBM -> bias -> HBM
        cm = torch.ones(S, S, device=device, dtype=torch.bool).tril_()
        sc = sc.masked_fill(~cm, float('-inf'))
        w = F.softmax(sc.float(), dim=-1).to(dtype)
        return torch.matmul(w, v)

    # 2: (Flex)---- score_mod  SRAM 
    def alibi_mod(score, b, h, q_idx, kv_idx):
        return score - alibi_slopes[h] * (q_idx - kv_idx).abs().to(dtype)
    def causal_mod(b, h, q_idx, kv_idx): return q_idx >= kv_idx
    bm = create_block_mask(causal_mod, B, 1, S, S, device=device)
    _ = flex_attention(q, k, v, score_mod=alibi_mod, block_mask=bm)
    torch.cuda.synchronize()

    mem_std = peak_mem(write_then_modify)
    time_std = bench(write_then_modify, warmup=1, runs=5)
    print(f"  Write-modify (Std): {time_std:.3f} ms, {mem_std:.3f} GB")

    mem_flex = peak_mem(lambda: flex_attention(q, k, v, score_mod=alibi_mod, block_mask=bm))
    time_flex = bench(lambda: flex_attention(q, k, v, score_mod=alibi_mod, block_mask=bm), warmup=2, runs=5)
    print(f"  Fused (Flex):   {time_flex:.3f} ms, {mem_flex:.3f} GB")

    # 
    o1 = write_then_modify()
    o2 = flex_attention(q, k, v, score_mod=alibi_mod, block_mask=bm)
    diff = round((o1.float() - o2.float()).abs().max().item(), 6)
    print(f"  : {diff}")

    # ===  B1:  ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # :HBM 
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_title('B1-1: HBM Access Path Comparison')

    # Standard flow
    ax1.text(0.5, 9.5, 'Standard (write-then-modify)', fontsize=11, fontweight='bold', color='#e74c3c')
    steps_std = ['QK^T -> HBM', 'Read HBM +Bias', 'Write HBM', 'Mask -> HBM', 'Softmax -> HBM', 'xV -> HBM']
    for i, s in enumerate(steps_std):
        y = 8.5 - i * 1.2
        ax1.add_patch(Rectangle((0.2, y-0.3), 4.0, 0.6, facecolor='#ffe0e0', edgecolor='#e74c3c'))
        ax1.text(2.2, y, s, ha='center', va='center', fontsize=8)
        if i < len(steps_std)-1:
            ax1.annotate('', xy=(2.2, y-0.5), xytext=(2.2, y-0.3), arrowprops=dict(arrowstyle='->', color='#e74c3c'))

    # Flex flow
    ax1.text(5.5, 9.5, 'Flex (fused computation)', fontsize=11, fontweight='bold', color='#3498db')
    ax1.add_patch(Rectangle((5.3, 6.7), 4.3, 2.3, facecolor='#e0f0ff', edgecolor='#3498db', linewidth=2))
    ax1.text(7.45, 8.5, 'Single Fused Kernel', ha='center', va='center', fontsize=10, fontweight='bold', color='#3498db')
    ax1.text(7.45, 7.8, 'QK^T + Bias + Mask', ha='center', va='center', fontsize=9)
    ax1.text(7.45, 7.2, '+ Softmax + xV', ha='center', va='center', fontsize=9)
    ax1.annotate('', xy=(7.45, 4.5), xytext=(7.45, 6.7), arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))
    ax1.add_patch(Rectangle((5.8, 3.9), 3.3, 0.6, facecolor='#e0ffe0', edgecolor='#2ecc71'))
    ax1.text(7.45, 4.2, 'Output -> HBM', ha='center', va='center', fontsize=9)

    ax1.text(2.2, 1.5, '6x HBM writes', ha='center', fontsize=11, color='#e74c3c', fontweight='bold')
    ax1.text(7.45, 3.0, '1x HBM write', ha='center', fontsize=11, color='#2ecc71', fontweight='bold')

    ax1.axis('off')

    # :
    labels = ['Latency (ms)', 'Memory (GB)']
    std_vals = [time_std, mem_std]
    flex_vals = [time_flex, mem_flex]
    x = np.arange(len(labels))
    w = 0.3
    ax2.bar(x - w/2, std_vals, w, label='Write-modify (Std)', color='#e74c3c')
    ax2.bar(x + w/2, flex_vals, w, label='Fused (Flex)', color='#3498db')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title('B1-2: score_mod Fused vs Write-Modify (ALiBi, S=2048)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    for i, (sv, fv) in enumerate(zip(std_vals, flex_vals)):
        ax2.text(i - w/2, sv + 0.1, f'{sv:.1f}', ha='center', fontsize=9)
        ax2.text(i + w/2, fv + 0.1, f'{fv:.1f}', ha='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'B1_score_mod_fusion.png')
    plt.close()
    print("  [] B1_score_mod_fusion.png ")

    DATA["B1"] = {"std": {"time": time_std, "mem": mem_std},
                  "flex": {"time": time_flex, "mem": mem_flex}, "diff": diff}

    del q, k, v; clear()

# ================================================================
# B2: BlockMask  ---- 
# ================================================================
def exp_B2_block_mask_visualization():
    print("\n" + "="*70)
    print("  B2: BlockMask Sparsity Visualization")
    print("="*70)
    S = 512  #  S 
    BS = 128  # BlockSize
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    #  mask  boolean 
    pos = torch.arange(S)

    masks_data = {}

    # Causal
    causal = (pos.unsqueeze(0) >= pos.unsqueeze(1)).numpy()
    masks_data["Causal"] = causal

    # Sliding Window
    sw = ((pos.unsqueeze(0) >= pos.unsqueeze(1)) & ((pos.unsqueeze(0) - pos.unsqueeze(1)) <= 64)).numpy()
    masks_data["SlidingWindow(64)"] = sw

    # Document (4 docs)
    doc_ids = (pos // (S // 4)).clamp(max=3)
    doc_causal = ((pos.unsqueeze(0) >= pos.unsqueeze(1)) & (doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1))).numpy()
    masks_data["Document(4) + Causal"] = doc_causal

    # Prefix LM
    prefix = ((pos.unsqueeze(1) < S // 4) | (pos.unsqueeze(0) >= pos.unsqueeze(1))).numpy()
    masks_data["PrefixLM(25%)"] = prefix

    # ===  B2: 4  mask  +  ===
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('B2: BlockMask Sparsity - Pixel vs Block (128x128)', fontsize=14)

    for col, (name, mask) in enumerate(masks_data.items()):
        # : mask
        ax = axes[0, col]
        ax.imshow(mask, cmap='RdYlGn', interpolation='nearest', aspect='equal')
        ax.set_title(f'{name}\n(pixel-level)', fontsize=10)
        ax.set_xlabel('KV Position')
        ax.set_ylabel('Q Position')

        # : mask
        ax2 = axes[1, col]
        num_blocks = S // BS
        block_mask = np.zeros((num_blocks, num_blocks))
        for bi in range(num_blocks):
            for bj in range(num_blocks):
                block = mask[bi*BS:(bi+1)*BS, bj*BS:(bj+1)*BS]
                if block.all():
                    block_mask[bi, bj] = 1  # full
                elif not block.any():
                    block_mask[bi, bj] = 0  # empty
                else:
                    block_mask[bi, bj] = 0.5  # partial

        cmap = plt.cm.RdYlGn
        cmap.set_under('black')
        ax2.imshow(block_mask, cmap=cmap, vmin=0, vmax=1, interpolation='nearest', aspect='equal')

        # 
        for i in range(num_blocks + 1):
            ax2.axhline(i - 0.5, color='gray', linewidth=0.5)
            ax2.axvline(i - 0.5, color='gray', linewidth=0.5)

        sparsity = 1.0 - mask.sum() / (S * S)
        ax2.set_title(f'Block (sparsity: {sparsity:.1%})\nGreen=compute Red=skip Yellow=partial', fontsize=9)
        ax2.set_xlabel('KV Block')
        ax2.set_ylabel('Q Block')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'B2_block_mask_visualization.png')
    print("  [] B2_block_mask_visualization.png ")

    #  mask 
    sparsity_data = {}
    for name, mask in masks_data.items():
        total = S * S
        active = mask.sum()
        sparsity_data[name] = round(1.0 - active / total, 4)
        print(f"    {name}:  = {sparsity_data[name]:.1%}")

    DATA["B2"] = {"sparsity": sparsity_data}

# ================================================================
# C1:  ---- Vanilla vs Flex 
# ================================================================
def exp_C1_code_comparison():
    print("\n" + "="*70)
    print("  C1: Code Comparison - Vanilla vs Flex (5 patterns)")
    print("="*70)
    S, H, D, B = 2048, 8, 64, 1
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    clear()
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    results = []

    patterns = {
        "Causal": {
            "vanilla_steps": "matmul -> tril_mask -> masked_fill -> softmax -> matmul",
            "flex_lines": "def mask(b,h,q,k): return q >= k",
            "vanilla_complexity": "",
            "flex_complexity": "",
        },
        "SlidingWindow(256)": {
            "vanilla_steps": "matmul -> abs_dist -> window_mask -> masked_fill -> softmax -> matmul",
            "flex_lines": "def mask(b,h,q,k): return (q >= k) & (q - k <= 256)",
            "vanilla_complexity": "",
            "flex_complexity": "",
        },
        "Document(4)+Causal": {
            "vanilla_steps": "matmul -> tril -> doc_eq -> combine -> masked_fill -> softmax -> matmul",
            "flex_lines": "def mask(b,h,q,k): return (q>=k) & (doc[q]==doc[k])",
            "vanilla_complexity": "",
            "flex_complexity": "",
        },
        "PrefixLM(25%)": {
            "vanilla_steps": "matmul -> prefix_mask -> causal -> combine -> masked_fill -> softmax -> matmul",
            "flex_lines": "def mask(b,h,q,k): return (k<512) | (q>=k)",
            "vanilla_complexity": "",
            "flex_complexity": "",
        },
        "ALiBi+Causal": {
            "vanilla_steps": "matmul -> build_dist_matrix -> loop over H -> add_bias -> tril -> masked_fill -> softmax -> matmul",
            "flex_lines": "def mod(s,b,h,q,k): return s - slope[h]*|q-k|",
            "vanilla_complexity": "",
            "flex_complexity": "",
        },
    }

    # 
    for name in ["Causal", "SlidingWindow(256)", "Document(4)+Causal", "PrefixLM(25%)", "ALiBi+Causal"]:
        print(f"  [{name}]")
        try:
            if name == "Causal":
                t_v = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
                def cm(b,h,qi,ki): return qi >= ki
                bm = create_block_mask(cm, B, 1, S, S, device=device)
                _ = flex_attention(q, k, v, block_mask=bm); torch.cuda.synchronize()
                t_f = bench(lambda: flex_attention(q, k, v, block_mask=bm))
            elif name.startswith("SlidingWindow"):
                def sw_fn():
                    sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
                    pos = torch.arange(S, device=device)
                    d = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
                    sc = sc.masked_fill(~((pos.unsqueeze(0) >= pos.unsqueeze(1)) & (d <= 256)), float('-inf'))
                    return torch.matmul(F.softmax(sc.float(), dim=-1).to(dtype), v)
                t_v = bench(sw_fn, warmup=1, runs=3)
                def swm(b,h,qi,ki): return (qi >= ki) & ((qi - ki) <= 256)
                bm = create_block_mask(swm, B, 1, S, S, device=device)
                _ = flex_attention(q, k, v, block_mask=bm); torch.cuda.synchronize()
                t_f = bench(lambda: flex_attention(q, k, v, block_mask=bm))
            elif name.startswith("Document"):
                doc_ids = (torch.arange(S, device=device) // (S//4)).clamp(max=3)
                def doc_fn():
                    sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
                    pos = torch.arange(S, device=device)
                    dm = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
                    cm = torch.ones(S, S, device=device, dtype=torch.bool).tril_()
                    sc = sc.masked_fill(~(cm & dm), float('-inf'))
                    return torch.matmul(F.softmax(sc.float(), dim=-1).to(dtype), v)
                t_v = bench(doc_fn, warmup=1, runs=3)
                def dm2(b,h,qi,ki):
                    return (qi >= ki) & (doc_ids[qi] == doc_ids[ki])
                bm = create_block_mask(dm2, B, 1, S, S, device=device)
                _ = flex_attention(q, k, v, block_mask=bm); torch.cuda.synchronize()
                t_f = bench(lambda: flex_attention(q, k, v, block_mask=bm))
            elif name.startswith("PrefixLM"):
                def plm_fn():
                    sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
                    pos = torch.arange(S, device=device)
                    pm = pos.unsqueeze(1) < (S//4)
                    cm = pos.unsqueeze(0) >= pos.unsqueeze(1)
                    sc = sc.masked_fill(~(pm | cm), float('-inf'))
                    return torch.matmul(F.softmax(sc.float(), dim=-1).to(dtype), v)
                t_v = bench(plm_fn, warmup=1, runs=3)
                def plm(b,h,qi,ki): return (ki < S//4) | (qi >= ki)
                bm = create_block_mask(plm, B, 1, S, S, device=device)
                _ = flex_attention(q, k, v, block_mask=bm); torch.cuda.synchronize()
                t_f = bench(lambda: flex_attention(q, k, v, block_mask=bm))
            elif name.startswith("ALiBi"):
                slopes = 1.0 / (2.0 ** (8 + torch.arange(H, device=device, dtype=dtype)))
                def alibi_fn():
                    sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
                    pos = torch.arange(S, device=device, dtype=dtype)
                    d = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
                    for h in range(H): sc[:, h] -= slopes[h] * d
                    cm = torch.ones(S, S, device=device, dtype=torch.bool).tril_()
                    sc = sc.masked_fill(~cm, float('-inf'))
                    return torch.matmul(F.softmax(sc.float(), dim=-1).to(dtype), v)
                t_v = bench(alibi_fn, warmup=1, runs=3)
                def am(s,b,h,qi,ki): return s - slopes[h] * (qi - ki).abs().to(dtype)
                def cm2(b,h,qi,ki): return qi >= ki
                bm = create_block_mask(cm2, B, 1, S, S, device=device)
                _ = flex_attention(q, k, v, score_mod=am, block_mask=bm); torch.cuda.synchronize()
                t_f = bench(lambda: flex_attention(q, k, v, score_mod=am, block_mask=bm))

            info = patterns[name]
            info["vanilla_ms"] = t_v
            info["flex_ms"] = t_f
            info["speedup"] = round(t_v / t_f, 2) if t_f > 0 else None
            results.append(info)
            print(f"    Vanilla: {t_v:.3f} ms | Flex: {t_f:.3f} ms | : {t_v/t_f:.2f}x")
        except Exception as e:
            print(f"    Error: {e}")
            results.append({"name": name, "error": str(e)})

    # ===  C1 ===
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r.get("name", k) for k, r in zip(patterns.keys(), results)]
    if not names:
        names = list(patterns.keys())
    v_times = [r.get("vanilla_ms", 0) for r in results]
    f_times = [r.get("flex_ms", 0) for r in results]

    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, v_times, w, label='Vanilla (manual)', color='#e74c3c')
    ax.bar(x + w/2, f_times, w, label='FlexAttention', color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('C1: 5 Attention Patterns - Vanilla vs FlexAttention')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for i, (vt, ft) in enumerate(zip(v_times, f_times)):
        ratio = vt/ft if ft > 0 else 0
        color = '#2ecc71' if ratio < 1 else '#e74c3c'
        label = f'{ratio:.1f}x'
        ax.text(i, max(vt, ft) + 1, label, ha='center', fontsize=10, fontweight='bold', color=color)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'C1_code_comparison.png')
    plt.close()
    print("  [] C1_code_comparison.png ")

    DATA["C1"] = results

    del q, k, v; clear()

# ================================================================
# C2: SDPA  vs Flex  ---- 
# ================================================================
def exp_C2_impossible_for_sdpa():
    print("\n" + "="*70)
    print("  C2: Impossible for SDPA - Easy for Flex")
    print("="*70)
    H, D, B = 8, 64, 1
    S = 2048
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    clear()
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    doc_ids = (torch.arange(S, device=device) // (S // 4)).clamp(max=3)
    slopes = 1.0 / (2.0 ** (8 + torch.arange(H, device=device, dtype=dtype)))

    combos = []

    # 1. Document + SlidingWindow
    print("  [Document + SlidingWindow]")
    def dsw(b,h,qi,ki):
        return (qi >= ki) & ((qi - ki) <= 256) & (doc_ids[qi] == doc_ids[ki])
    bm = create_block_mask(dsw, B, 1, S, S, device=device)
    _ = flex_attention(q, k, v, block_mask=bm); torch.cuda.synchronize()
    t = bench(lambda: flex_attention(q, k, v, block_mask=bm))
    mem = peak_mem(lambda: flex_attention(q, k, v, block_mask=bm))
    total_blks = bm.kv_num_blocks.sum().item()
    max_blks = B * (S // 128) ** 2
    sparse = round(1.0 - total_blks / max_blks, 4)
    print(f"    Flex: {t:.3f} ms, {mem:.3f} GB, sparse={sparse:.1%}")
    print(f"    SDPA:  ( Document+SW )")
    combos.append({"pattern": "Document+SlidingWindow", "flex_ms": t, "flex_mem": mem, "sparsity": sparse, "sdpa": ""})

    # 2. Document + ALiBi
    print("  [Document + ALiBi]")
    def dm(b,h,qi,ki): return (qi >= ki) & (doc_ids[qi] == doc_ids[ki])
    def am(s,b,h,qi,ki): return s - slopes[h] * (qi - ki).abs().to(dtype)
    bm2 = create_block_mask(dm, B, 1, S, S, device=device)
    _ = flex_attention(q, k, v, score_mod=am, block_mask=bm2); torch.cuda.synchronize()
    t2 = bench(lambda: flex_attention(q, k, v, score_mod=am, block_mask=bm2))
    mem2 = peak_mem(lambda: flex_attention(q, k, v, score_mod=am, block_mask=bm2))
    print(f"    Flex: {t2:.3f} ms, {mem2:.3f} GB")
    print(f"    SDPA:  ( score_mod +  mask)")
    combos.append({"pattern": "Document+ALiBi", "flex_ms": t2, "flex_mem": mem2, "sdpa": ""})

    # 3. PrefixLM + Softcapping
    print("  [PrefixLM + Softcapping]")
    CAP = 50.0
    def plm(b,h,qi,ki): return (ki < S//4) | (qi >= ki)
    def sc(s,b,h,qi,ki): return CAP * torch.tanh(s / CAP)
    bm3 = create_block_mask(plm, B, 1, S, S, device=device)
    _ = flex_attention(q, k, v, score_mod=sc, block_mask=bm3); torch.cuda.synchronize()
    t3 = bench(lambda: flex_attention(q, k, v, score_mod=sc, block_mask=bm3))
    mem3 = peak_mem(lambda: flex_attention(q, k, v, score_mod=sc, block_mask=bm3))
    print(f"    Flex: {t3:.3f} ms, {mem3:.3f} GB")
    print(f"    SDPA:  ( Softcapping + PrefixLM)")
    combos.append({"pattern": "PrefixLM+Softcapping", "flex_ms": t3, "flex_mem": mem3, "sdpa": ""})

    # 4. Full combo: Document + SW + ALiBi + Softcapping
    print("  [Document + SW + ALiBi + Softcapping ---- ]")
    def ultimate_mask(b,h,qi,ki):
        return (qi >= ki) & ((qi - ki) <= 256) & (doc_ids[qi] == doc_ids[ki])
    def ultimate_score(s,b,h,qi,ki):
        s = s - slopes[h] * (qi - ki).abs().to(dtype)
        return CAP * torch.tanh(s / CAP)
    bm4 = create_block_mask(ultimate_mask, B, 1, S, S, device=device)
    _ = flex_attention(q, k, v, score_mod=ultimate_score, block_mask=bm4); torch.cuda.synchronize()
    t4 = bench(lambda: flex_attention(q, k, v, score_mod=ultimate_score, block_mask=bm4))
    mem4 = peak_mem(lambda: flex_attention(q, k, v, score_mod=ultimate_score, block_mask=bm4))
    print(f"    Flex: {t4:.3f} ms, {mem4:.3f} GB")
    print(f"    SDPA: ")
    combos.append({"pattern": "Doc+SW+ALiBi+Softcap", "flex_ms": t4, "flex_mem": mem4, "sdpa": ""})

    # ===  C2 ===
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [c["pattern"] for c in combos]
    times = [c["flex_ms"] for c in combos]
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
    bars = ax.barh(names, times, color=colors, edgecolor='black')
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{t:.1f} ms', va='center', fontsize=10)
    ax.set_xlabel('Latency (ms)')
    ax.set_title('C2: SDPA Cannot - FlexAttention Can')
    ax.axvline(x=0, color='gray', linewidth=0.5)
    #  "SDPA: N/A" 
    for i, name in enumerate(names):
        ax.text(times[i] * 0.5, i, 'SDPA: N/A', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'C2_impossible_for_sdpa.png')
    plt.close()
    print("  [] C2_impossible_for_sdpa.png ")

    DATA["C2"] = combos

    del q, k, v; clear()

# ================================================================
# C3: 
# ================================================================
def exp_C3_scalability():
    print("\n" + "="*70)
    print("  C3: Scalability - Latency & Memory vs Seq Length")
    print("="*70)
    H, D, B = 8, 64, 1
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]
    vanilla_t, vanilla_m = {}, {}
    flex_t, flex_m = {}, {}

    for S in seq_lengths:
        print(f"  [S={S}]")
        clear()
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)
        doc_ids = (torch.arange(S, device=device) // max(1, S // 4)).clamp(max=3)

        # Vanilla: Document(4) + Causal
        try:
            def van():
                sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
                cm = torch.ones(S, S, device=device, dtype=torch.bool).tril_()
                dm = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
                sc = sc.masked_fill(~(cm & dm), float('-inf'))
                return torch.matmul(F.softmax(sc.float(), dim=-1).to(dtype), v)
            vanilla_m[S] = peak_mem(van)
            vanilla_t[S] = bench(van, warmup=1, runs=3)
        except: vanilla_t[S] = None; vanilla_m[S] = None; clear()

        # Flex: Document(4) + Causal
        try:
            def mk(b,h,qi,ki): return (qi >= ki) & (doc_ids[qi] == doc_ids[ki])
            bm = create_block_mask(mk, B, 1, S, S, device=device)
            _ = flex_attention(q, k, v, block_mask=bm); torch.cuda.synchronize()
            flex_m[S] = peak_mem(lambda: flex_attention(q, k, v, block_mask=bm))
            flex_t[S] = bench(lambda: flex_attention(q, k, v, block_mask=bm), warmup=2, runs=5)
        except: flex_t[S] = None; flex_m[S] = None; clear()

        print(f"    Vanilla: {vanilla_t.get(S, 'OOM'):>8} ms | {vanilla_m.get(S, 'OOM'):>8} GB")
        print(f"    Flex:    {flex_t.get(S, 'OOM'):>8} ms | {flex_m.get(S, 'OOM'):>8} GB")
        del q, k, v; clear()

    # ===  C3:  ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    seqs = [s for s in seq_lengths if vanilla_t.get(s) or flex_t.get(s)]

    # 
    ax1.plot(seqs, [vanilla_t.get(s) for s in seqs], 'o-', color='#e74c3c', linewidth=2, label='Vanilla (manual)')
    ax1.plot(seqs, [flex_t.get(s) for s in seqs], '^-', color='#3498db', linewidth=2, label='FlexAttention')
    ax1.set_xlabel('Sequence Length S')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('C3-1: Doc(4)+Causal Latency Scalability')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_yscale('log')

    # 
    ax2.plot(seqs, [vanilla_m.get(s) for s in seqs], 'o-', color='#e74c3c', linewidth=2, label='Vanilla (manual)')
    ax2.plot(seqs, [flex_m.get(s) for s in seqs], '^-', color='#3498db', linewidth=2, label='FlexAttention')
    ax2.axhline(y=22.0, color='red', linestyle='--', linewidth=2, label='L4 VRAM Limit')
    ax2.set_xlabel('Sequence Length S')
    ax2.set_ylabel('Peak Memory (GB)')
    ax2.set_title('C3-2: Doc(4)+Causal Memory Scalability')
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'C3_scalability.png')
    plt.close()
    print("  [] C3_scalability.png ")

    DATA["C3"] = {"vanilla": {"time": vanilla_t, "mem": vanilla_m},
                  "flex": {"time": flex_t, "mem": flex_m}}

# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  FlexAttention ")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  : {OUT_DIR}")

    experiments = [
        ("A1_memory_explosion", exp_A1_memory_explosion),
        ("A2_bandwidth_starvation", exp_A2_bandwidth_starvation),
        ("A3_engineering_nightmare", exp_A3_engineering_nightmare),
        ("B1_score_mod_fusion", exp_B1_score_mod_fusion),
        ("B2_block_mask_visualization", exp_B2_block_mask_visualization),
        ("C1_code_comparison", exp_C1_code_comparison),
        ("C2_impossible_for_sdpa", exp_C2_impossible_for_sdpa),
        ("C3_scalability", exp_C3_scalability),
    ]

    for name, fn in experiments:
        try:
            fn()
        except Exception as e:
            print(f"\n  {name} FAILED: {e}")
            import traceback; traceback.print_exc()
            DATA[name + "_error"] = str(e)

    # Save data
    out_json = Path(__file__).parent / "deep_dive_results.json"
    with open(out_json, "w") as f:
        json.dump(DATA, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"   {out_json}")
    print(f"   {OUT_DIR}/")
    print(f"{'='*70}")
