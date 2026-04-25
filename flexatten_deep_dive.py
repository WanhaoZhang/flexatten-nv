#!/usr/bin/env python3
"""
FlexAttention 深度原理实验与可视化
===================================
板块 A: 没有 FlexAttention 的世界 —— 三大痛点
板块 B: FlexAttention 原理拆解
板块 C: FlexAttention 带来的改变
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
plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans']
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
# A1: 显存爆炸实验 —— S² 中间矩阵的灾难
# ================================================================
def exp_A1_memory_explosion():
    print("\n" + "="*70)
    print("  A1: 显存爆炸 —— Standard Attention 的 O(S²) 灾难")
    print("="*70)
    H, D, B = 8, 64, 1

    # 理论计算 S×S 矩阵显存
    theory = {}
    for S in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        # 每个中间矩阵: S*S*2bytes(fp16), 共5个(scores, causal_mask, combined, softmax, output)
        bytes_per_matrix = S * S * 2
        total_bytes = bytes_per_matrix * 5  # scores + masks + softmax + etc
        theory[S] = round(total_bytes / 1024**3, 3)

    # 实测 Standard / SDPA / Flex 峰值显存
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

    # === 画图 A1: 显存对比 ===
    fig, ax = plt.subplots(figsize=(10, 6))
    seqs = [512, 1024, 2048, 4096, 8192]
    x = np.arange(len(seqs))
    w = 0.25

    vals_std = [actual_std.get(s) for s in seqs]
    vals_sdpa = [actual_sdpa.get(s) for s in seqs]
    vals_flex = [actual_flex.get(s) for s in seqs]
    vals_theory = [theory[s] for s in seqs]

    ax.bar(x - 1.5*w, vals_theory, w, label='Theory (5×S² mat, fp16)', color='#ff6b6b', alpha=0.6, hatch='//')
    ax.bar(x - 0.5*w, [v if v else 0 for v in vals_std], w, label='Standard Attention (实测)', color='#e74c3c')
    ax.bar(x + 0.5*w, [v if v else 0 for v in vals_sdpa], w, label='SDPA / FlashAttention2', color='#2ecc71')
    ax.bar(x + 1.5*w, [v if v else 0 for v in vals_flex], w, label='FlexAttention', color='#3498db')

    ax.axhline(y=22.0, color='red', linestyle='--', linewidth=2, label='L4 VRAM 上限 (22 GB)')

    ax.set_xlabel('序列长度 S')
    ax.set_ylabel('峰值显存 (GB)')
    ax.set_title('A1: 显存爆炸 —— Standard Attention 的 O(S²) 灾难 vs SDPA/Flex')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seqs])
    ax.legend(loc='upper left')
    ax.set_ylim(0, 12)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'A1_memory_explosion.png')
    plt.close()
    print("  [图] A1_memory_explosion.png 已生成")

    DATA["A1"] = {"theory": theory, "std": actual_std, "sdpa": actual_sdpa, "flex": actual_flex}

# ================================================================
# A2: 带宽饥饿实验 —— HBM 往返次数对比
# ================================================================
def exp_A2_bandwidth_starvation():
    print("\n" + "="*70)
    print("  A2: 带宽饥饿 —— Standard 的多次 HBM 往返 vs SDPA 的单次闭环")
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

        # Standard: QK^T → mask → softmax → ×V (4次HBM写入)
        try:
            def std_fn():
                sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
                mask = torch.ones(S, S, device=device, dtype=torch.bool).tril_()
                sc = sc.masked_fill(~mask, float('-inf'))
                w = F.softmax(sc.float(), dim=-1).to(dtype)
                return torch.matmul(w, v)
            std_times[S] = bench(std_fn, warmup=2, runs=5)
        except: std_times[S] = None

        # SDPA: 一次 Fused kernel
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

    # === 画图 A2: 速度对比 ===
    fig, ax = plt.subplots(figsize=(10, 6))
    seqs = sorted(std_times.keys())
    ax.plot(seqs, [std_times[s] for s in seqs], 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Standard Attention (多kernel串联)')
    ax.plot(seqs, [sdpa_times[s] for s in seqs], 's-', color='#2ecc71', linewidth=2, markersize=8, label='SDPA / FlashAttention2 (单Fused kernel)')
    ax.plot(seqs, [flex_times[s] for s in seqs], '^-', color='#3498db', linewidth=2, markersize=8, label='FlexAttention (Triton Fused kernel)')

    ax.set_xlabel('序列长度 S')
    ax.set_ylabel('延迟 (ms)')
    ax.set_title('A2: 带宽饥饿 —— 多次 HBM 往返 vs Fused Kernel 单次闭环')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'A2_bandwidth_starvation.png')
    plt.close()
    print("  [图] A2_bandwidth_starvation.png 已生成")

    DATA["A2"] = {"std": std_times, "sdpa": sdpa_times, "flex": flex_times}

# ================================================================
# A3: 工程噩梦 —— 组合注意力模式的代码复杂度
# ================================================================
def exp_A3_engineering_nightmare():
    print("\n" + "="*70)
    print("  A3: 工程噩梦 —— 实现 Document + SlidingWindow + ALiBi 组合")
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

    # === Vanilla 实现 ===
    print("  [Vanilla 实现] 7步手动组合...")
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
    print(f"    耗时: {time_vanilla:.3f} ms | 显存: {mem_vanilla:.3f} GB")

    # === Flex 实现 ===
    print("  [Flex 实现] 3步声明式...")
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    # mask_mod: 返回 bool
    def combined_mask(b, h, q_idx, kv_idx):
        causal_ok = q_idx >= kv_idx
        sw_ok = (q_idx - kv_idx) <= WINDOW
        doc_ok = doc_ids[q_idx] == doc_ids[kv_idx]
        return causal_ok & sw_ok & doc_ok

    # score_mod: 修改分数
    def alibi_mod(score, b, h, q_idx, kv_idx):
        return score - alibi_slopes[h] * (q_idx - kv_idx).abs().to(dtype)

    bm = create_block_mask(combined_mask, B, 1, S, S, device=device)
    _ = flex_attention(q, k, v, score_mod=alibi_mod, block_mask=bm)
    torch.cuda.synchronize()

    mem_flex = peak_mem(lambda: flex_attention(q, k, v, score_mod=alibi_mod, block_mask=bm))
    time_flex = bench(lambda: flex_attention(q, k, v, score_mod=alibi_mod, block_mask=bm), warmup=2, runs=5)
    print(f"    耗时: {time_flex:.3f} ms | 显存: {mem_flex:.3f} GB")

    # 数值验证
    out_v = vanilla_combined()
    out_f = flex_attention(q, k, v, score_mod=alibi_mod, block_mask=bm)
    diff = round((out_v.float() - out_f.float()).abs().max().item(), 6)
    print(f"    数值误差: {diff}")

    # === 画图 A3: 代码复杂度 vs 性能对比 ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：代码步骤数
    categories = ['Vanilla\n(手动组合)', 'Flex\n(声明式)']
    steps = [7, 3]
    colors = ['#e74c3c', '#3498db']
    bars = ax1.bar(categories, steps, color=colors, width=0.5, edgecolor='black')
    ax1.set_ylabel('实现步骤数')
    ax1.set_title('A3-1: 代码复杂度 —— Document+SlidingWindow+ALiBi 组合')
    for bar, s in zip(bars, steps):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1, str(s), ha='center', fontsize=16, fontweight='bold')
    ax1.set_ylim(0, 9)
    ax1.grid(axis='y', alpha=0.3)

    # 右图：性能对比
    metrics = ['延迟 (ms)', '显存 (GB)']
    vanilla_vals = [time_vanilla, mem_vanilla * 1000]  # scale for visibility
    flex_vals = [time_flex, mem_flex * 1000]
    x = np.arange(len(metrics))
    w = 0.3
    ax2_twin = ax2.twinx()
    b1 = ax2.bar(x - w/2, [time_vanilla, mem_vanilla], w, label='Vanilla', color='#e74c3c')
    b2 = ax2.bar(x + w/2, [time_flex, mem_flex], w, label='Flex', color='#3498db')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_title('A3-2: 性能对比 —— 组合注意力模式')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    for bar in b1:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f'{bar.get_height():.1f}', ha='center', fontsize=10)
    for bar in b2:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f'{bar.get_height():.1f}', ha='center', fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'A3_engineering_nightmare.png')
    plt.close()
    print("  [图] A3_engineering_nightmare.png 已生成")

    DATA["A3"] = {"vanilla": {"time": time_vanilla, "mem": mem_vanilla},
                  "flex": {"time": time_flex, "mem": mem_flex}, "diff": diff}

    del q, k, v; clear()

# ================================================================
# B1: score_mod 算子融合原理
# ================================================================
def exp_B1_score_mod_fusion():
    print("\n" + "="*70)
    print("  B1: score_mod 算子融合 —— 写后改 vs 融合计算")
    print("="*70)
    H, D, B = 8, 64, 1
    S = 2048
    alibi_slopes = 1.0 / (2.0 ** (8 + torch.arange(H, device=device, dtype=dtype)))

    clear()
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    # 方式1: 写后改（Standard）—— 先算 QK^T 写HBM, 再加 bias 再写HBM
    def write_then_modify():
        sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)  # 写HBM
        pos = torch.arange(S, device=device, dtype=dtype)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        for h in range(H):
            sc[:, h] -= alibi_slopes[h] * dist  # 读HBM → 加bias → 写HBM
        cm = torch.ones(S, S, device=device, dtype=torch.bool).tril_()
        sc = sc.masked_fill(~cm, float('-inf'))
        w = F.softmax(sc.float(), dim=-1).to(dtype)
        return torch.matmul(w, v)

    # 方式2: 融合计算（Flex）—— score_mod 在 SRAM 寄存器中一步完成
    def alibi_mod(score, b, h, q_idx, kv_idx):
        return score - alibi_slopes[h] * (q_idx - kv_idx).abs().to(dtype)
    def causal_mod(b, h, q_idx, kv_idx): return q_idx >= kv_idx
    bm = create_block_mask(causal_mod, B, 1, S, S, device=device)
    _ = flex_attention(q, k, v, score_mod=alibi_mod, block_mask=bm)
    torch.cuda.synchronize()

    mem_std = peak_mem(write_then_modify)
    time_std = bench(write_then_modify, warmup=1, runs=5)
    print(f"  写后改 (Standard): {time_std:.3f} ms, {mem_std:.3f} GB")

    mem_flex = peak_mem(lambda: flex_attention(q, k, v, score_mod=alibi_mod, block_mask=bm))
    time_flex = bench(lambda: flex_attention(q, k, v, score_mod=alibi_mod, block_mask=bm), warmup=2, runs=5)
    print(f"  融合计算 (Flex):   {time_flex:.3f} ms, {mem_flex:.3f} GB")

    # 验证数值
    o1 = write_then_modify()
    o2 = flex_attention(q, k, v, score_mod=alibi_mod, block_mask=bm)
    diff = round((o1.float() - o2.float()).abs().max().item(), 6)
    print(f"  数值误差: {diff}")

    # === 画图 B1: 算子融合对比 ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：HBM 访存流程示意
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_title('B1-1: HBM 访存路径对比')

    # Standard flow
    ax1.text(0.5, 9.5, 'Standard (写后改)', fontsize=11, fontweight='bold', color='#e74c3c')
    steps_std = ['QK^T → HBM', '读HBM +Bias', '写回 HBM', 'Mask → HBM', 'Softmax → HBM', '×V → HBM']
    for i, s in enumerate(steps_std):
        y = 8.5 - i * 1.2
        ax1.add_patch(Rectangle((0.2, y-0.3), 4.0, 0.6, facecolor='#ffe0e0', edgecolor='#e74c3c'))
        ax1.text(2.2, y, s, ha='center', va='center', fontsize=8)
        if i < len(steps_std)-1:
            ax1.annotate('', xy=(2.2, y-0.5), xytext=(2.2, y-0.3), arrowprops=dict(arrowstyle='->', color='#e74c3c'))

    # Flex flow
    ax1.text(5.5, 9.5, 'Flex (融合计算)', fontsize=11, fontweight='bold', color='#3498db')
    ax1.add_patch(Rectangle((5.3, 6.7), 4.3, 2.3, facecolor='#e0f0ff', edgecolor='#3498db', linewidth=2))
    ax1.text(7.45, 8.5, '单 Fused Kernel', ha='center', va='center', fontsize=10, fontweight='bold', color='#3498db')
    ax1.text(7.45, 7.8, 'QK^T + Bias + Mask', ha='center', va='center', fontsize=9)
    ax1.text(7.45, 7.2, '+ Softmax + ×V', ha='center', va='center', fontsize=9)
    ax1.annotate('', xy=(7.45, 4.5), xytext=(7.45, 6.7), arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))
    ax1.add_patch(Rectangle((5.8, 3.9), 3.3, 0.6, facecolor='#e0ffe0', edgecolor='#2ecc71'))
    ax1.text(7.45, 4.2, 'Output → HBM', ha='center', va='center', fontsize=9)

    ax1.text(2.2, 1.5, '6次 HBM 写入', ha='center', fontsize=11, color='#e74c3c', fontweight='bold')
    ax1.text(7.45, 3.0, '1次 HBM 写入', ha='center', fontsize=11, color='#2ecc71', fontweight='bold')

    ax1.axis('off')

    # 右图：实测数据
    labels = ['延迟 (ms)', '显存 (GB)']
    std_vals = [time_std, mem_std]
    flex_vals = [time_flex, mem_flex]
    x = np.arange(len(labels))
    w = 0.3
    ax2.bar(x - w/2, std_vals, w, label='写后改 (Standard)', color='#e74c3c')
    ax2.bar(x + w/2, flex_vals, w, label='融合计算 (Flex)', color='#3498db')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title('B1-2: score_mod 融合 vs 写后改 (ALiBi, S=2048)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    for i, (sv, fv) in enumerate(zip(std_vals, flex_vals)):
        ax2.text(i - w/2, sv + 0.1, f'{sv:.1f}', ha='center', fontsize=9)
        ax2.text(i + w/2, fv + 0.1, f'{fv:.1f}', ha='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'B1_score_mod_fusion.png')
    plt.close()
    print("  [图] B1_score_mod_fusion.png 已生成")

    DATA["B1"] = {"std": {"time": time_std, "mem": mem_std},
                  "flex": {"time": time_flex, "mem": mem_flex}, "diff": diff}

    del q, k, v; clear()

# ================================================================
# B2: BlockMask 块级稀疏原理 —— 可视化
# ================================================================
def exp_B2_block_mask_visualization():
    print("\n" + "="*70)
    print("  B2: BlockMask 块级稀疏原理可视化")
    print("="*70)
    S = 512  # 用小 S 画清晰的图
    BS = 128  # BlockSize
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    # 生成不同 mask 的 boolean 矩阵
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

    # === 画图 B2: 4 种 mask 的像素级 + 块级对比 ===
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('B2: BlockMask 块级稀疏原理 —— 像素级 Mask vs 块级 BlockMask (128×128)', fontsize=14)

    for col, (name, mask) in enumerate(masks_data.items()):
        # 上行：像素级 mask
        ax = axes[0, col]
        ax.imshow(mask, cmap='RdYlGn', interpolation='nearest', aspect='equal')
        ax.set_title(f'{name}\n(像素级)', fontsize=10)
        ax.set_xlabel('KV 位置')
        ax.set_ylabel('Q 位置')

        # 下行：块级 mask
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

        # 画块边界
        for i in range(num_blocks + 1):
            ax2.axhline(i - 0.5, color='gray', linewidth=0.5)
            ax2.axvline(i - 0.5, color='gray', linewidth=0.5)

        sparsity = 1.0 - mask.sum() / (S * S)
        ax2.set_title(f'块级 (稀疏率: {sparsity:.1%})\n绿=全计算 红=全跳过 黄=部分', fontsize=9)
        ax2.set_xlabel('KV Block')
        ax2.set_ylabel('Q Block')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'B2_block_mask_visualization.png')
    print("  [图] B2_block_mask_visualization.png 已生成")

    # 统计各 mask 的稀疏率
    sparsity_data = {}
    for name, mask in masks_data.items():
        total = S * S
        active = mask.sum()
        sparsity_data[name] = round(1.0 - active / total, 4)
        print(f"    {name}: 稀疏率 = {sparsity_data[name]:.1%}")

    DATA["B2"] = {"sparsity": sparsity_data}

# ================================================================
# C1: 代码对比 —— Vanilla vs Flex 完整示例
# ================================================================
def exp_C1_code_comparison():
    print("\n" + "="*70)
    print("  C1: 代码对比 —— Vanilla vs Flex (5 种注意力模式)")
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
            "vanilla_steps": "matmul → tril_mask → masked_fill → softmax → matmul",
            "flex_lines": "def mask(b,h,q,k): return q >= k",
            "vanilla_complexity": "中",
            "flex_complexity": "低",
        },
        "SlidingWindow(256)": {
            "vanilla_steps": "matmul → abs_dist → window_mask → masked_fill → softmax → matmul",
            "flex_lines": "def mask(b,h,q,k): return (q >= k) & (q - k <= 256)",
            "vanilla_complexity": "中",
            "flex_complexity": "低",
        },
        "Document(4)+Causal": {
            "vanilla_steps": "matmul → tril → doc_eq → combine → masked_fill → softmax → matmul",
            "flex_lines": "def mask(b,h,q,k): return (q>=k) & (doc[q]==doc[k])",
            "vanilla_complexity": "高",
            "flex_complexity": "低",
        },
        "PrefixLM(25%)": {
            "vanilla_steps": "matmul → prefix_mask → causal → combine → masked_fill → softmax → matmul",
            "flex_lines": "def mask(b,h,q,k): return (k<512) | (q>=k)",
            "vanilla_complexity": "高",
            "flex_complexity": "低",
        },
        "ALiBi+Causal": {
            "vanilla_steps": "matmul → build_dist_matrix → loop over H → add_bias → tril → masked_fill → softmax → matmul",
            "flex_lines": "def mod(s,b,h,q,k): return s - slope[h]*|q-k|",
            "vanilla_complexity": "极高",
            "flex_complexity": "低",
        },
    }

    # 运行实测
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
            print(f"    Vanilla: {t_v:.3f} ms | Flex: {t_f:.3f} ms | 比率: {t_v/t_f:.2f}x")
        except Exception as e:
            print(f"    Error: {e}")
            results.append({"name": name, "error": str(e)})

    # === 画图 C1 ===
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r.get("name", k) for k, r in zip(patterns.keys(), results)]
    if not names:
        names = list(patterns.keys())
    v_times = [r.get("vanilla_ms", 0) for r in results]
    f_times = [r.get("flex_ms", 0) for r in results]

    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, v_times, w, label='Vanilla (手动实现)', color='#e74c3c')
    ax.bar(x + w/2, f_times, w, label='FlexAttention', color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('延迟 (ms)')
    ax.set_title('C1: 5 种注意力模式 —— Vanilla 手动实现 vs FlexAttention')
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
    print("  [图] C1_code_comparison.png 已生成")

    DATA["C1"] = results

    del q, k, v; clear()

# ================================================================
# C2: SDPA 做不了 vs Flex 轻松实现 —— 组合模式
# ================================================================
def exp_C2_impossible_for_sdpa():
    print("\n" + "="*70)
    print("  C2: SDPA 做不了的事 —— Flex 轻松实现")
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
    print(f"    SDPA: 无法实现 (不支持 Document+SW 组合)")
    combos.append({"pattern": "Document+SlidingWindow", "flex_ms": t, "flex_mem": mem, "sparsity": sparse, "sdpa": "不支持"})

    # 2. Document + ALiBi
    print("  [Document + ALiBi]")
    def dm(b,h,qi,ki): return (qi >= ki) & (doc_ids[qi] == doc_ids[ki])
    def am(s,b,h,qi,ki): return s - slopes[h] * (qi - ki).abs().to(dtype)
    bm2 = create_block_mask(dm, B, 1, S, S, device=device)
    _ = flex_attention(q, k, v, score_mod=am, block_mask=bm2); torch.cuda.synchronize()
    t2 = bench(lambda: flex_attention(q, k, v, score_mod=am, block_mask=bm2))
    mem2 = peak_mem(lambda: flex_attention(q, k, v, score_mod=am, block_mask=bm2))
    print(f"    Flex: {t2:.3f} ms, {mem2:.3f} GB")
    print(f"    SDPA: 无法实现 (不支持 score_mod + 自定义 mask)")
    combos.append({"pattern": "Document+ALiBi", "flex_ms": t2, "flex_mem": mem2, "sdpa": "不支持"})

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
    print(f"    SDPA: 无法实现 (不支持 Softcapping + PrefixLM)")
    combos.append({"pattern": "PrefixLM+Softcapping", "flex_ms": t3, "flex_mem": mem3, "sdpa": "不支持"})

    # 4. Full combo: Document + SW + ALiBi + Softcapping
    print("  [Document + SW + ALiBi + Softcapping —— 终极组合]")
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
    print(f"    SDPA: 完全无法实现")
    combos.append({"pattern": "Doc+SW+ALiBi+Softcap (终极)", "flex_ms": t4, "flex_mem": mem4, "sdpa": "不支持"})

    # === 画图 C2 ===
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [c["pattern"] for c in combos]
    times = [c["flex_ms"] for c in combos]
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
    bars = ax.barh(names, times, color=colors, edgecolor='black')
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{t:.1f} ms', va='center', fontsize=10)
    ax.set_xlabel('延迟 (ms)')
    ax.set_title('C2: SDPA 无法实现的注意力模式 —— FlexAttention 轻松搞定')
    ax.axvline(x=0, color='gray', linewidth=0.5)
    # 添加 "SDPA: 不支持" 标签
    for i, name in enumerate(names):
        ax.text(times[i] * 0.5, i, 'SDPA: 不支持', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'C2_impossible_for_sdpa.png')
    plt.close()
    print("  [图] C2_impossible_for_sdpa.png 已生成")

    DATA["C2"] = combos

    del q, k, v; clear()

# ================================================================
# C3: 扩展性全面对比
# ================================================================
def exp_C3_scalability():
    print("\n" + "="*70)
    print("  C3: 扩展性全面对比 —— 不同 S 下的显存/速度")
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

    # === 画图 C3: 双子图 ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    seqs = [s for s in seq_lengths if vanilla_t.get(s) or flex_t.get(s)]

    # 时间
    ax1.plot(seqs, [vanilla_t.get(s) for s in seqs], 'o-', color='#e74c3c', linewidth=2, label='Vanilla (手动)')
    ax1.plot(seqs, [flex_t.get(s) for s in seqs], '^-', color='#3498db', linewidth=2, label='FlexAttention')
    ax1.set_xlabel('序列长度 S')
    ax1.set_ylabel('延迟 (ms)')
    ax1.set_title('C3-1: Document(4)+Causal 延迟扩展性')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_yscale('log')

    # 显存
    ax2.plot(seqs, [vanilla_m.get(s) for s in seqs], 'o-', color='#e74c3c', linewidth=2, label='Vanilla (手动)')
    ax2.plot(seqs, [flex_m.get(s) for s in seqs], '^-', color='#3498db', linewidth=2, label='FlexAttention')
    ax2.axhline(y=22.0, color='red', linestyle='--', linewidth=2, label='L4 VRAM 上限')
    ax2.set_xlabel('序列长度 S')
    ax2.set_ylabel('峰值显存 (GB)')
    ax2.set_title('C3-2: Document(4)+Causal 显存扩展性')
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'C3_scalability.png')
    plt.close()
    print("  [图] C3_scalability.png 已生成")

    DATA["C3"] = {"vanilla": {"time": vanilla_t, "mem": vanilla_m},
                  "flex": {"time": flex_t, "mem": flex_m}}

# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  FlexAttention 深度原理实验与可视化")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  输出目录: {OUT_DIR}")

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
    print(f"  所有结果已保存到 {out_json}")
    print(f"  图表已保存到 {OUT_DIR}/")
    print(f"{'='*70}")
