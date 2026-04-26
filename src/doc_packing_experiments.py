#!/usr/bin/env python3
"""
Document Packing + Causal Attention: Three Implementations Compared
===================================================================
PyTorch Vanilla | CUDA (via FlashAttention/SDPA proxy) | FlexAttention
"""

import torch
import torch.nn.functional as F
import json
import time
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150

device = "cuda"
dtype = torch.float16
OUT = Path(__file__).parent / "figures_doc_packing"
OUT.mkdir(exist_ok=True)
DATA = {}

def clear():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

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
    return {"mean": round(sum(ts)/len(ts), 3), "min": round(min(ts), 3), "max": round(max(ts), 3)}

def peak_mem(fn):
    clear(); torch.cuda.reset_peak_memory_stats()
    r = fn(); torch.cuda.synchronize()
    m = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
    del r; return m

# ================================================================
# Exp1: Vanilla vs Flex across sequence lengths + doc counts
# ================================================================
def exp1_main_comparison():
    print("=" * 70)
    print("  Exp1: Vanilla PyTorch vs FlexAttention - Full Comparison")
    print("=" * 70)
    H, D, B = 8, 64, 1
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    rows = []
    for S in [256, 512, 1024, 2048, 4096, 8192]:
        for ndocs in [2, 4, 8]:
            if S // ndocs < 64: continue
            print(f"  [S={S}, docs={ndocs}]")
            clear()
            doc_ids = (torch.arange(S, device=device) // (S // ndocs)).clamp(max=ndocs-1)
            q = torch.randn(B, H, S, D, device=device, dtype=dtype)
            k = torch.randn(B, H, S, D, device=device, dtype=dtype)
            v = torch.randn(B, H, S, D, device=device, dtype=dtype)

            # --- Vanilla PyTorch ---
            v_ok = True
            try:
                def vanilla():
                    sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
                    cm = torch.ones(S, S, device=device, dtype=torch.bool).tril_()
                    dm = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
                    sc = sc.masked_fill(~(cm & dm), float('-inf'))
                    w = F.softmax(sc.float(), dim=-1).to(dtype)
                    return torch.matmul(w, v)
                mem_v = peak_mem(vanilla)
                t_v = bench(vanilla, warmup=1, runs=3)
            except RuntimeError:
                v_ok = False; mem_v = None; t_v = None; clear()

            # --- FlexAttention ---
            f_ok = True
            try:
                def mask_mod(b, h, qi, ki):
                    return (qi >= ki) & (doc_ids[qi] == doc_ids[ki])
                bm = create_block_mask(mask_mod, B, 1, S, S, device=device)
                _ = flex_attention(q, k, v, block_mask=bm); torch.cuda.synchronize()

                # sparsity
                total_blks = bm.kv_num_blocks.sum().item()
                max_blks = B * (S // 128) ** 2
                sparsity = round(1.0 - total_blks / max_blks, 4) if max_blks > 0 else 0

                def flex():
                    return flex_attention(q, k, v, block_mask=bm)
                mem_f = peak_mem(flex)
                t_f = bench(flex, warmup=2, runs=5)
            except RuntimeError:
                f_ok = False; mem_f = None; t_f = None; sparsity = None; clear()

            # numerical diff
            diff = None
            if v_ok and f_ok:
                o1 = vanilla(); o2 = flex_attention(q, k, v, block_mask=bm)
                diff = round((o1.float() - o2.float()).abs().max().item(), 6)

            r = {"S": S, "ndocs": ndocs,
                 "vanilla": {"time": t_v, "mem": mem_v, "ok": v_ok},
                 "flex": {"time": t_f, "mem": mem_f, "ok": f_ok, "sparsity": sparsity},
                 "diff": diff}
            rows.append(r)
            v_s = f"{t_v['mean']:.2f}ms/{mem_v:.3f}GB" if v_ok else "OOM"
            f_s = f"{t_f['mean']:.2f}ms/{mem_f:.3f}GB/sp={sparsity:.0%}" if f_ok else "OOM"
            print(f"    Vanilla: {v_s}")
            print(f"    Flex:    {f_s}")
            if diff is not None: print(f"    Diff: {diff}")
            del q, k, v; clear()

    # === Chart 1: Memory comparison ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ndocs in [2, 4, 8]:
        sub = [r for r in rows if r["ndocs"] == ndocs]
        seqs = [r["S"] for r in sub]
        vm = [r["vanilla"]["mem"] if r["vanilla"]["ok"] else None for r in sub]
        fm = [r["flex"]["mem"] if r["flex"]["ok"] else None for r in sub]
        ax1.plot(seqs, vm, 'o--', label=f'Vanilla (docs={ndocs})', alpha=0.7)
        ax1.plot(seqs, fm, 's-', label=f'Flex (docs={ndocs})', alpha=0.7)
    ax1.axhline(y=22, color='red', ls='--', lw=2, label='L4 VRAM Limit')
    ax1.set_xlabel('Sequence Length S'); ax1.set_ylabel('Peak Memory (GB)')
    ax1.set_title('Memory Usage: Vanilla vs FlexAttention')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # Chart 2: Speed comparison
    for ndocs in [2, 4, 8]:
        sub = [r for r in rows if r["ndocs"] == ndocs]
        seqs = [r["S"] for r in sub]
        vt = [r["vanilla"]["time"]["mean"] if r["vanilla"]["ok"] else None for r in sub]
        ft = [r["flex"]["time"]["mean"] if r["flex"]["ok"] else None for r in sub]
        ax2.plot(seqs, vt, 'o--', label=f'Vanilla (docs={ndocs})', alpha=0.7)
        ax2.plot(seqs, ft, 's-', label=f'Flex (docs={ndocs})', alpha=0.7)
    ax2.set_xlabel('Sequence Length S'); ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Latency: Vanilla vs FlexAttention')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3); ax2.set_yscale('log')

    fig.tight_layout(); fig.savefig(OUT / 'exp1_memory_speed.png'); plt.close()
    print("  [Chart] exp1_memory_speed.png")

    DATA["exp1"] = rows

# ================================================================
# Exp2: Step-by-step memory breakdown (Vanilla)
# ================================================================
def exp2_vanilla_memory_breakdown():
    print("\n" + "=" * 70)
    print("  Exp2: Vanilla Memory Breakdown - Where does O(S^2) come from?")
    print("=" * 70)
    S, H, D, B = 2048, 8, 64, 1
    clear()

    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    ndocs = 4
    doc_ids = (torch.arange(S, device=device) // (S // ndocs)).clamp(max=ndocs-1)

    steps = {}
    # Step 1: QK^T
    torch.cuda.reset_peak_memory_stats()
    scores = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
    torch.cuda.synchronize()
    steps["1_QKT"] = round(torch.cuda.max_memory_allocated() / 1024**3, 4)

    # Step 2: Causal mask
    causal = torch.ones(S, S, device=device, dtype=torch.bool).tril_()
    steps["2_causal_mask"] = round(torch.cuda.max_memory_allocated() / 1024**3, 4)

    # Step 3: Doc mask
    doc_mask = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
    steps["3_doc_mask"] = round(torch.cuda.max_memory_allocated() / 1024**3, 4)

    # Step 4: Combine masks
    combined = causal & doc_mask
    steps["4_combined_mask"] = round(torch.cuda.max_memory_allocated() / 1024**3, 4)

    # Step 5: masked_fill
    scores = scores.masked_fill(~combined, float('-inf'))
    steps["5_masked_fill"] = round(torch.cuda.max_memory_allocated() / 1024**3, 4)

    # Step 6: Softmax
    attn_w = F.softmax(scores.float(), dim=-1).to(dtype)
    steps["6_softmax"] = round(torch.cuda.max_memory_allocated() / 1024**3, 4)

    # Step 7: x V
    output = torch.matmul(attn_w, v)
    steps["7_output"] = round(torch.cuda.max_memory_allocated() / 1024**3, 4)

    for step, mem in steps.items():
        print(f"    {step}: {mem:.4f} GB")

    # === Chart 2: Memory waterfall ===
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(steps.keys())
    vals = list(steps.values())
    increments = [vals[0]] + [vals[i] - vals[i-1] for i in range(1, len(vals))]
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c']

    bottom = 0
    for i, (label, inc) in enumerate(zip(labels, increments)):
        ax.bar('Vanilla Pipeline', inc, bottom=bottom, color=colors[i % len(colors)],
               label=f'{label} (+{inc:.4f} GB)', edgecolor='white', linewidth=0.5)
        bottom += inc

    ax.set_ylabel('Cumulative Memory (GB)')
    ax.set_title(f'Exp2: Memory Waterfall - Vanilla Doc Packing (S={S}, docs={ndocs})')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / 'exp2_memory_waterfall.png'); plt.close()
    print("  [Chart] exp2_memory_waterfall.png")

    DATA["exp2"] = steps
    del q, k, v; clear()

# ================================================================
# Exp3: Sparsity visualization for different doc counts
# ================================================================
def exp3_sparsity_visualization():
    print("\n" + "=" * 70)
    print("  Exp3: BlockMask Sparsity Visualization")
    print("=" * 70)
    S = 512
    BS = 128
    pos = torch.arange(S)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    for col, ndocs in enumerate([2, 4, 8, 16]):
        doc_ids = (pos // (S // ndocs)).clamp(max=ndocs-1)

        # Pixel-level mask
        mask = ((pos.unsqueeze(0) >= pos.unsqueeze(1)) &
                (doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1))).numpy()

        ax = axes[0, col]
        ax.imshow(mask, cmap='RdYlGn', interpolation='nearest')
        ax.set_title(f'Doc({ndocs})+Causal\nPixel-level (S={S})', fontsize=10)
        ax.set_xlabel('KV Position'); ax.set_ylabel('Q Position')

        # Block-level mask
        nb = S // BS
        bm = np.zeros((nb, nb))
        for bi in range(nb):
            for bj in range(nb):
                block = mask[bi*BS:(bi+1)*BS, bj*BS:(bj+1)*BS]
                if block.all(): bm[bi, bj] = 1.0
                elif not block.any(): bm[bi, bj] = 0.0
                else: bm[bi, bj] = 0.5

        ax2 = axes[1, col]
        ax2.imshow(bm, cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
        for i in range(nb+1):
            ax2.axhline(i-0.5, color='gray', lw=0.5)
            ax2.axvline(i-0.5, color='gray', lw=0.5)

        sparse = round(1.0 - mask.sum() / (S*S), 3)
        ax2.set_title(f'Block-level (128x128)\nSparsity={sparse:.1%}', fontsize=10)
        ax2.set_xlabel('KV Block'); ax2.set_ylabel('Q Block')
        print(f"    Doc({ndocs}): sparsity={sparse:.1%}")

    fig.suptitle('Exp3: How BlockMask Compresses Doc Packing Masks', fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / 'exp3_sparsity.png'); plt.close()
    print("  [Chart] exp3_sparsity.png")
    DATA["exp3"] = "done"

# ================================================================
# Exp4: Flex vs SDPA (standard causal) - why SDPA still matters
# ================================================================
def exp4_sdpa_baseline():
    print("\n" + "=" * 70)
    print("  Exp4: SDPA Baseline - Standard Causal (no doc packing)")
    print("=" * 70)
    H, D, B = 8, 64, 1
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    rows = []
    for S in [512, 1024, 2048, 4096, 8192]:
        clear()
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)

        # SDPA
        mem_s = peak_mem(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        t_s = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))

        # Flex
        def cm(b,h,qi,ki): return qi >= ki
        bm = create_block_mask(cm, B, 1, S, S, device=device)
        _ = flex_attention(q, k, v, block_mask=bm); torch.cuda.synchronize()
        mem_f = peak_mem(lambda: flex_attention(q, k, v, block_mask=bm))
        t_f = bench(lambda: flex_attention(q, k, v, block_mask=bm))

        print(f"  [S={S}] SDPA: {t_s['mean']:.3f}ms/{mem_s:.3f}GB | Flex: {t_f['mean']:.3f}ms/{mem_f:.3f}GB | Ratio: {t_f['mean']/t_s['mean']:.1f}x")
        rows.append({"S": S, "sdpa": {"time": t_s, "mem": mem_s}, "flex": {"time": t_f, "mem": mem_f},
                      "ratio": round(t_f["mean"]/t_s["mean"], 1)})
        del q, k, v; clear()

    # === Chart 4 ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    seqs = [r["S"] for r in rows]
    ax1.plot(seqs, [r["sdpa"]["time"]["mean"] for r in rows], 's-', color='#2ecc71', lw=2, ms=8, label='SDPA (FlashAttention2)')
    ax1.plot(seqs, [r["flex"]["time"]["mean"] for r in rows], '^-', color='#3498db', lw=2, ms=8, label='FlexAttention')
    ax1.set_xlabel('Sequence Length S'); ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Standard Causal: SDPA vs FlexAttention')
    ax1.legend(); ax1.grid(alpha=0.3); ax1.set_yscale('log')

    ax2.bar(range(len(seqs)), [r["ratio"] for r in rows], color='#e74c3c', edgecolor='black')
    ax2.set_xticks(range(len(seqs))); ax2.set_xticklabels([str(s) for s in seqs])
    ax2.set_xlabel('Sequence Length S'); ax2.set_ylabel('Flex / SDPA Ratio')
    ax2.set_title('How much slower is FlexAttention vs SDPA?')
    for i, r in enumerate(rows):
        ax2.text(i, r["ratio"]+2, f'{r["ratio"]}x', ha='center', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout(); fig.savefig(OUT / 'exp4_sdpa_baseline.png'); plt.close()
    print("  [Chart] exp4_sdpa_baseline.png")
    DATA["exp4"] = rows

# ================================================================
# Exp5: OOM boundary detection
# ================================================================
def exp5_oom_boundary():
    print("\n" + "=" * 70)
    print("  Exp5: OOM Boundary - How far can each method go?")
    print("=" * 70)
    H, D, B = 8, 64, 1
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    rows = []
    for S in [2048, 4096, 8192, 12288, 16384, 20480, 24576, 32768]:
        ndocs = max(2, S // 1024)
        print(f"  [S={S}, docs={ndocs}]")

        # Vanilla
        v_ok = True
        try:
            clear()
            q = torch.randn(B, H, S, D, device=device, dtype=dtype)
            k = torch.randn(B, H, S, D, device=device, dtype=dtype)
            v = torch.randn(B, H, S, D, device=device, dtype=dtype)
            doc_ids = (torch.arange(S, device=device) // (S // ndocs)).clamp(max=ndocs-1)
            torch.cuda.reset_peak_memory_stats()
            sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
            cm = torch.ones(S, S, device=device, dtype=torch.bool).tril_()
            dm = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
            sc.masked_fill_(~(cm & dm), float('-inf'))
            w = F.softmax(sc.float(), dim=-1).to(dtype)
            _ = torch.matmul(w, v)
            torch.cuda.synchronize()
            mem_v = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        except RuntimeError:
            v_ok = False; mem_v = None; clear()

        # Flex
        f_ok = True
        try:
            clear()
            q = torch.randn(B, H, S, D, device=device, dtype=dtype)
            k = torch.randn(B, H, S, D, device=device, dtype=dtype)
            v = torch.randn(B, H, S, D, device=device, dtype=dtype)
            doc_ids = (torch.arange(S, device=device) // (S // ndocs)).clamp(max=ndocs-1)
            def mm(b,h,qi,ki): return (qi >= ki) & (doc_ids[qi] == doc_ids[ki])
            bm = create_block_mask(mm, B, 1, S, S, device=device)
            _ = flex_attention(q, k, v, block_mask=bm); torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            _ = flex_attention(q, k, v, block_mask=bm)
            torch.cuda.synchronize()
            mem_f = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        except RuntimeError:
            f_ok = False; mem_f = None; clear()

        v_s = f"{mem_v:.3f}GB" if v_ok else "OOM"
        f_s = f"{mem_f:.3f}GB" if f_ok else "OOM"
        print(f"    Vanilla: {v_s} | Flex: {f_s}")
        rows.append({"S": S, "ndocs": ndocs, "vanilla_mem": mem_v, "flex_mem": mem_f,
                      "vanilla_ok": v_ok, "flex_ok": f_ok})
        if not v_ok and not f_ok:
            break
        del q, k, v; clear()

    # === Chart 5 ===
    fig, ax = plt.subplots(figsize=(10, 6))
    seqs = [r["S"] for r in rows]
    vm = [r["vanilla_mem"] if r["vanilla_ok"] else 0 for r in rows]
    fm = [r["flex_mem"] if r["flex_ok"] else 0 for r in rows]

    x = np.arange(len(seqs)); w = 0.35
    ax.bar(x - w/2, vm, w, label='Vanilla PyTorch', color='#e74c3c')
    ax.bar(x + w/2, fm, w, label='FlexAttention', color='#3498db')
    ax.axhline(y=22, color='red', ls='--', lw=2, label='L4 VRAM Limit')

    # Mark OOM
    for i, r in enumerate(rows):
        if not r["vanilla_ok"]:
            ax.text(i - w/2, 0.5, 'OOM', ha='center', color='#e74c3c', fontweight='bold')
        if not r["flex_ok"]:
            ax.text(i + w/2, 0.5, 'OOM', ha='center', color='#3498db', fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in seqs])
    ax.set_xlabel('Sequence Length S'); ax.set_ylabel('Peak Memory (GB)')
    ax.set_title('Exp5: OOM Boundary - Doc Packing + Causal')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / 'exp5_oom_boundary.png'); plt.close()
    print("  [Chart] exp5_oom_boundary.png")
    DATA["exp5"] = rows

# ================================================================
# Exp6: Numerical accuracy heatmap
# ================================================================
def exp6_accuracy():
    print("\n" + "=" * 70)
    print("  Exp6: Numerical Accuracy - Vanilla vs Flex")
    print("=" * 70)
    H, D, B = 8, 64, 1
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    rows = []
    for S in [256, 512, 1024, 2048, 4096]:
        for ndocs in [2, 4, 8]:
            if S // ndocs < 64: continue
            clear()
            q = torch.randn(B, H, S, D, device=device, dtype=dtype)
            k = torch.randn(B, H, S, D, device=device, dtype=dtype)
            v = torch.randn(B, H, S, D, device=device, dtype=dtype)
            doc_ids = (torch.arange(S, device=device) // (S // ndocs)).clamp(max=ndocs-1)

            # Vanilla
            sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
            cm = torch.ones(S, S, device=device, dtype=torch.bool).tril_()
            dm = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
            sc = sc.masked_fill(~(cm & dm), float('-inf'))
            w = F.softmax(sc.float(), dim=-1).to(dtype)
            out_v = torch.matmul(w, v)

            # Flex
            def mm(b,h,qi,ki): return (qi >= ki) & (doc_ids[qi] == doc_ids[ki])
            bm = create_block_mask(mm, B, 1, S, S, device=device)
            out_f = flex_attention(q, k, v, block_mask=bm)

            diff = round((out_v.float() - out_f.float()).abs().max().item(), 6)
            mean_diff = round((out_v.float() - out_f.float()).abs().mean().item(), 6)
            rows.append({"S": S, "ndocs": ndocs, "max_diff": diff, "mean_diff": mean_diff})
            print(f"    S={S}, docs={ndocs}: max_diff={diff}, mean_diff={mean_diff}")
            del q, k, v; clear()

    # === Chart 6 ===
    fig, ax = plt.subplots(figsize=(10, 6))
    for ndocs in [2, 4, 8]:
        sub = [r for r in rows if r["ndocs"] == ndocs]
        ax.plot([r["S"] for r in sub], [r["max_diff"] for r in sub],
                'o-', label=f'docs={ndocs} (max diff)')
    ax.axhline(y=0.001, color='green', ls='--', label='FP16 normal range (< 0.001)')
    ax.axhline(y=0.01, color='red', ls='--', label='Error threshold (> 0.01)')
    ax.set_xlabel('Sequence Length S'); ax.set_ylabel('Max Absolute Diff')
    ax.set_title('Exp6: Numerical Accuracy - Vanilla vs FlexAttention')
    ax.legend(); ax.grid(alpha=0.3); ax.set_yscale('log')
    fig.tight_layout(); fig.savefig(OUT / 'exp6_accuracy.png'); plt.close()
    print("  [Chart] exp6_accuracy.png")
    DATA["exp6"] = rows

# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  Document Packing + Causal Attention: Complete Analysis")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")

    for name, fn in [("exp1", exp1_main_comparison),
                     ("exp2", exp2_vanilla_memory_breakdown),
                     ("exp3", exp3_sparsity_visualization),
                     ("exp4", exp4_sdpa_baseline),
                     ("exp5", exp5_oom_boundary),
                     ("exp6", exp6_accuracy)]:
        try:
            fn()
        except Exception as e:
            print(f"\n  {name} FAILED: {e}")
            import traceback; traceback.print_exc()

    out = Path(__file__).parent / "doc_packing_results.json"
    with open(out, "w") as f:
        json.dump(DATA, f, indent=2, default=str)
    print(f"\n  Results: {out}")
    print(f"  Charts: {OUT}/")
    print("  DONE!")
