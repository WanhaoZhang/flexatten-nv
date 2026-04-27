"""
Plotting script for FlexAttention Backward Benchmark results.
Reads backward_benchmark_results.json and generates figures.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {
    'Causal': '#2563EB',
    'SlidingWindow_256': '#16A34A',
    'SlidingWindow_512': '#0D9488',
    'PrefixLM_128': '#EA580C',
    'PrefixLM_256': '#DC2626',
    'FlexAttention': '#2563EB',
    'SDPA_causal': '#16A34A',
    'Vanilla_eager': '#EA580C',
}

def load_results():
    path = os.path.join(RESULTS_DIR, "backward_benchmark_results.json")
    with open(path) as f:
        return json.load(f)

def plot1_fwd_bwd_by_mask(data):
    """Figure 1: Forward vs Backward latency by mask type and seq_len."""
    exp1 = data["experiment1_latency"]
    valid = [r for r in exp1 if r.get("forward_ms") and not r.get("oom")]

    masks = list(dict.fromkeys(r["mask"] for r in valid))
    seq_lens = sorted(set(r["seq_len"] for r in valid))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Forward
    ax = axes[0]
    for mask in masks:
        vals = [(r["seq_len"], r["forward_ms"]) for r in valid if r["mask"] == mask]
        vals.sort()
        if vals:
            xs, ys = zip(*vals)
            ax.plot(xs, ys, 'o-', label=mask, color=COLORS.get(mask, None), linewidth=2, markersize=6)
    ax.set_xlabel('Sequence Length', fontsize=11)
    ax.set_ylabel('Forward Latency (ms)', fontsize=11)
    ax.set_title('Forward Latency by Mask Type', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(seq_lens)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    # Backward
    ax = axes[1]
    for mask in masks:
        vals = [(r["seq_len"], r["backward_ms"]) for r in valid if r["mask"] == mask]
        vals.sort()
        if vals:
            xs, ys = zip(*vals)
            ax.plot(xs, ys, 'o-', label=mask, color=COLORS.get(mask, None), linewidth=2, markersize=6)
    ax.set_xlabel('Sequence Length', fontsize=11)
    ax.set_ylabel('Backward Latency (ms)', fontsize=11)
    ax.set_title('Backward Latency by Mask Type', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(seq_lens)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_fwd_bwd_latency.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Generated: fig1_fwd_bwd_latency.png")

def plot2_peak_memory(data):
    """Figure 2: Peak memory by mask type and seq_len."""
    exp1 = data["experiment1_latency"]
    valid = [r for r in exp1 if r.get("total_peak_mb") and not r.get("oom")]

    masks = list(dict.fromkeys(r["mask"] for r in valid))
    seq_lens = sorted(set(r["seq_len"] for r in valid))

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(seq_lens))
    width = 0.15
    for i, mask in enumerate(masks):
        vals = [next((r["total_peak_mb"] for r in valid if r["mask"] == mask and r["seq_len"] == sl), 0)
                for sl in seq_lens]
        ax.bar(x + i * width, vals, width, label=mask, color=COLORS.get(mask, None), alpha=0.85)

    ax.set_xlabel('Sequence Length', fontsize=11)
    ax.set_ylabel('Peak Memory (MB)', fontsize=11)
    ax.set_title('Training Peak Memory by Mask Type', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * len(masks) / 2)
    ax.set_xticklabels([str(sl) for sl in seq_lens])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_peak_memory.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Generated: fig2_peak_memory.png")

def plot3_fwd_bwd_ratio(data):
    """Figure 3: Forward/Backward ratio (how much slower is backward?)."""
    exp1 = data["experiment1_latency"]
    valid = [r for r in exp1 if r.get("forward_ms") and r.get("backward_ms") and not r.get("oom")]

    masks = list(dict.fromkeys(r["mask"] for r in valid))
    seq_lens = sorted(set(r["seq_len"] for r in valid))

    fig, ax = plt.subplots(figsize=(10, 6))

    for mask in masks:
        vals = [(r["seq_len"], r["backward_ms"] / r["forward_ms"]) for r in valid if r["mask"] == mask]
        vals.sort()
        if vals:
            xs, ys = zip(*vals)
            ax.plot(xs, ys, 'o-', label=mask, color=COLORS.get(mask, None), linewidth=2, markersize=6)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='fwd = bwd')
    ax.set_xlabel('Sequence Length', fontsize=11)
    ax.set_ylabel('Backward / Forward Ratio', fontsize=11)
    ax.set_title('Backward-to-Forward Latency Ratio', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(seq_lens)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_fwd_bwd_ratio.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Generated: fig3_fwd_bwd_ratio.png")

def plot4_compilation_overhead(data):
    """Figure 4: Compilation overhead bar chart."""
    exp2 = data.get("experiment2_compilation", [])
    if not exp2:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    masks = [r["mask"] for r in exp2]
    first_call = [r["first_call_ms"] for r in exp2]
    steady = [r["steady_state_ms"] for r in exp2]
    overhead = [r["compile_overhead_ms"] for r in exp2]

    x = np.arange(len(masks))
    width = 0.3

    bars1 = ax.bar(x - width/2, first_call, width, label='First Call (with compile)', color='#EA580C', alpha=0.85)
    bars2 = ax.bar(x + width/2, steady, width, label='Steady State', color='#2563EB', alpha=0.85)

    # Annotate overhead ratio
    for i, r in enumerate(exp2):
        ax.text(x[i], max(first_call[i], steady[i]) + 5,
                f'{r["overhead_ratio"]}x overhead',
                ha='center', fontsize=9, fontweight='bold', color='#DC2626')

    ax.set_xlabel('Mask Type', fontsize=11)
    ax.set_ylabel('Total Time (ms)', fontsize=11)
    ax.set_title('Compilation Overhead: First Call vs Steady State (fwd+bwd)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(masks)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_compilation_overhead.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Generated: fig4_compilation_overhead.png")

def plot5_doc_packing(data):
    """Figure 5: Doc packing performance vs number of documents."""
    exp3 = data.get("experiment3_doc_packing", [])
    if not exp3:
        return

    valid = [r for r in exp3 if r.get("forward_ms") and not r.get("oom")]
    seq_lens = sorted(set(r["seq_len"] for r in valid))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for seq_len in seq_lens:
        vals = [(r["n_docs"], r["total_ms"], r["theoretical_sparsity"])
                for r in valid if r["seq_len"] == seq_len]
        if not vals:
            continue
        vals.sort()
        docs, totals, sparsities = zip(*vals)

        axes[0].plot(docs, totals, 'o-', label=f'seq={seq_len}', linewidth=2, markersize=6)
        axes[1].plot(docs, [s * 100 for s in sparsities], 'o-', label=f'seq={seq_len}', linewidth=2, markersize=6)

    axes[0].set_xlabel('Number of Documents', fontsize=11)
    axes[0].set_ylabel('Total Time (fwd+bwd) (ms)', fontsize=11)
    axes[0].set_title('Doc Packing: Latency vs Document Count', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Number of Documents', fontsize=11)
    axes[1].set_ylabel('Theoretical Sparsity (%)', fontsize=11)
    axes[1].set_title('Doc Packing: Sparsity vs Document Count', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig5_doc_packing.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Generated: fig5_doc_packing.png")

def plot6_backend_comparison(data):
    """Figure 6: FlexAttention vs SDPA vs Vanilla comparison."""
    exp4 = data.get("experiment4_backend_comparison", [])
    if not exp4:
        return

    valid = [r for r in exp4 if r.get("forward_ms") and not r.get("oom")]
    backends = list(dict.fromkeys(r["backend"] for r in valid))
    seq_lens = sorted(set(r["seq_len"] for r in valid))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, (metric, title) in enumerate([
        ("forward_ms", "Forward Latency"),
        ("backward_ms", "Backward Latency"),
        ("total_ms", "Total (fwd+bwd) Latency"),
    ]):
        ax = axes[ax_idx]
        for backend in backends:
            vals = [(r["seq_len"], r[metric]) for r in valid if r["backend"] == backend]
            vals.sort()
            if vals:
                xs, ys = zip(*vals)
                ax.plot(xs, ys, 'o-', label=backend, color=COLORS.get(backend, None),
                        linewidth=2, markersize=6)
        ax.set_xlabel('Sequence Length', fontsize=11)
        ax.set_ylabel(f'{title} (ms)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_xticks(seq_lens)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig6_backend_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Generated: fig6_backend_comparison.png")

if __name__ == '__main__':
    data = load_results()
    plot1_fwd_bwd_by_mask(data)
    plot2_peak_memory(data)
    plot3_fwd_bwd_ratio(data)
    plot4_compilation_overhead(data)
    plot5_doc_packing(data)
    plot6_backend_comparison(data)
    print(f"\nAll figures saved to {FIGURES_DIR}")
