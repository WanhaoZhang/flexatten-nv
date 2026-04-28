#!/usr/bin/env python3
"""Generate charts for EXL2 vs AWQ experiment."""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("/tmp/flexatten-nv-push/docs/exl2_awq/results/exl2_awq_results.json") as f:
    data = json.load(f)
OUT = "/tmp/flexatten-nv-push/docs/exl2_awq/figures"
DPI = 150


def fig1():
    """FP16 prefill latency vs seq_len."""
    d = data["experiment1_fp16"]
    seq_results = [r for r in d if "seq_len" in r]
    decode_result = [r for r in d if "decode_ms" in r][0]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    seqs = [r["seq_len"] for r in seq_results]
    prefill_ms = [r["prefill_ms"] for r in seq_results]
    tps = [r["tokens_per_s"] for r in seq_results]

    ax1.bar([str(s) for s in seqs], prefill_ms, color="#4e79a7", alpha=0.7,
            edgecolor="black", linewidth=0.5, label="Prefill Latency (ms)")
    ax1.set_ylabel("Prefill Latency (ms)", color="#4e79a7")
    ax1.set_xlabel("Sequence Length")

    ax2 = ax1.twinx()
    ax2.plot([str(s) for s in seqs], tps, "D-", color="#e15759", linewidth=2,
             markersize=7, label="Throughput (tok/s)")
    ax2.set_ylabel("Throughput (tokens/s)", color="#e15759")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title(f"FP16 Baseline: Prefill Latency (decode={decode_result['decode_ms']:.2f}ms)")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig1_fp16_prefill.png", dpi=DPI)
    plt.close(fig)


def fig2():
    """INT4 compression comparison."""
    d = data["experiment2_simulated_int4"]
    r = [x for x in d if "compression_ratio" in x][0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: weight size comparison
    labels = ["FP16 Weights", "INT4 Quantized"]
    sizes = [r["original_weight_bytes"] / 1e6, r["quantized_weight_bytes"] / 1e6]
    colors = ["#4e79a7", "#59a14f"]
    axes[0].bar(labels, sizes, color=colors, edgecolor="black", linewidth=0.5)
    for bar, s in zip(axes[0].patches, sizes):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"{s:.0f}MB", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Size (MB)")
    axes[0].set_title(f"Weight Compression ({r['compression_ratio']:.2f}x)")
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")

    # Right: dequant overhead
    cats = ["FP16 Forward", "Dequant Overhead"]
    vals = [r["fp16_forward_ms"], r["dequant_overhead_ms"]]
    axes[1].bar(cats, vals, color=["#4e79a7", "#e15759"], edgecolor="black", linewidth=0.5)
    for bar, v in zip(axes[1].patches, vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f"{v:.1f}ms", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Time (ms)")
    axes[1].set_title(f"Dequant Overhead ({r['dequant_vs_forward']:.2f}x)")
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(f"{OUT}/fig2_int4_compression.png", dpi=DPI)
    plt.close(fig)


def fig3():
    """vLLM throughput vs batch size."""
    d = data["experiment3_vllm"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bs = [r["batch_size"] for r in d]
    tps = [r["tokens_per_s"] for r in d]

    ax.bar([str(b) for b in bs], tps, color="#4e79a7", edgecolor="black", linewidth=0.5)
    for bar, t in zip(ax.patches, tps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{t:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("vLLM Throughput Scaling (FP16, Qwen2.5-0.5B)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig3_vllm_throughput.png", dpi=DPI)
    plt.close(fig)


def fig4():
    """Bandwidth analysis - theoretical decode performance."""
    d = data["experiment4_bandwidth"]
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    names = [r["method"] for r in d]
    decode_ms = [r["theoretical_decode_ms"] for r in d]
    decode_tps = [r["theoretical_decode_tps"] for r in d]
    speedup = [r["vs_fp16_speedup"] for r in d]

    x = np.arange(len(names))
    width = 0.35

    bars = ax1.bar(x - width / 2, decode_ms, width, color="#e15759", alpha=0.7,
                   edgecolor="black", linewidth=0.5, label="Decode Time (ms)")
    ax1.set_ylabel("Theoretical Decode Time (ms)", color="#e15759")
    ax1.set_xlabel("Quantization Method")

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, decode_tps, width, color="#59a14f", alpha=0.7,
            edgecolor="black", linewidth=0.5, label="Decode TPS")
    ax2.set_ylabel("Theoretical Decode TPS", color="#59a14f")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=8)
    ax1.set_title("Bandwidth Bottleneck: Theoretical Decode Performance (L4 300GB/s)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig4_bandwidth_analysis.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("All EXL2 vs AWQ charts done.")
