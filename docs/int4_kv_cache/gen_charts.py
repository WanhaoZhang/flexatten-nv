#!/usr/bin/env python3
"""Generate charts for INT4 KV Cache experiment."""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("/tmp/flexatten-nv-push/docs/int4_kv_cache/results/int4_kv_cache_results.json") as f:
    data = json.load(f)
OUT = "/tmp/flexatten-nv-push/docs/int4_kv_cache/figures"
DPI = 150


def fig1():
    """FP16 vs FP8 KV cache throughput comparison."""
    fp16 = [r for r in data["experiment1_fp16_kv"] if "batch_size" in r]
    fp8 = [r for r in data["experiment2_fp8_kv"] if "batch_size" in r]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bs = [r["batch_size"] for r in fp16]
    tps_fp16 = [r["tokens_per_s"] for r in fp16]
    tps_fp8 = [r["tokens_per_s"] for r in fp8]

    x = np.arange(len(bs))
    width = 0.35
    bars1 = ax.bar(x - width / 2, tps_fp16, width, color="#4e79a7", edgecolor="black",
                   linewidth=0.5, label="FP16 KV Cache")
    bars2 = ax.bar(x + width / 2, tps_fp8, width, color="#e15759", edgecolor="black",
                   linewidth=0.5, label="FP8 KV Cache")

    for bar, t in zip(bars2, tps_fp8):
        speedup = t / tps_fp16[bars2.index(bar)] if tps_fp16[bars2.index(bar)] > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{t:.0f}", ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bs])
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("FP16 vs FP8 KV Cache: Throughput Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig1_fp16_vs_fp8.png", dpi=DPI)
    plt.close(fig)


def fig2():
    """KV Cache capacity: max tokens and concurrent requests."""
    d = data["experiment3_capacity"]
    capacity = [r for r in d if "max_single_request_tokens" in r]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    names = [r["method"].replace("KV Cache ", "") for r in capacity]
    max_tokens_k = [r["max_single_request_tokens_k"] for r in capacity]
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]

    bars = ax1.bar(names, max_tokens_k, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, max_tokens_k):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{v:.0f}K", ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Max Single Request Tokens (K)")
    ax1.set_title("KV Cache Capacity by Precision (L4 24GB, 16GB KV)")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig2_capacity.png", dpi=DPI)
    plt.close(fig)


def fig3():
    """Concurrent requests at different seq lengths."""
    d = data["experiment3_capacity"]
    concurrent = [r for r in d if "max_concurrent_requests" in r]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    dtypes = ["FP16", "FP8", "INT4"]
    seq_lens = sorted(set(r["seq_len"] for r in concurrent))

    x = np.arange(len(seq_lens))
    width = 0.25
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]

    for i, dtype in enumerate(dtypes):
        vals = []
        for sl in seq_lens:
            match = [r for r in concurrent if r["method"] == f"KV Capacity ({dtype})" and r["seq_len"] == sl]
            vals.append(match[0]["max_concurrent_requests"] if match else 0)
        ax.bar(x + i * width, vals, width, color=colors[i], edgecolor="black",
               linewidth=0.5, label=dtype)

    ax.set_xticks(x + width)
    ax.set_xticklabels([str(sl) for sl in seq_lens])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Max Concurrent Requests")
    ax.set_title("Max Concurrent Requests by KV Cache Precision")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig3_concurrent.png", dpi=DPI)
    plt.close(fig)


def fig4():
    """Quantization error comparison."""
    d = data["experiment4_quality"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names = [r["method"] for r in d]
    mse = [r["avg_mse"] for r in d]
    cos_sim = [r["avg_cosine_similarity"] for r in d]
    sizes = [r["total_kv_mb"] for r in d]

    # Left: MSE
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]
    bars = axes[0].bar(names, mse, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("MSE")
    axes[0].set_title("KV Cache Quantization Error")
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")

    # Right: cosine similarity and size
    ax2 = axes[1]
    bars2 = ax2.bar(names, sizes, color=colors, edgecolor="black", linewidth=0.5, alpha=0.7)
    for bar, s, cs in zip(bars2, sizes, cos_sim):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{s:.0f}MB\ncos={cs:.4f}", ha="center", fontsize=8, fontweight="bold")
    ax2.set_ylabel("Total KV Cache Size (MB, seq=1024)")
    ax2.set_title("KV Cache Size vs Precision")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(f"{OUT}/fig4_quality.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("All INT4 KV Cache charts done.")
