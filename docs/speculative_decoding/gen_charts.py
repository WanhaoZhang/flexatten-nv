#!/usr/bin/env python3
"""Generate charts for Speculative Decoding experiment."""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("/tmp/flexatten-nv-push/docs/speculative_decoding/results/speculative_decoding_results.json") as f:
    data = json.load(f)
OUT = "/tmp/flexatten-nv-push/docs/speculative_decoding/figures"
DPI = 150


def fig1():
    """Autoregressive decode latency vs KV cache size."""
    d = data["experiment1_autoregressive"]
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    pls = [r["prompt_length"] for r in d]
    decode_ms = [r["decode_ms_per_token"] for r in d]
    kv_mb = [r["kv_cache_mb"] for r in d]

    ax1.bar([str(p) for p in pls], decode_ms, color="#4e79a7", alpha=0.7,
            edgecolor="black", linewidth=0.5, label="Decode Latency (ms)")
    ax1.set_ylabel("Decode Latency (ms/tok)", color="#4e79a7")
    ax1.set_xlabel("Prompt Length (KV Cache Size)")

    ax2 = ax1.twinx()
    ax2.plot([str(p) for p in pls], kv_mb, "D-", color="#e15759", linewidth=2,
             markersize=7, label="KV Cache (MB)")
    ax2.set_ylabel("KV Cache Size (MB)", color="#e15759")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("Autoregressive Decode: Latency vs KV Cache Size")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig1_decode_latency.png", dpi=DPI)
    plt.close(fig)


def fig2():
    """Speculative decoding simulation results."""
    d = data["experiment2_speculative_sim"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ks = [r["draft_size_k"] for r in d]
    draft_ms = [r["total_draft_ms"] for r in d]
    verify_ms = [r["verify_ms"] for r in d]
    accept = [r["acceptance_rate"] for r in d]
    speedup = [r["theoretical_speedup"] for r in d]

    # Left: draft + verify time
    x = np.arange(len(ks))
    axes[0].bar(x - 0.15, draft_ms, 0.3, color="#e15759", edgecolor="black",
                linewidth=0.5, label="Draft Time")
    axes[0].bar(x + 0.15, verify_ms, 0.3, color="#59a14f", edgecolor="black",
                linewidth=0.5, label="Verify Time")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(k) for k in ks])
    axes[0].set_xlabel("Draft Size (K)")
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_title("Draft + Verify Latency")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")

    # Right: acceptance rate + speedup
    ax2 = axes[1]
    bars = ax2.bar([str(k) for k in ks], accept, color="#4e79a7", edgecolor="black",
                   linewidth=0.5, alpha=0.7)
    ax2.set_ylabel("Acceptance Rate", color="#4e79a7")
    ax2.set_xlabel("Draft Size (K)")

    ax3 = ax2.twinx()
    ax3.plot([str(k) for k in ks], speedup, "D-", color="#e15759", linewidth=2,
             markersize=7, label="Speedup")
    ax3.set_ylabel("Speedup vs Autoregressive", color="#e15759")
    ax3.legend(loc="upper right")
    ax2.set_title("Acceptance Rate & Speedup")

    plt.tight_layout()
    fig.savefig(f"{OUT}/fig2_speculation_sim.png", dpi=DPI)
    plt.close(fig)


def fig3():
    """Theoretical speedup heatmap."""
    d = data["experiment3_speedup_theory"]

    alphas = sorted(set(r["acceptance_rate"] for r in d))
    gammas = sorted(set(r["draft_size"] for r in d))

    matrix = np.zeros((len(alphas), len(gammas)))
    for r in d:
        i = alphas.index(r["acceptance_rate"])
        j = gammas.index(r["draft_size"])
        matrix[i][j] = r["speedup"]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(gammas)))
    ax.set_xticklabels([str(g) for g in gammas])
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"{a:.2f}" for a in alphas])
    ax.set_xlabel("Draft Size (gamma)")
    ax.set_ylabel("Acceptance Rate (alpha)")
    ax.set_title("Theoretical Speculative Decoding Speedup")

    for i in range(len(alphas)):
        for j in range(len(gammas)):
            ax.text(j, i, f"{matrix[i][j]:.2f}x", ha="center", va="center",
                    fontsize=7, fontweight="bold")

    plt.colorbar(im, label="Speedup")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig3_speedup_heatmap.png", dpi=DPI)
    plt.close(fig)


def fig4():
    """vLLM throughput vs output length."""
    d = data["experiment4_vllm_throughput"]
    baseline = [r for r in d if r["method"] == "vLLM Baseline"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    max_toks = [r["max_tokens"] for r in baseline]
    output_tps = [r["output_tps"] for r in baseline]
    total_tps = [r["total_tps"] for r in baseline]

    ax.bar([str(m) for m in max_toks], output_tps, color="#4e79a7", edgecolor="black",
           linewidth=0.5, alpha=0.7, label="Output TPS")
    ax.plot([str(m) for m in max_toks], total_tps, "D-", color="#e15759", linewidth=2,
            markersize=7, label="Total TPS (prefill+decode)")
    ax.set_xlabel("Max Output Tokens (BS=8)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("vLLM Throughput vs Output Length")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig4_vllm_throughput.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("All Speculative Decoding charts done.")
