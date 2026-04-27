#!/usr/bin/env python3
"""Generate charts for Liger Kernel experiment results."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open("/tmp/flexatten-nv-push/docs/liger_kernel/results/liger_kernel_results.json") as f:
    data = json.load(f)

OUTPUT_DIR = "/tmp/flexatten-nv-push/docs/liger_kernel/figures"
DPI = 150


def fig1_logits_explosion():
    exp1 = data["experiment1_logits_memory"]
    seq_lens = [d["seq_len"] for d in exp1]
    logits_mb = [d["logits_tensor_mb"] for d in exp1]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [str(s) for s in seq_lens],
        logits_mb,
        color=["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#b07aa1", "#ff6f61"],
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels on top of bars
    for bar, val in zip(bars, logits_mb):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Logits Tensor Memory (MB)", fontsize=12)
    ax.set_title("Logits Tensor Memory Explosion [B,S,V]", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(logits_mb) * 1.15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig1_logits_explosion.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig1_logits_explosion.png")


def fig2_chunked_savings():
    exp2 = data["experiment2_fusion_simulation"]

    # Collect unique seq_lens (up to 4096) and chunk_sizes
    seq_lens = sorted(set(d["seq_len"] for d in exp2 if d["seq_len"] <= 4096))
    chunk_sizes = sorted(set(d["chunk_size"] for d in exp2))

    # Get unfused values per seq_len (they are the same across chunk sizes)
    unfused_by_seq = {}
    for d in exp2:
        if d["seq_len"] not in unfused_by_seq:
            unfused_by_seq[d["seq_len"]] = d["peak_unfused_mb"]

    # Build lines: unfused + each chunk_size
    chunked_by_cs = {cs: [] for cs in chunk_sizes}
    for cs in chunk_sizes:
        for sl in seq_lens:
            for d in exp2:
                if d["seq_len"] == sl and d["chunk_size"] == cs:
                    chunked_by_cs[cs].append(d["peak_chunked_mb"])
                    break

    unfused_vals = [unfused_by_seq[sl] for sl in seq_lens]
    x = np.arange(len(seq_lens))
    x_labels = [str(s) for s in seq_lens]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    colors = {
        "unfused": "#e15759",
        1: "#4e79a7",
        4: "#59a14f",
        16: "#f28e2b",
        64: "#b07aa1",
    }

    ax.plot(x, unfused_vals, marker="o", linewidth=2, label="Unfused", color=colors["unfused"])
    for cs in chunk_sizes:
        ax.plot(
            x,
            chunked_by_cs[cs],
            marker="s",
            linewidth=2,
            label=f"chunk_{cs}",
            color=colors[cs],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Peak Memory (MB)", fontsize=12)
    ax.set_title("Chunked CE Memory Savings", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig2_chunked_savings.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig2_chunked_savings.png")


def fig3_saving_pct():
    exp2 = data["experiment2_fusion_simulation"]
    seq_lens = sorted(set(d["seq_len"] for d in exp2))
    chunk_sizes = sorted(set(d["chunk_size"] for d in exp2))

    # Build a matrix: rows=seq_lens, cols=chunk_sizes
    saving_matrix = {}
    for d in exp2:
        key = (d["seq_len"], d["chunk_size"])
        saving_matrix[key] = d["saving_pct"]

    x = np.arange(len(seq_lens))
    width = 0.18
    colors = {1: "#4e79a7", 4: "#59a14f", 16: "#f28e2b", 64: "#b07aa1"}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    offsets = np.arange(len(chunk_sizes)) - (len(chunk_sizes) - 1) / 2

    for i, cs in enumerate(chunk_sizes):
        vals = [saving_matrix[(sl, cs)] for sl in seq_lens]
        bars = ax.bar(
            x + offsets[i] * width,
            vals,
            width,
            label=f"chunk_{cs}",
            color=colors[cs],
            edgecolor="black",
            linewidth=0.4,
        )
        # Add value labels
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seq_lens])
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Memory Saving (%)", fontsize=12)
    ax.set_title("Memory Saving % by Chunk Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, 75)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig3_saving_pct.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig3_saving_pct.png")


def fig4_oom_comparison():
    exp3 = data["experiment3_max_seq_oom"]

    # Determine max seq_len before OOM for each method
    methods = ["unfused", "chunked_16", "chunked_1"]
    max_seq = {}
    for method in methods:
        max_non_oom = 0
        for d in exp3:
            if d["method"] == method and not d.get("oom", False):
                if d["seq_len"] > max_non_oom:
                    max_non_oom = d["seq_len"]
        max_seq[method] = max_non_oom

    labels = ["Unfused", "Chunked-16", "Chunked-1"]
    values = [max_seq["unfused"], max_seq["chunked_16"], max_seq["chunked_1"]]
    colors = ["#e15759", "#f28e2b", "#4e79a7"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            str(val),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Max Sequence Length Before OOM", fontsize=12)
    ax.set_title("Max Sequence Length Before OOM", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig4_oom_comparison.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig4_oom_comparison.png")


if __name__ == "__main__":
    fig1_logits_explosion()
    fig2_chunked_savings()
    fig3_saving_pct()
    fig4_oom_comparison()
    print("All charts generated successfully.")
