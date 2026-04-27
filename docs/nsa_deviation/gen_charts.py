#!/usr/bin/env python3
"""Generate charts for NSA Deviation analysis."""

import json
import matplotlib.pyplot as plt
import numpy as np

with open("/tmp/flexatten-nv-push/docs/nsa_deviation/results/nsa_deviation_results.json") as f:
    data = json.load(f)

OUT = "/tmp/flexatten-nv-push/docs/nsa_deviation/figures"
DPI = 150


def fig1_sparsity_vs_speedup():
    exp1 = data["experiment1_sparsity_vs_speedup"]
    configs = [d["config"] for d in exp1]
    actual = [d["actual_speedup"] for d in exp1]
    theoretical = [d["theoretical_speedup"] for d in exp1]
    deviation = [d["deviation_pct"] for d in exp1]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(configs))
    w = 0.3

    bars1 = ax1.bar(x - w/2, theoretical, w, label="Theoretical Speedup", color="#4e79a7", edgecolor="black", linewidth=0.5)
    bars2 = ax1.bar(x + w/2, actual, w, label="Actual Speedup", color="#e15759", edgecolor="black", linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, deviation, "D--", color="#59a14f", markersize=7, linewidth=2, label="Deviation %")
    ax2.set_ylabel("Deviation (%)", color="#59a14f")

    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Speedup (x)")
    ax1.set_title("NSA-like Sparsity vs Actual Speedup (FlexAttention/Triton, L4)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig1_sparsity_vs_speedup.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig1")


def fig2_block_size_impact():
    exp2 = data["experiment2_block_size_impact"]
    blocks = [d["block_size"] for d in exp2]
    fwd = [d["forward_ms"] for d in exp2]
    bwd = [d["backward_ms"] for d in exp2]
    total = [d["total_ms"] for d in exp2]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(blocks))
    w = 0.25
    ax.bar(x - w, fwd, w, label="Forward", color="#f28e2b", edgecolor="black", linewidth=0.5)
    ax.bar(x, bwd, w, label="Backward", color="#4e79a7", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, total, w, label="Total", color="#e15759", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"BS={b}" for b in blocks])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("BLOCK_SIZE Impact on NSA Performance (No Difference!)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig2_block_size.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig2")


def fig3_deviation_curve():
    exp3 = data["experiment3_deviation_curve"]
    sparsity = [d["sparsity"] * 100 for d in exp3]
    theoretical = [d["theoretical_speedup"] for d in exp3]
    actual = [d["actual_speedup"] for d in exp3]
    efficiency = [d["efficiency"] * 100 for d in exp3]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(sparsity, theoretical, "o-", color="#4e79a7", linewidth=2, label="Theoretical Speedup")
    ax1.plot(sparsity, actual, "s-", color="#e15759", linewidth=2, label="Actual Speedup")
    ax1.set_xlabel("Sparsity (%)")
    ax1.set_ylabel("Speedup (x)")

    ax2 = ax1.twinx()
    ax2.fill_between(sparsity, efficiency, alpha=0.2, color="#59a14f")
    ax2.plot(sparsity, efficiency, "^--", color="#59a14f", linewidth=2, label="Efficiency (%)")
    ax2.set_ylabel("Efficiency (%)", color="#59a14f")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")
    ax1.set_title("Theoretical vs Actual: Sparsity Efficiency Curve")
    ax1.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig3_deviation_curve.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig3")


if __name__ == "__main__":
    fig1_sparsity_vs_speedup()
    fig2_block_size_impact()
    fig3_deviation_curve()
    print("All NSA deviation charts done.")
