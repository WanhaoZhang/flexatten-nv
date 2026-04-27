#!/usr/bin/env python3
"""Generate charts for MoE Inference Wall experiment results."""

import json
import matplotlib.pyplot as plt
import numpy as np

with open("/tmp/flexatten-nv-push/docs/moe_wall/results/moe_wall_results.json") as f:
    data = json.load(f)

OUT = "/tmp/flexatten-nv-push/docs/moe_wall/figures"
DPI = 150


def fig1_weight_footprint():
    exp1 = data["experiment1_weight_footprint"]
    names = [d["name"].replace(" (", "\n(") for d in exp1]
    total = [d["total_weights_gb"] for d in exp1]
    active = [d["active_weights_gb"] for d in exp1]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x - w/2, total, w, label="Total Weights", color="#e15759", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w/2, active, w, label="Active Weights (top-2)", color="#4e79a7", edgecolor="black", linewidth=0.5)
    for bar, val in zip(b1, total):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}", ha="center", fontsize=8, fontweight="bold")
    for bar, val in zip(b2, active):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Weight Memory (GB, FP16)")
    ax.set_title("MoE Model Weight Footprint: Total vs Active")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig1_weight_footprint.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig1")


def fig2_load_balance():
    exp3 = data["experiment3_load_balance"]
    names = [d["distribution"] for d in exp3]
    balance = [d["load_balance_ratio"] * 100 for d in exp3]
    wasted = [d["wasted_compute_pct"] for d in exp3]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    w = 0.35
    ax1.bar(x - w/2, balance, w, label="Load Balance %", color="#4e79a7", edgecolor="black", linewidth=0.5)
    ax1.bar(x + w/2, wasted, w, label="Wasted Compute %", color="#e15759", edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Expert Load Balance vs Wasted Compute")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig2_load_balance.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig2")


def fig3_dense_vs_moe():
    exp4 = data["experiment4_dense_vs_moe"]
    batch = [d["batch_size"] for d in exp4]
    dense = [d["dense_us"] for d in exp4]
    moe = [d["moe_us"] for d in exp4]
    ratio = [d["moe_overhead_ratio"] for d in exp4]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(batch))
    w = 0.35
    ax1.bar(x - w/2, dense, w, label="Dense FFN", color="#4e79a7", edgecolor="black", linewidth=0.5)
    b2 = ax1.bar(x + w/2, moe, w, label="MoE FFN (8 experts, top-2)", color="#e15759", edgecolor="black", linewidth=0.5)
    for bar, r in zip(b2, ratio):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, f"{r:.1f}x", ha="center", fontsize=9, fontweight="bold", color="#e15759")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(b) for b in batch])
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Latency (us)")
    ax1.set_title("Dense vs MoE FFN Latency (L4 GPU)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig3_dense_vs_moe.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig3")


def fig4_decode_throughput():
    exp1 = data["experiment1_weight_footprint"]
    names = [d["name"].split("(")[0].strip() for d in exp1 if d["decode_tok_per_s"] > 0]
    tok_s = [d["decode_tok_per_s"] for d in exp1 if d["decode_tok_per_s"] > 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4e79a7", "#59a14f", "#f28e2b", "#b07aa1", "#e15759"]
    bars = ax.barh(names, tok_s, color=colors[:len(names)], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, tok_s):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2, f"{val:.0f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Theoretical Decode Throughput (tokens/s)")
    ax.set_title("Decode Throughput by Model (Bandwidth-Bound, L4 300GB/s)")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig4_decode_throughput.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig4")


if __name__ == "__main__":
    fig1_weight_footprint()
    fig2_load_balance()
    fig3_dense_vs_moe()
    fig4_decode_throughput()
    print("All MoE charts done.")
