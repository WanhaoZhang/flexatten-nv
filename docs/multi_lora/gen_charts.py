#!/usr/bin/env python3
"""Generate charts for Multi-LoRA experiment."""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("/tmp/flexatten-nv-push/docs/multi_lora/results/multi_lora_results.json") as f:
    data = json.load(f)
OUT = "/tmp/flexatten-nv-push/docs/multi_lora/figures"
DPI = 150

def fig1():
    d = data["experiment1_multi_throughput"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = [r["num_adapters"] for r in d]
    tps = [r["tokens_per_s"] for r in d]
    rel = [r["throughput_vs_single"] for r in d]
    bars = ax.bar([str(i) for i in x], tps, color="#4e79a7", edgecolor="black", linewidth=0.5)
    for bar, r in zip(bars, rel):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20, f"{r:.2f}x", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Number of LoRA Adapters"); ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Multi-LoRA Throughput (rank=8, BS=4, seq=64)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig1_multi_throughput.png", dpi=DPI); plt.close(fig)

def fig2():
    d = data["experiment2_memory_overhead"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = [r["rank"] for r in d]
    y = [r["per_adapter_mb"] for r in d]
    ax.bar([str(i) for i in x], y, color="#f28e2b", edgecolor="black", linewidth=0.5)
    for i, v in enumerate(y):
        ax.text(i, v+0.3, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xlabel("LoRA Rank"); ax.set_ylabel("Memory per Adapter (MB)")
    ax.set_title("LoRA Memory Overhead per Adapter (8 adapters)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig2_memory_overhead.png", dpi=DPI); plt.close(fig)

def fig3():
    d = data["experiment3_same_vs_mixed"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(d)); w = 0.35
    same = [r["same_lora_ms"] for r in d]
    mixed = [r["mixed_lora_ms"] for r in d]
    sd = [r["slowdown"] for r in d]
    bars1 = ax.bar(x-w/2, same, w, label="Same-LoRA (batched)", color="#4e79a7", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x+w/2, mixed, w, label="Mixed-LoRA (sequential)", color="#e15759", edgecolor="black", linewidth=0.5)
    for i, s in enumerate(sd):
        ax.text(x[i]+w/2, mixed[i]+10, f"{s:.1f}x", ha="center", fontsize=9, fontweight="bold", color="#e15759")
    ax.set_xticks(x); ax.set_xticklabels([f"BS={r['batch_size']}" for r in d])
    ax.set_ylabel("Latency (ms)"); ax.set_title("Same-LoRA vs Mixed-LoRA Batch Latency")
    ax.legend(); ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig3_same_vs_mixed.png", dpi=DPI); plt.close(fig)

def fig4():
    d = data["experiment4_rank_impact"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = [r["rank"] for r in d]
    y = [r["latency_ms"] for r in d]
    colors = ["#59a14f" if r["rank"]==0 else "#4e79a7" for r in d]
    bars = ax.bar([str(i) for i in x], y, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=y[0], color="red", linestyle="--", alpha=0.5, label=f"Baseline (no LoRA): {y[0]:.1f}ms")
    ax.set_xlabel("LoRA Rank"); ax.set_ylabel("Inference Latency (ms)")
    ax.set_title("LoRA Rank Impact on Inference Latency (seq=128)")
    ax.legend(); ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig4_rank_impact.png", dpi=DPI); plt.close(fig)

def fig5():
    d = data["experiment5_max_adapters"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = [r["num_adapters"] for r in d]
    y = [r["memory_gb"] for r in d]
    ax.plot(x, y, "o-", color="#4e79a7", linewidth=2, markersize=5)
    ax.axhline(y=24, color="red", linestyle="--", alpha=0.5, label="L4 VRAM (24GB)")
    ax.axhline(y=1, color="#59a14f", linestyle=":", alpha=0.5, label="Base Model (~1GB)")
    ax.set_xlabel("Number of LoRA Adapters"); ax.set_ylabel("GPU Memory (GB)")
    ax.set_title("LoRA Adapter Capacity on L4 (rank=8)")
    ax.legend(); ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig5_max_adapters.png", dpi=DPI); plt.close(fig)

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("All Multi-LoRA charts done.")
