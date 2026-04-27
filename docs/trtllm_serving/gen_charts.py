#!/usr/bin/env python3
"""Generate charts for TRT-LLM Serving analysis."""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("/tmp/flexatten-nv-push/docs/trtllm_serving/results/trtllm_serving_results.json") as f:
    data = json.load(f)

OUT = "/tmp/flexatten-nv-push/docs/trtllm_serving/figures"
DPI = 150

results = data["benchmark_results"]
precisions = ["BF16", "FP8", "INT4-AWQ"]
colors = {"BF16": "#e15759", "FP8": "#f28e2b", "INT4-AWQ": "#4e79a7"}


def fig1_throughput():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(2)
    w = 0.25
    for i, prec in enumerate(precisions):
        b1 = [d["tokens_per_sec"] for d in results if d["precision"] == prec and d["batch_size"] == 1]
        b8 = [d["tokens_per_sec"] for d in results if d["precision"] == prec and d["batch_size"] == 8]
        ax.bar(x + (i - 1) * w, [b1[0], b8[0]], w, label=prec, color=colors[prec], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Batch=1", "Batch=8"])
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("TRT-LLM Throughput: BF16 vs FP8 vs INT4-AWQ (Qwen-7B, L4)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig1_throughput.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig1")


def fig2_latency():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(2)
    w = 0.25
    for i, prec in enumerate(precisions):
        b1 = [d["latency_ms"] for d in results if d["precision"] == prec and d["batch_size"] == 1]
        b8 = [d["latency_ms"] for d in results if d["precision"] == prec and d["batch_size"] == 8]
        ax.bar(x + (i - 1) * w, [b1[0], b8[0]], w, label=prec, color=colors[prec], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Batch=1", "Batch=8"])
    ax.set_ylabel("E2E Latency (ms)")
    ax.set_title("TRT-LLM End-to-End Latency (input=128, output=128)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig2_latency.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig2")


def fig3_memory():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(2)
    w = 0.25
    for i, prec in enumerate(precisions):
        b1 = [d["gpu_peak_mem_gb"] for d in results if d["precision"] == prec and d["batch_size"] == 1]
        b8 = [d["gpu_peak_mem_gb"] for d in results if d["precision"] == prec and d["batch_size"] == 8]
        bars = ax.bar(x + (i - 1) * w, [b1[0], b8[0]], w, label=prec, color=colors[prec], edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, [b1[0], b8[0]]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f"{val:.1f}", ha="center", fontsize=8, fontweight="bold")
    ax.axhline(y=24, color="red", linestyle="--", alpha=0.5, label="L4 VRAM (24GB)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Batch=1", "Batch=8"])
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.set_title("TRT-LLM Peak Memory Usage (Qwen-7B, L4)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig3_memory.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig3")


def fig4_efficiency():
    fig, ax = plt.subplots(figsize=(8, 5))
    precs = ["BF16", "FP8", "INT4-AWQ"]
    mem_b1 = [data["benchmark_results"][i*2]["gpu_peak_mem_gb"] for i in range(3)]
    tps_b1 = [data["benchmark_results"][i*2]["tokens_per_sec"] for i in range(3)]
    eff = [t/m for t, m in zip(tps_b1, mem_b1)]
    cols = [colors[p] for p in precs]
    bars = ax.bar(precs, eff, color=cols, edgecolor="black", linewidth=0.5, width=0.5)
    for bar, val in zip(bars, eff):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Throughput per GB (tokens/s/GB)")
    ax.set_title("Memory Efficiency: Throughput per GB of VRAM (Batch=1)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig4_efficiency.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig4")


if __name__ == "__main__":
    fig1_throughput()
    fig2_latency()
    fig3_memory()
    fig4_efficiency()
    print("All TRT-LLM charts done.")
