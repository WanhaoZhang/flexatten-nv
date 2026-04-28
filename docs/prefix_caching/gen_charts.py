#!/usr/bin/env python3
"""Generate charts for Prefix Caching experiment."""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("/tmp/flexatten-nv-push/docs/prefix_caching/results/prefix_caching_results.json") as f:
    data = json.load(f)
OUT = "/tmp/flexatten-nv-push/docs/prefix_caching/figures"
DPI = 150

def fig1():
    d = data["experiment1_ttft"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(d)); w = 0.35
    no_cache = [r["ttft_no_cache_ms"] for r in d]
    cached = [r["ttft_cached_ms"] for r in d]
    ax.bar(x-w/2, no_cache, w, label="No Cache", color="#e15759", edgecolor="black", linewidth=0.5)
    ax.bar(x+w/2, cached, w, label="With Prefix Cache", color="#4e79a7", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"Q{r['question_idx']}" for r in d])
    ax.set_ylabel("TTFT (ms)"); ax.set_title("Prefix Caching: TTFT Comparison (1000-token prefix)")
    ax.legend(); ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig1_ttft_comparison.png", dpi=DPI); plt.close(fig)

def fig2():
    d = data["experiment2_prefix_length"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = [r["prefix_length_tokens"] for r in d]
    cold = [r["ttft_cold_ms"] for r in d]
    warm = [r["ttft_warm_ms"] for r in d]
    ax.plot(x, cold, "o-", color="#e15759", linewidth=2, label="Cold (no cache)")
    ax.plot(x, warm, "s-", color="#4e79a7", linewidth=2, label="Warm (cached)")
    ax.fill_between(x, warm, cold, alpha=0.15, color="#59a14f", label="Saved time")
    ax.set_xlabel("Prefix Length (tokens)"); ax.set_ylabel("TTFT (ms)")
    ax.set_title("Prefix Caching Benefit vs Prefix Length")
    ax.legend(); ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig2_prefix_length.png", dpi=DPI); plt.close(fig)

def fig3():
    d = data["experiment3_eviction"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(d)); w = 0.25
    cold = [r["cold_ms"] for r in d]
    warm = [r["warm_ms"] for r in d]
    evict = [r["after_eviction_ms"] for r in d]
    ax.bar(x-w, cold, w, label="Cold", color="#e15759", edgecolor="black", linewidth=0.5)
    ax.bar(x, warm, w, label="Warm (cached)", color="#4e79a7", edgecolor="black", linewidth=0.5)
    ax.bar(x+w, evict, w, label="After eviction", color="#f28e2b", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"Doc{r['doc_idx']}" for r in d])
    ax.set_ylabel("TTFT (ms)"); ax.set_title("Cache Eviction: Multi-Document Interleave")
    ax.legend(); ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig3_eviction.png", dpi=DPI); plt.close(fig)

def fig4():
    d = data["experiment4_batch"]
    d = [r for r in d if "batch_size" in r]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    bs = [r["batch_size"] for r in d]
    tps = [r["tokens_per_s"] for r in d]
    per_req = [r["per_request_ms"] for r in d]
    ax1.bar([str(b) for b in bs], tps, color="#4e79a7", alpha=0.7, edgecolor="black", linewidth=0.5, label="Throughput (tok/s)")
    ax1.set_ylabel("Throughput (tokens/s)", color="#4e79a7")
    ax2 = ax1.twinx()
    ax2.plot([str(b) for b in bs], per_req, "D-", color="#e15759", linewidth=2, label="Per-request (ms)")
    ax2.set_ylabel("Per-request Latency (ms)", color="#e15759")
    ax1.set_xlabel("Batch Size"); ax1.set_title("Batch Prefix Sharing Throughput")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="upper left")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig4_batch_throughput.png", dpi=DPI); plt.close(fig)

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("All Prefix Caching charts done.")
