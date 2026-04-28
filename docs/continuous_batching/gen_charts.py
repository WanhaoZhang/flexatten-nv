#!/usr/bin/env python3
"""Generate charts for Continuous Batching experiment."""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("/tmp/flexatten-nv-push/docs/continuous_batching/results/continuous_batching_results.json") as f:
    data = json.load(f)
OUT = "/tmp/flexatten-nv-push/docs/continuous_batching/figures"
DPI = 150

def fig1():
    d = data["experiment1_variable_output"]
    reqs = [r for r in d if "request_idx" in r]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = [r["max_tokens"] for r in reqs]
    gen = [r["generated_tokens"] for r in reqs]
    ax.bar([str(i) for i in x], gen, color="#4e79a7", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Max Tokens"); ax.set_ylabel("Generated Tokens")
    ax.set_title("Continuous Batching: Variable Output Lengths (batch of 5)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig1_variable_output.png", dpi=DPI); plt.close(fig)

def fig2():
    d = data["experiment2_throughput"]
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    bs = [r["batch_size"] for r in d]
    tps = [r["tokens_per_s"] for r in d]
    per_req = [r["per_request_ms"] for r in d]
    ax1.bar([str(b) for b in bs], tps, color="#4e79a7", alpha=0.7, edgecolor="black", linewidth=0.5, label="Throughput (tok/s)")
    ax1.set_ylabel("Throughput (tokens/s)", color="#4e79a7")
    ax2 = ax1.twinx()
    ax2.plot([str(b) for b in bs], per_req, "D-", color="#e15759", linewidth=2, label="Per-request (ms)")
    ax2.set_ylabel("Per-request Latency (ms)", color="#e15759")
    ax1.set_xlabel("Batch Size"); ax1.set_title("Continuous Batching: Throughput Scaling")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="center right")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig2_throughput_scaling.png", dpi=DPI); plt.close(fig)

def fig3():
    d = data["experiment3_mixed_lengths"]
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r["label"] for r in d]
    tps = [r["tokens_per_s"] for r in d]
    colors = ["#4e79a7", "#e15759", "#59a14f"]
    bars = ax.bar(labels, tps, color=colors, edgecolor="black", linewidth=0.5)
    for bar, t in zip(bars, tps):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20, f"{t:.0f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Throughput (tokens/s)"); ax.set_title("Scheduling: All-Short vs All-Long vs Mixed")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig3_mixed_scheduling.png", dpi=DPI); plt.close(fig)

def fig4():
    d = data["experiment4_concurrent"]
    reqs = [r for r in d if "request_idx" in r]
    summary = [r for r in d if "aggregate_tps" in r][0]
    fig, ax = plt.subplots(figsize=(10, 5))
    max_tok = [r["max_tokens"] for r in reqs]
    gen_tok = [r["generated_tokens"] for r in reqs]
    x = np.arange(len(reqs))
    ax.bar(x - 0.2, max_tok, 0.35, label="Max Tokens", color="#f28e2b", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.2, gen_tok, 0.35, label="Generated Tokens", color="#4e79a7", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"Req{r['request_idx']}" for r in reqs], fontsize=7)
    ax.set_ylabel("Tokens"); ax.set_title(f"Concurrent Request Simulation (16 req, {summary['total_time_ms']:.0f}ms, {summary['aggregate_tps']:.0f} tok/s)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig4_concurrent.png", dpi=DPI); plt.close(fig)

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("All Continuous Batching charts done.")
