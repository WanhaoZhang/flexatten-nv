#!/usr/bin/env python3
"""Generate charts for TRT-LLM Serving experiment."""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("/tmp/flexatten-nv-push/docs/trt_llm_serving/results/trt_llm_serving_results.json") as f:
    data = json.load(f)
OUT = "/tmp/flexatten-nv-push/docs/trt_llm_serving/figures"
DPI = 150


def fig1():
    """vLLM serving throughput matrix."""
    d = data["experiment1_vllm_baseline"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bs_vals = [r["batch_size"] for r in d]
    max_tok_vals = [r["max_tokens"] for r in d]
    output_tps = [r["output_tps"] for r in d]

    labels = [f"BS={b}\nMT={m}" for b, m in zip(bs_vals, max_tok_vals)]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(labels)))

    bars = ax.bar(range(len(labels)), output_tps, color=colors, edgecolor="black", linewidth=0.5)
    for bar, t in zip(bars, output_tps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{t:.0f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Output Throughput (tokens/s)")
    ax.set_title("vLLM Serving Throughput (BS × Max Tokens)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig1_vllm_serving.png", dpi=DPI)
    plt.close(fig)


def fig2():
    """Theoretical decode performance comparison."""
    d = data["experiment2_efficiency_model"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Filter for SL=2048 and SL=8192
    for idx, sl in enumerate([2048, 8192]):
        subset = [r for r in d if r["seq_len"] == sl]
        methods = [r["method"] for r in subset]
        decode_ms = [r["total_decode_ms"] for r in subset]
        batch_tps = [r["batch32_tps"] for r in subset]

        ax = axes[idx]
        x = np.arange(len(methods))
        width = 0.35
        bars1 = ax.bar(x - width / 2, decode_ms, width, color="#e15759", alpha=0.7,
                       edgecolor="black", linewidth=0.5, label="Decode (ms)")
        ax.set_ylabel("Decode Time (ms)", color="#e15759")

        ax2 = ax.twinx()
        ax2.bar(x + width / 2, batch_tps, width, color="#59a14f", alpha=0.7,
                edgecolor="black", linewidth=0.5, label="BS=32 TPS")
        ax2.set_ylabel("Throughput (tok/s, BS=32)", color="#59a14f")

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"Theoretical Performance (SL={sl})")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

    plt.tight_layout()
    fig.savefig(f"{OUT}/fig2_theoretical.png", dpi=DPI)
    plt.close(fig)


def fig3():
    """TTFT analysis."""
    d = data["experiment3_latency"]
    ttft = [r for r in d if "ttft_ms" in r]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    pls = [r["prompt_length"] for r in ttft]
    ttft_ms = [r["ttft_ms"] for r in ttft]
    prefill_tps = [r["prefill_tps"] for r in ttft]

    ax1.bar([str(p) for p in pls], ttft_ms, color="#4e79a7", alpha=0.7,
            edgecolor="black", linewidth=0.5, label="TTFT (ms)")
    ax1.set_ylabel("TTFT (ms)", color="#4e79a7")
    ax1.set_xlabel("Prompt Length (tokens)")

    ax2 = ax1.twinx()
    ax2.plot([str(p) for p in pls], prefill_tps, "D-", color="#e15759", linewidth=2,
             markersize=7, label="Prefill TPS")
    ax2.set_ylabel("Prefill Speed (tokens/s)", color="#e15759")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("vLLM TTFT vs Prompt Length")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig3_ttft.png", dpi=DPI)
    plt.close(fig)


def fig4():
    """Concurrency limits comparison."""
    d = data["experiment4_concurrency"]
    methods = sorted(set(r["method"] for r in d))
    seq_lens = sorted(set(r["seq_len"] for r in d))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(seq_lens))
    width = 0.2
    colors = ["#4e79a7", "#e15759", "#59a14f", "#f28e2b"]

    for i, method in enumerate(methods):
        vals = []
        for sl in seq_lens:
            match = [r for r in d if r["method"] == method and r["seq_len"] == sl]
            vals.append(match[0]["max_concurrent_requests"] if match else 0)
        ax.bar(x + i * width, vals, width, color=colors[i % len(colors)],
               edgecolor="black", linewidth=0.5, label=method)

    ax.set_xticks(x + width * len(methods) / 2)
    ax.set_xticklabels([f"SL={s}" for s in seq_lens])
    ax.set_ylabel("Max Concurrent Requests")
    ax.set_title("Maximum Concurrent Requests by Framework & Precision")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig4_concurrency.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("All TRT-LLM Serving charts done.")
