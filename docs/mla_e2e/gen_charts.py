#!/usr/bin/env python3
"""Generate charts for MLA E2E experiment."""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("/tmp/flexatten-nv-push/docs/mla_e2e/results/mla_e2e_results.json") as f:
    data = json.load(f)
OUT = "/tmp/flexatten-nv-push/docs/mla_e2e/figures"
DPI = 150


def fig1():
    """KV Cache compression comparison."""
    d = data["experiment1_kv_compression"]
    configs = [r for r in d if "total_kv_mb" in r and "compression_vs_mha" in r]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    names = [r["method"] for r in configs]
    sizes = [r["total_kv_mb"] for r in configs]
    compression = [r["compression_vs_mha"] for r in configs]

    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = ax.bar(range(len(names)), sizes, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    for bar, s, c in zip(bars, sizes, compression):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{s:.0f}MB\n({c:.1f}x)", ha="center", fontsize=7, fontweight="bold")
    ax.set_ylabel("KV Cache Size (MB, seq=4096)")
    ax.set_title("KV Cache Compression: MHA vs GQA vs MLA")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig1_kv_compression.png", dpi=DPI)
    plt.close(fig)


def fig2():
    """MLA projection latency vs MHA KV read."""
    d = data["experiment2_projection_latency"]
    mha_reads = [r for r in d if r["method"] == "MHA KV Read"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    seq_lens = [r["seq_len"] for r in mha_reads]
    mha_ms = [r["latency_ms"] for r in mha_reads]

    ax.plot([str(s) for s in seq_lens], mha_ms, "o-", color="#4e79a7", linewidth=2,
            markersize=8, label="MHA KV Read")

    mla_methods = sorted(set(r["method"] for r in d if r["method"] != "MHA KV Read"))
    colors = ["#e15759", "#59a14f", "#f28e2b"]
    for i, method in enumerate(mla_methods):
        mla_data = [r for r in d if r["method"] == method]
        mla_ms = [r["latency_ms"] for r in mla_data]
        ax.plot([str(s) for s in seq_lens], mla_ms, "D-", color=colors[i % len(colors)],
                linewidth=2, markersize=6, label=method)

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("MLA Projection Latency vs MHA KV Read")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig2_projection_latency.png", dpi=DPI)
    plt.close(fig)


def fig3():
    """Memory breakdown at different seq lengths."""
    d = data["experiment3_memory_breakdown"]
    mem = [r for r in d if "model_weights_mb" in r and "seq_len" in r]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    seq_lens = [r["seq_len"] for r in mem]
    model_mb = [r["model_weights_mb"] for r in mem]
    kv_fp16 = [r["kv_fp16_mb"] for r in mem]
    kv_mla = [r["kv_mla_128_mb"] for r in mem]

    x = np.arange(len(seq_lens))
    width = 0.25

    ax.bar(x - width, model_mb, width, label="Model Weights", color="#4e79a7", edgecolor="black", linewidth=0.5)
    ax.bar(x, kv_fp16, width, label="KV (FP16)", color="#e15759", edgecolor="black", linewidth=0.5)
    ax.bar(x + width, kv_mla, width, label="KV (MLA-128)", color="#59a14f", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"SL={s}" for s in seq_lens])
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Breakdown: Model Weights + KV Cache")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig3_memory_breakdown.png", dpi=DPI)
    plt.close(fig)


def fig4():
    """End-to-end decode latency by KV config."""
    d = data["experiment4_e2e_decode"]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    methods = sorted(set(r["method"] for r in d))
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for i, method in enumerate(methods):
        method_data = [r for r in d if r["method"] == method]
        seq_lens = [str(r["seq_len"]) for r in method_data]
        latencies = [r["kv_read_ms"] for r in method_data]
        ax.plot(seq_lens, latencies, "o-", color=colors[i], linewidth=2, markersize=6, label=method)

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("KV Read Latency (ms)")
    ax.set_title("Decode KV Read Latency by Attention Configuration")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig4_e2e_decode.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("All MLA E2E charts done.")
