#!/usr/bin/env python3
"""Generate charts for MLA/GQA experiment."""

import json
import matplotlib.pyplot as plt
import numpy as np

with open("/tmp/flexatten-nv-push/docs/mla_gqa_analysis/results/mla_gqa_results.json") as f:
    data = json.load(f)

OUT = "/tmp/flexatten-nv-push/docs/mla_gqa_analysis/figures"
DPI = 150


def fig1_kv_memory():
    exp1 = data["experiment1_kv_memory"]
    configs = ["MHA-32", "GQA-8", "GQA-4", "MLA-512"]
    seq_lens = sorted(set(d["seq_len"] for d in exp1))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cfg in configs:
        vals = [d["kv_actual_mb"] for d in exp1 if d["config"] == cfg]
        seqs = [d["seq_len"] for d in exp1 if d["config"] == cfg]
        ax.plot(seqs, vals, marker="o", linewidth=2, label=cfg)

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("KV Cache Memory (MB)")
    ax.set_title("KV Cache Memory: MHA vs GQA vs MLA")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig1_kv_memory.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig1")


def fig2_decode_latency():
    exp2 = data["experiment2_decode_latency"]
    configs = ["MHA-32", "GQA-8", "GQA-4", "MLA-512"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cfg in configs:
        vals = [d["latency_us"] for d in exp2 if d["config"] == cfg]
        seqs = [d["seq_len"] for d in exp2 if d["config"] == cfg]
        ax.plot(seqs, vals, marker="s", linewidth=2, label=cfg)

    ax.set_xlabel("Sequence Length (KV Cache Length)")
    ax.set_ylabel("Decode Latency (us)")
    ax.set_title("Decode Latency: Bandwidth-Bound (query=1 token)")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig2_decode_latency.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig2")


def fig3_prefill_latency():
    exp3 = data["experiment3_prefill_latency"]
    configs = ["MHA-32", "GQA-8", "GQA-4", "MLA-512"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cfg in configs:
        vals = [d["prefill_us"] for d in exp3 if d["config"] == cfg]
        seqs = [d["seq_len"] for d in exp3 if d["config"] == cfg]
        ax.plot(seqs, vals, marker="^", linewidth=2, label=cfg)

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Prefill Latency (us)")
    ax.set_title("Prefill Latency: Compute-Bound (full sequence)")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig3_prefill_latency.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig3")


def fig4_mla_projection():
    exp4 = data["experiment4_mla_projection"]
    seq_lens = [d["seq_len"] for d in exp4]
    proj = [d["projection_us"] for d in exp4]
    direct = [d["direct_access_us"] for d in exp4]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(seq_lens))
    w = 0.35
    ax.bar(x - w/2, proj, w, label="MLA Projection (latent→KV)", color="#e15759", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, direct, w, label="Direct KV Access", color="#4e79a7", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seq_lens])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Latency (us)")
    ax.set_title("MLA Latent→KV Projection Overhead")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{OUT}/fig4_mla_projection.png", dpi=DPI)
    plt.close(fig)
    print("Saved fig4")


if __name__ == "__main__":
    fig1_kv_memory()
    fig2_decode_latency()
    fig3_prefill_latency()
    fig4_mla_projection()
    print("All MLA/GQA charts done.")
