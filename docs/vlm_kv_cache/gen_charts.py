#!/usr/bin/env python3
"""Generate charts for VLM KV Cache analysis."""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("/tmp/flexatten-nv-push/docs/vlm_kv_cache/results/vlm_kv_cache_results.json") as f:
    data = json.load(f)
OUT = "/tmp/flexatten-nv-push/docs/vlm_kv_cache/figures"
DPI = 150

def fig1():
    d = data["experiment1_visual_tokens"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    resolutions = ["224x224", "512x512", "1080p", "1080p_FHD", "1440p_QHD", "4K_UHD"]
    patch_sizes = [14, 16, 32]
    colors = {"14": "#e15759", "16": "#f28e2b", "32": "#4e79a7"}
    x = np.arange(len(resolutions))
    w = 0.25
    for i, ps in enumerate(patch_sizes):
        vals = [r["num_visual_tokens"] for r in d if r["patch_size"]==ps and r["resolution"] in resolutions]
        res_order = [r for r in resolutions]
        ordered_vals = []
        for ro in res_order:
            for r in d:
                if r["patch_size"]==ps and r["resolution"]==ro:
                    ordered_vals.append(r["num_visual_tokens"])
        ax.bar(x + (i-1)*w, ordered_vals, w, label=f"PS={ps}", color=colors[str(ps)], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(resolutions, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Visual Token Count"); ax.set_title("Visual Token Count by Resolution and Patch Size")
    ax.legend(); ax.set_yscale("log"); ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig1_visual_tokens.png", dpi=DPI); plt.close(fig)

def fig2():
    d = data["experiment2_kv_comparison"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = [r["name"] for r in d]
    kv_gb = [r["kv_cache_gb"] for r in d]
    visual_pct = [r["visual_token_pct"] for r in d]
    colors = ["#4e79a7" if r["num_images"]==0 else "#e15759" for r in d]
    bars = ax.bar(range(len(names)), kv_gb, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=24, color="red", linestyle="--", alpha=0.5, label="L4 VRAM (24GB)")
    for i, (bar, vp) in enumerate(zip(bars, visual_pct)):
        if vp > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{vp:.0f}%\nvisual", ha="center", fontsize=7, color="red")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("KV Cache (GB)"); ax.set_title("KV Cache: Text-only vs Multimodal Inputs")
    ax.legend(); ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig2_kv_comparison.png", dpi=DPI); plt.close(fig)

def fig3():
    d = data["experiment3_prefill_simulation"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    names = [r["name"] for r in d]
    est_ms = [r["estimated_ms"] for r in d]
    compute = [r["compute_limited_ms"] for r in d]
    memory = [r["memory_limited_ms"] for r in d]
    x = np.arange(len(names))
    ax.bar(x - 0.2, memory, 0.35, label="Memory-bound", color="#4e79a7", edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.2, compute, 0.35, label="Compute-bound", color="#e15759", edgecolor="black", linewidth=0.5)
    ax.plot(x, est_ms, "D-", color="#59a14f", linewidth=2, label="Estimated (max)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Prefill Latency (ms)"); ax.set_title("Prefill Latency Estimation (L4)")
    ax.legend(fontsize=8); ax.set_yscale("log"); ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig3_prefill_latency.png", dpi=DPI); plt.close(fig)

def fig4():
    d = data["experiment4_cache_strategies"]
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r["strategy"] for r in d]
    savings = [r["compute_saving_pct"] for r in d]
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    bars = ax.bar(names, savings, color=colors, edgecolor="black", linewidth=0.5)
    for bar, s in zip(bars, savings):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{s:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Compute Saving (%)"); ax.set_title("Cache Strategy: Compute Savings")
    ax.set_ylim(0, 110); ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig4_cache_strategies.png", dpi=DPI); plt.close(fig)

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("All VLM KV Cache charts done.")
