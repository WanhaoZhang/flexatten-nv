#!/usr/bin/env python3
"""Generate charts for GaLore/DoRA experiment."""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("/tmp/flexatten-nv-push/docs/galore_dora/results/galore_dora_results.json") as f:
    data = json.load(f)
OUT = "/tmp/flexatten-nv-push/docs/galore_dora/figures"
DPI = 150

def fig1():
    d = data["experiment1_memory"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = [r["name"] for r in d]
    mems = [r["peak_memory_gb"] for r in d]
    colors = ["#e15759" if r["method"]=="full" else "#4e79a7" if r["method"]=="lora" else "#59a14f" for r in d]
    bars = ax.bar(range(len(names)), mems, color=colors, edgecolor="black", linewidth=0.5)
    for bar, m in zip(bars, mems):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f"{m:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Peak Memory (GB)"); ax.set_title("Training Memory: Full FT vs LoRA vs DoRA")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#e15759", label="Full FT"), Patch(facecolor="#4e79a7", label="LoRA"), Patch(facecolor="#59a14f", label="DoRA")])
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig1_memory_footprint.png", dpi=DPI); plt.close(fig)

def fig2():
    d = data["experiment2_convergence"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"LoRA-r8": "#4e79a7", "LoRA-r32": "#f28e2b", "DoRA-r8": "#59a14f", "DoRA-r32": "#e15759"}
    for name, v in d.items():
        losses = v["losses"][::10]  # sample every 10 steps
        steps = list(range(0, 200, 10))
        ax.plot(steps, losses, linewidth=2, label=f"{name} (final={v['final_loss']:.2f})", color=colors[name])
    ax.set_xlabel("Training Step"); ax.set_ylabel("Loss")
    ax.set_title("Loss Convergence: LoRA vs DoRA (200 steps)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig2_convergence.png", dpi=DPI); plt.close(fig)

def fig3():
    d = data["experiment3_max_seq"]
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r["name"] for r in d]
    seqs = [r["max_seq_len"] for r in d]
    colors = ["#e15759" if r["method"]=="full" else "#4e79a7" if r["method"]=="lora" else "#59a14f" for r in d]
    bars = ax.bar(names, seqs, color=colors, edgecolor="black", linewidth=0.5)
    for bar, s in zip(bars, seqs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50, str(s), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Max Sequence Length"); ax.set_title("Maximum Trainable Sequence Length on L4")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig3_max_seq_len.png", dpi=DPI); plt.close(fig)

def fig4():
    d = data["experiment4_rank_sweep"]
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ranks = [r["rank"] for r in d]
    mems = [r["peak_memory_gb"] for r in d]
    steps = [r["step_time_ms"] for r in d]
    ax1.bar([str(r) for r in ranks], mems, color="#4e79a7", alpha=0.7, edgecolor="black", linewidth=0.5, label="Memory (GB)")
    ax1.set_ylabel("Peak Memory (GB)", color="#4e79a7")
    ax2 = ax1.twinx()
    ax2.plot([str(r) for r in ranks], steps, "o-", color="#e15759", linewidth=2, label="Step Time (ms)")
    ax2.set_ylabel("Step Time (ms)", color="#e15759")
    ax1.set_xlabel("LoRA Rank"); ax1.set_title("LoRA Rank Sweep: Memory and Training Speed")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="upper left")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(); fig.savefig(f"{OUT}/fig4_rank_sweep.png", dpi=DPI); plt.close(fig)

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("All GaLore/DoRA charts done.")
