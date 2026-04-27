#!/usr/bin/env python3
"""Generate Triton Fusion experiment charts."""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Load data
with open("/tmp/flexatten-nv-push/docs/triton_fusion/results/triton_fusion_results.json") as f:
    data = json.load(f)

OUT = "/tmp/flexatten-nv-push/docs/triton_fusion/figures"
DPI = 150

# ---------------------------------------------------------------------------
# Fig 1: Baseline latency bar chart
# ---------------------------------------------------------------------------
exp1 = data["experiment1_baseline_latency"]
labels = [f"b{d['batch']}_s{d['seq']}" for d in exp1]
latencies = [d["baseline_us"] for d in exp1]

fig1, ax1 = plt.subplots(figsize=(8, 4.5))
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
bars = ax1.bar(labels, latencies, color=colors, edgecolor="white", linewidth=0.6)
ax1.set_xlabel("Configuration (batch_seq)", fontsize=11)
ax1.set_ylabel("Latency (us)", fontsize=11)
ax1.set_title("Baseline Latency (RMSNorm + RoPE + SiLU Separate)", fontsize=13)
for bar, val in zip(bars, latencies):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
             f"{val:.0f}", ha="center", va="bottom", fontsize=8)
ax1.set_ylim(0, max(latencies) * 1.15)
fig1.tight_layout()
fig1.savefig(f"{OUT}/fig1_baseline_latency.png", dpi=DPI)
plt.close(fig1)
print("Saved fig1_baseline_latency.png")

# ---------------------------------------------------------------------------
# Fig 2: Grouped bar – Triton SiLU vs PyTorch SiLU
# ---------------------------------------------------------------------------
exp2 = data["experiment2_fused_latency"]
labels2 = [f"b{d['batch']}_s{d['seq']}" for d in exp2]
triton_silu = [d["fused_silu_us"] for d in exp2]
torch_silu = [d["torch_silu_us"] for d in exp2]

x2 = np.arange(len(labels2))
w = 0.32

fig2, ax2 = plt.subplots(figsize=(9, 4.5))
b1 = ax2.bar(x2 - w / 2, triton_silu, w, label="Triton SiLU", color="#4C72B0", edgecolor="white")
b2 = ax2.bar(x2 + w / 2, torch_silu, w, label="PyTorch SiLU", color="#DD8452", edgecolor="white")
ax2.set_xticks(x2)
ax2.set_xticklabels(labels2, fontsize=9)
ax2.set_xlabel("Configuration (batch_seq)", fontsize=11)
ax2.set_ylabel("Latency (us)", fontsize=11)
ax2.set_title("Triton SiLU vs PyTorch SiLU Latency", fontsize=13)
ax2.legend(fontsize=10)
for bar_group in (b1, b2):
    for bar in bar_group:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 5,
                 f"{h:.0f}", ha="center", va="bottom", fontsize=7)
ax2.set_ylim(0, max(max(triton_silu), max(torch_silu)) * 1.18)
fig2.tight_layout()
fig2.savefig(f"{OUT}/fig2_fused_vs_torch_silu.png", dpi=DPI)
plt.close(fig2)
print("Saved fig2_fused_vs_torch_silu.png")

# ---------------------------------------------------------------------------
# Fig 3: Grouped bar – unfused vs fused memory traffic (MB)
# ---------------------------------------------------------------------------
exp3 = data["experiment3_memory_traffic"]
seq_lens = [str(d["seq_len"]) for d in exp3]
unfused_mb = [d["unfused_traffic_mb"] for d in exp3]
fused_mb = [d["fused_traffic_mb"] for d in exp3]

x3 = np.arange(len(seq_lens))
w3 = 0.32

fig3, ax3 = plt.subplots(figsize=(9, 4.5))
b3u = ax3.bar(x3 - w3 / 2, unfused_mb, w3, label="Unfused", color="#C44E52", edgecolor="white")
b3f = ax3.bar(x3 + w3 / 2, fused_mb, w3, label="Fused", color="#55A868", edgecolor="white")
ax3.set_xticks(x3)
ax3.set_xticklabels(seq_lens, fontsize=10)
ax3.set_xlabel("Sequence Length", fontsize=11)
ax3.set_ylabel("HBM Traffic (MB)", fontsize=11)
ax3.set_title("Unfused vs Fused Memory Traffic", fontsize=13)
ax3.legend(fontsize=10)
for bar_group in (b3u, b3f):
    for bar in bar_group:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 5,
                 f"{h:.1f}", ha="center", va="bottom", fontsize=7)
ax3.set_ylim(0, max(unfused_mb) * 1.15)
fig3.tight_layout()
fig3.savefig(f"{OUT}/fig3_memory_traffic.png", dpi=DPI)
plt.close(fig3)
print("Saved fig3_memory_traffic.png")

# ---------------------------------------------------------------------------
# Fig 4: Grouped bar – RMSNorm speedup across dim x n_rows
# ---------------------------------------------------------------------------
exp4 = data["experiment4_rmsnorm_standalone"]
dims = sorted(set(d["dim"] for d in exp4))
nrows_vals = sorted(set(d["n_rows"] for d in exp4))

# Build a speedup matrix: rows=dim, cols=n_rows
speedup_map = {}
for d in exp4:
    speedup_map[(d["dim"], d["n_rows"])] = d["speedup"]

fig4, ax4 = plt.subplots(figsize=(9, 5))
x4 = np.arange(len(nrows_vals))
n_groups = len(dims)
total_width = 0.75
bar_w = total_width / n_groups
colors4 = ["#4C72B0", "#55A868", "#C44E52"]

for i, dim in enumerate(dims):
    offsets = x4 - total_width / 2 + bar_w * (i + 0.5)
    vals = [speedup_map[(dim, nr)] for nr in nrows_vals]
    rects = ax4.bar(offsets, vals, bar_w * 0.9, label=f"dim={dim}",
                    color=colors4[i], edgecolor="white")
    for rect, v in zip(rects, vals):
        ax4.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.03,
                 f"{v:.2f}x", ha="center", va="bottom", fontsize=8)

ax4.set_xticks(x4)
ax4.set_xticklabels([str(nr) for nr in nrows_vals], fontsize=10)
ax4.set_xlabel("n_rows", fontsize=11)
ax4.set_ylabel("Speedup (Triton / PyTorch)", fontsize=11)
ax4.set_title("RMSNorm Speedup: Triton vs PyTorch", fontsize=13)
ax4.legend(fontsize=10)
ax4.set_ylim(0, 3.0)
ax4.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="_nolegend_")
fig4.tight_layout()
fig4.savefig(f"{OUT}/fig4_rmsnorm_speedup.png", dpi=DPI)
plt.close(fig4)
print("Saved fig4_rmsnorm_speedup.png")

print("\nAll figures generated successfully.")
