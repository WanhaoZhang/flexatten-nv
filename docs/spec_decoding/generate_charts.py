#!/usr/bin/env python3
"""Generate charts for Speculative Decoding experiment results."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DATA_PATH = "/tmp/flexatten-nv-push/docs/spec_decoding/results/spec_decoding_results.json"
OUT_DIR = "/tmp/flexatten-nv-push/docs/spec_decoding/figures/"

with open(DATA_PATH) as f:
    data = json.load(f)

# ---------------------------------------------------------------------------
# Common style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# ===========================================================================
# Fig 1 – Speedup vs Acceptance Rate (separate lines for gamma)
# ===========================================================================
exp1 = data["experiment1_acceptance_rate"]
TARGET_GAMMAS = [1, 2, 4, 5, 8]
MARKERS = {1: "o", 2: "s", 4: "^", 5: "D", 8: "v"}

fig1, ax1 = plt.subplots(figsize=(8, 5))
for g in TARGET_GAMMAS:
    rows = sorted(
        [r for r in exp1 if r["gamma"] == g],
        key=lambda r: r["acceptance_rate"],
    )
    xs = [r["acceptance_rate"] for r in rows]
    ys = [r["speedup"] for r in rows]
    ax1.plot(xs, ys, marker=MARKERS[g], label=f"gamma={g}", linewidth=1.8, markersize=5)

ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0, label="Break-even")
ax1.set_xlabel("Acceptance Rate")
ax1.set_ylabel("Speedup")
ax1.set_title("Speculative Decoding Speedup vs Acceptance Rate (alpha=0.1)")
ax1.legend(loc="upper left")
ax1.set_xlim(0.48, 1.02)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(OUT_DIR + "fig1_acceptance_speedup.png", dpi=150)
plt.close(fig1)
print("[OK] fig1_acceptance_speedup.png")

# ===========================================================================
# Fig 2 – Task-type speedup bar chart
# ===========================================================================
exp3 = data["experiment3_task_simulation"]
TASK_LABELS = ["Chat", "Code", "Math", "Translation", "Creative"]
tasks_full = {
    "Chat": "Chat (generic)",
    "Code": "Code generation",
    "Math": "Math reasoning",
    "Translation": "Translation",
    "Creative": "Creative writing",
}

speedups = []
colors = []
for short in TASK_LABELS:
    row = next(r for r in exp3 if r["task"] == tasks_full[short])
    speedups.append(row["speedup_analytical"])
    colors.append("#4C8C2B" if row["profitable"] is True or row["profitable"] == "True" else "#C0392B")

fig2, ax2 = plt.subplots(figsize=(8, 5))
bars = ax2.bar(TASK_LABELS, speedups, color=colors, edgecolor="black", linewidth=0.6, width=0.6)

# Annotate each bar
for bar, val in zip(bars, speedups):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.2f}x",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0, label="Break-even")
ax2.set_ylabel("Speedup")
ax2.set_title("Speculative Decoding Speedup by Task Type")
ax2.legend(loc="upper right")
ax2.set_ylim(0, max(speedups) * 1.18)
ax2.grid(axis="y", alpha=0.3)

# Color legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4C8C2B", edgecolor="black", label="Profitable"),
    Patch(facecolor="#C0392B", edgecolor="black", label="Loss"),
]
ax2.legend(handles=legend_elements, loc="upper right")

fig2.tight_layout()
fig2.savefig(OUT_DIR + "fig2_task_type_speedup.png", dpi=150)
plt.close(fig2)
print("[OK] fig2_task_type_speedup.png")

# ===========================================================================
# Fig 3 – Optimal gamma vs acceptance rate (lines for alpha)
# ===========================================================================
exp5 = data["experiment5_optimal_gamma"]
ALPHA5 = [0.05, 0.08, 0.10, 0.15, 0.20]
ALPHA5_FLOATS = [float(a) for a in ALPHA5]
CMAP5 = plt.cm.viridis

fig3, ax3 = plt.subplots(figsize=(8, 5))
for idx, a in enumerate(ALPHA5_FLOATS):
    rows = sorted(
        [r for r in exp5 if abs(r["alpha"] - a) < 1e-6],
        key=lambda r: r["acceptance_rate"],
    )
    xs = [r["acceptance_rate"] for r in rows]
    ys = [r["optimal_gamma"] for r in rows]
    color = CMAP5(idx / (len(ALPHA5_FLOATS) - 1))
    ax3.plot(xs, ys, marker="o", label=f"alpha={a:.2f}", linewidth=1.8, markersize=5, color=color)

ax3.set_xlabel("Acceptance Rate")
ax3.set_ylabel("Optimal gamma")
ax3.set_title("Optimal gamma vs Acceptance Rate")
ax3.legend(loc="upper left")
ax3.grid(True, alpha=0.3)
ax3.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
fig3.tight_layout()
fig3.savefig(OUT_DIR + "fig3_optimal_gamma.png", dpi=150)
plt.close(fig3)
print("[OK] fig3_optimal_gamma.png")

# ===========================================================================
# Fig 4 – Speedup vs alpha grouped by acceptance_rate (gamma=5)
# ===========================================================================
exp2 = data["experiment2_draft_cost"]
GAMMA_FILTER = 5
AR_ORDER = [0.7, 0.8, 0.9, 0.95]

# Collect alpha values that actually appear with gamma=5
alpha_vals = sorted(set(r["alpha"] for r in exp2 if r["gamma"] == GAMMA_FILTER))

fig4, ax4 = plt.subplots(figsize=(10, 5.5))

bar_width = 0.18
n_groups = len(alpha_vals)
x_base = np.arange(n_groups)
cmap4 = plt.cm.Set2

for i, ar in enumerate(AR_ORDER):
    ys = []
    for a in alpha_vals:
        row = next(
            r for r in exp2
            if r["gamma"] == GAMMA_FILTER and abs(r["alpha"] - a) < 1e-6 and abs(r["acceptance_rate"] - ar) < 1e-6
        )
        ys.append(row["speedup"])
    offset = (i - len(AR_ORDER) / 2 + 0.5) * bar_width
    bars = ax4.bar(x_base + offset, ys, bar_width, label=f"AR={ar:.2f}", color=cmap4(i / (len(AR_ORDER) - 1)), edgecolor="black", linewidth=0.4)

ax4.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0)
ax4.set_xlabel("alpha (draft verification cost ratio)")
ax4.set_ylabel("Speedup")
ax4.set_title("Speedup vs alpha (gamma=5) Grouped by Acceptance Rate")
ax4.set_xticks(x_base)
ax4.set_xticklabels([f"{a:.2f}" for a in alpha_vals], rotation=45, ha="right")
ax4.legend(loc="upper right", title="Acceptance Rate")
ax4.grid(axis="y", alpha=0.3)
fig4.tight_layout()
fig4.savefig(OUT_DIR + "fig4_alpha_speedup.png", dpi=150)
plt.close(fig4)
print("[OK] fig4_alpha_speedup.png")

print("\nAll 4 figures saved to", OUT_DIR)
