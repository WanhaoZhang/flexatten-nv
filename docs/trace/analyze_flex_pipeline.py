#!/usr/bin/env python3
"""FlexAttention GPU Pipeline Static Analysis & Visualization"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# ============================================================
# Chart 1: Pipeline Overview
# ============================================================
def chart_pipeline_overview():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    layers = [
        ("Layer 1: User API", "flex_attention() / create_block_mask()", '#2ecc71', 9.0, 3, 8),
        ("Layer 2: Dynamo Trace", "FlexAttentionHigherOrderVariable\ntrace score_mod / mask_mod", '#3498db', 7.5, 3, 8),
        ("Layer 3: HOP Dispatch", "FlexAttentionHOP\neager / autograd / proxy", '#9b59b6', 6.0, 3, 8),
        ("Layer 4: Inductor Lowering", "register_lowering\nbranch selection", '#e74c3c', 4.5, 3, 8),
        ("Layer 5a: Triton Path", "TritonTemplate\nflex_attention kernel", '#e67e22', 3.0, 1, 5.5),
        ("Layer 5b: CuteDSL Path", "CuteDSLTemplate\nflash_attn.cute + CUTLASS", '#e67e22', 3.0, 7.5, 5.5),
        ("Layer 6: GPU Execution", "Triton -> PTX | CuteDSL -> CUTLASS -> PTX", '#1abc9c', 1.5, 3, 8),
    ]

    for name, desc, color, y, x_start, width in layers:
        box = FancyBboxPatch((x_start, y - 0.5), width, 1.0, boxstyle="round,pad=0.1",
                             facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(box)
        cx = x_start + width / 2
        ax.text(cx, y, f"{name}\n{desc}", ha='center', va='center', fontsize=9, fontweight='bold')

    arrow_kw = dict(arrowstyle='->', color='#34495e', lw=2, mutation_scale=15)
    ax.annotate('', xy=(7, 8.0), xytext=(7, 8.5), arrowprops=arrow_kw)
    ax.annotate('', xy=(7, 6.5), xytext=(7, 7.0), arrowprops=arrow_kw)
    ax.annotate('', xy=(7, 5.0), xytext=(7, 5.5), arrowprops=arrow_kw)
    ax.annotate('', xy=(3.75, 3.5), xytext=(5.5, 4.0), arrowprops=arrow_kw)
    ax.annotate('', xy=(10.25, 3.5), xytext=(8.5, 4.0), arrowprops=arrow_kw)
    ax.annotate('', xy=(5.5, 1.5), xytext=(3.75, 2.5), arrowprops=arrow_kw)
    ax.annotate('', xy=(8.5, 1.5), xytext=(10.25, 2.5), arrowprops=arrow_kw)

    ax.text(5.5, 4.3, 'Default path\n(PT 2.5+)', ha='center', fontsize=8, color='#27ae60', fontstyle='italic')
    ax.text(8.5, 4.3, 'BACKEND=FLASH\n(PT 2.7+ / nightly)', ha='center', fontsize=8, color='#c0392b', fontstyle='italic')

    ax.set_title('FlexAttention Compilation Pipeline: Python API -> GPU Kernel', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'pipeline_overview.png'), bbox_inches='tight')
    plt.close()
    print("Chart 1: pipeline_overview.png done")


# ============================================================
# Chart 2: BlockMask BCSR structure
# ============================================================
def chart_blockmask_structure():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    mask = np.tril(np.ones((8, 8)))
    ax.imshow(mask, cmap='RdYlGn', aspect='equal')
    ax.set_title('(a) Dense Causal Mask\n8x8 blocks', fontsize=11, fontweight='bold')
    ax.set_xlabel('KV Block Index')
    ax.set_ylabel('Q Block Index')
    for i in range(9):
        ax.axhline(i - 0.5, color='gray', lw=0.5)
        ax.axvline(i - 0.5, color='gray', lw=0.5)

    ax = axes[1]
    colors_map = np.zeros((8, 8, 3))
    for i in range(8):
        for j in range(8):
            if j < i:
                colors_map[i, j] = [0.18, 0.8, 0.44]  # green = full
            elif j == i:
                colors_map[i, j] = [0.93, 0.69, 0.13]  # yellow = partial
    ax.imshow(colors_map, aspect='equal')
    ax.set_title('(b) Block-Sparse Format\nFull + Partial Separation', fontsize=11, fontweight='bold')
    ax.set_xlabel('KV Block Index')
    ax.set_ylabel('Q Block Index')
    for i in range(9):
        ax.axhline(i - 0.5, color='gray', lw=0.5)
        ax.axvline(i - 0.5, color='gray', lw=0.5)
    green_p = mpatches.Patch(color=[0.18, 0.8, 0.44], label='Full Block (skip mask_mod)')
    yellow_p = mpatches.Patch(color=[0.93, 0.69, 0.13], label='Partial Block (needs mask_mod)')
    ax.legend(handles=[green_p, yellow_p], fontsize=8, loc='upper left')

    ax = axes[2]
    ax.axis('off')
    table_data = [
        ['kv_num_blocks', '[Q_BLOCKS]', 'Non-zero blocks per row'],
        ['kv_indices', '[Q_BLOCKS, MAX]', 'Block column indices (ordered)'],
        ['full_kv_num_blocks', '[Q_BLOCKS]', 'Count of full blocks'],
        ['full_kv_indices', '[Q_BLOCKS, MAX]', 'Full block indices'],
    ]
    table = ax.table(cellText=table_data,
                     colLabels=['Tensor', 'Shape', 'Meaning'],
                     cellLoc='center', loc='center',
                     colWidths=[0.35, 0.25, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#3498db')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#ecf0f1')
    ax.set_title('(c) BlockMask Storage Format\n(BCSR-like)', fontsize=11, fontweight='bold')

    fig.suptitle('BlockMask Data Structure: Dense -> Block-Sparse -> BCSR', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'blockmask_structure.png'), bbox_inches='tight')
    plt.close()
    print("Chart 2: blockmask_structure.png done")


# ============================================================
# Chart 3: Triton kernel flow
# ============================================================
def chart_triton_kernel_flow():
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    steps = [
        (1, 7.0, "Grid Launch\n(ceil_div(S, BLOCK_M), B*H, 1)", '#3498db'),
        (1, 5.8, "Load Q Block\n[BLOCK_M, D]", '#2ecc71'),
        (1, 4.6, "Iterate Sparse KV Blocks\n(kv_num_blocks, kv_indices)", '#e74c3c'),
        (4.5, 4.6, "Load K/V Blocks\n[D, BLOCK_N]", '#9b59b6'),
        (4.5, 3.4, "QK^T + score_mod\n+ mask_mod (partial)", '#e67e22'),
        (4.5, 2.2, "Online Softmax\nupdate m_i, l_i, acc", '#1abc9c'),
        (8, 5.8, "Full Block Opt.\nSkip mask_mod\n~15% speedup", '#27ae60'),
        (8, 4.6, "Write Output\n[BLOCK_M, D]", '#34495e'),
    ]

    for x, y, text, color in steps:
        box = FancyBboxPatch((x - 0.5, y - 0.45), 3.5, 0.9, boxstyle="round,pad=0.1",
                             facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 1.25, y, text, ha='center', va='center', fontsize=8.5, fontweight='bold')

    arrow_kw = dict(arrowstyle='->', color='#2c3e50', lw=1.5, mutation_scale=12)
    ax.annotate('', xy=(2.5, 6.25), xytext=(2.5, 6.75), arrowprops=arrow_kw)
    ax.annotate('', xy=(2.5, 5.05), xytext=(2.5, 5.55), arrowprops=arrow_kw)
    ax.annotate('', xy=(2.5, 5.35), xytext=(2.5, 4.15), arrowprops=arrow_kw)
    ax.text(0.3, 4.7, 'Sparse\nBlock\nLoop', fontsize=8, color='#e74c3c', fontstyle='italic')
    ax.annotate('', xy=(5.75, 5.05), xytext=(5.75, 4.15), arrowprops=arrow_kw)
    ax.annotate('', xy=(5.75, 3.85), xytext=(5.75, 2.95), arrowprops=arrow_kw)
    ax.annotate('', xy=(8, 5.35), xytext=(4, 5.8), arrowprops=arrow_kw)
    ax.text(5.5, 5.9, 'full block -> skip mask', fontsize=7, color='#27ae60', fontstyle='italic')
    ax.annotate('', xy=(9.25, 5.05), xytext=(9.25, 4.15), arrowprops=arrow_kw)
    ax.annotate('', xy=(6.5, 3.4), xytext=(7.5, 4.6), arrowprops=arrow_kw)

    ax.set_title('FlexAttention Triton Forward Kernel Execution Flow', fontsize=13, fontweight='bold', pad=15)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'triton_kernel_flow.png'), bbox_inches='tight')
    plt.close()
    print("Chart 3: triton_kernel_flow.png done")


# ============================================================
# Chart 4: Triton vs CuteDSL comparison
# ============================================================
def chart_path_comparison():
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    triton_steps = [
        (1.5, 5.0, "Python score_mod / mask_mod", '#3498db'),
        (1.5, 3.8, "Dynamo -> FX Graph", '#3498db'),
        (1.5, 2.6, "Inductor -> TritonTemplate", '#3498db'),
        (1.5, 1.4, "Triton JIT -> PTX -> GPU", '#3498db'),
    ]
    cute_steps = [
        (8.5, 5.0, "Python score_mod / mask_mod", '#e74c3c'),
        (8.5, 3.8, "Dynamo -> FX Graph", '#e74c3c'),
        (8.5, 2.6, "Inductor -> CuteDSLTemplate", '#e74c3c'),
        (8.5, 1.4, "flash_attn.cute -> CUTLASS -> GPU", '#e74c3c'),
    ]

    for steps_list in [triton_steps, cute_steps]:
        for x, y, text, color in steps_list:
            box = FancyBboxPatch((x - 1.3, y - 0.4), 3.6, 0.8, boxstyle="round,pad=0.1",
                                 facecolor=color, alpha=0.2, edgecolor=color, linewidth=1.5)
            ax.add_patch(box)
            ax.text(x + 0.5, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    arrow_kw = dict(arrowstyle='->', lw=1.5, mutation_scale=12)
    for col_x, color in [(2.0, '#3498db'), (9.0, '#e74c3c')]:
        for y_from, y_to in [(4.6, 4.2), (3.4, 3.0), (2.2, 1.8)]:
            ax.annotate('', xy=(col_x, y_to), xytext=(col_x, y_from),
                       arrowprops={**arrow_kw, 'color': color})

    ax.text(2.0, 5.7, 'Triton Path', ha='center', fontsize=12, fontweight='bold', color='#3498db')
    ax.text(9.0, 5.7, 'CuteDSL Path', ha='center', fontsize=12, fontweight='bold', color='#e74c3c')

    # Middle comparison table
    comparisons = [
        ("PT Version", "2.5+", "2.7+/nightly"),
        ("Compiler", "Triton", "CuteDSL + CUTLASS"),
        ("GPU Req.", "sm75+", "sm90+ (Hopper)"),
        ("L4 Usable", "Yes", "No"),
        ("Kernel Gen", "Python->Triton->PTX", "Python->CuTe->C++->PTX"),
        ("Optim. Level", "Triton autotune", "CUTLASS 3.x tile-level"),
    ]
    for i, (metric, triton_val, cute_val) in enumerate(comparisons):
        y = 5.0 - i * 0.7
        ax.text(5.5, y, metric, ha='right', va='center', fontsize=8, fontweight='bold')
        ax.text(6.0, y, '|', ha='center', va='center', fontsize=8, color='gray')
        ax.text(6.5, y, f'{triton_val} / {cute_val}', ha='left', va='center', fontsize=8)

    ax.set_title('FlexAttention: Two GPU Compilation Paths Compared', fontsize=13, fontweight='bold', pad=15)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'path_comparison.png'), bbox_inches='tight')
    plt.close()
    print("Chart 4: path_comparison.png done")


# ============================================================
# Chart 5: Autotune configs
# ============================================================
def chart_autotune_configs():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    configs = {
        'H100\n(sm90)': (128, 128, 8, 3),
        'A100\n(sm80)': (128, 64, 8, 3),
        'ROCm\n(MI300)': (128, 64, 8, 1),
        'L4 Default\n(sm89)': (128, 64, 4, 3),
    }
    labels = list(configs.keys())
    bm = [v[0] for v in configs.values()]
    bn = [v[1] for v in configs.values()]
    nw = [v[2] for v in configs.values()]
    ns = [v[3] for v in configs.values()]

    x = np.arange(len(labels))
    w = 0.2
    ax = axes[0]
    bars1 = ax.bar(x - 1.5*w, bm, w, label='BLOCK_M', color='#3498db')
    bars2 = ax.bar(x - 0.5*w, bn, w, label='BLOCK_N', color='#2ecc71')
    bars3 = ax.bar(x + 0.5*w, nw, w, label='num_warps', color='#e74c3c')
    bars4 = ax.bar(x + 1.5*w, ns, w, label='num_stages', color='#9b59b6')
    ax.set_xlabel('GPU Architecture')
    ax.set_ylabel('Value')
    ax.set_title('Forward Kernel Autotune Config\n(FP16, head_dim=128)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{int(h)}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 2), textcoords='offset points', ha='center', fontsize=7)

    ax = axes[1]
    dtypes = ['FP32\nD=64', 'FP32\nD=128', 'FP32\nD=256', 'BF16\nD=64', 'BF16\nD=128', 'BF16\nD=256',
              'FP16\nD=64', 'FP16\nD=128', 'FP16\nD=256']
    all_configs = [
        (128,32,4,3), (32,64,4,3), (32,32,4,3),  # FP32
        (128,128,4,3), (128,64,8,3), (64,32,4,3),  # BF16
        (128,128,4,3), (128,128,8,3), (64,32,4,3),  # FP16
    ]
    bm2 = [v[0] for v in all_configs]
    bn2 = [v[1] for v in all_configs]
    x2 = np.arange(len(dtypes))
    ax.bar(x2 - 0.2, bm2, 0.4, label='BLOCK_M', color='#3498db')
    ax.bar(x2 + 0.2, bn2, 0.4, label='BLOCK_N', color='#2ecc71')
    ax.set_xlabel('dtype x head_dim')
    ax.set_ylabel('Value')
    ax.set_title('H100 Forward Config by dtype & head_dim', fontweight='bold')
    ax.set_xticks(x2)
    ax.set_xticklabels(dtypes, fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'autotune_configs.png'), bbox_inches='tight')
    plt.close()
    print("Chart 5: autotune_configs.png done")


# ============================================================
# Chart 6: Block size impact
# ============================================================
def chart_block_size_impact():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    block_sizes = [16, 32, 64, 128, 256, 512]
    seq_len = 2048
    sparsities = []
    for bs in block_sizes:
        n = seq_len // bs
        total = n * n
        non_zero = sum(n - i for i in range(n))
        sparsities.append((1 - non_zero / total) * 100)

    ax = axes[0]
    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e74c3c', '#c0392b', '#8e44ad']
    bars = ax.bar([str(bs) for bs in block_sizes], sparsities, color=colors)
    for bar, sp in zip(bars, sparsities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{sp:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('BLOCK_SIZE')
    ax.set_ylabel('Sparsity (%)')
    ax.set_title(f'(a) Causal Mask Sparsity vs BLOCK_SIZE\n(S={seq_len})', fontweight='bold')
    ax.set_ylim(0, 60)
    ax.grid(axis='y', alpha=0.3)

    seq_lens = [512, 1024, 2048, 4096, 8192]
    bs_16, bs_128 = [], []
    for sl in seq_lens:
        n = sl // 16
        bs_16.append((1 - sum(n - i for i in range(n)) / (n * n)) * 100)
        n = sl // 128
        bs_128.append((1 - sum(n - i for i in range(n)) / (n * n)) * 100)

    ax = axes[1]
    x = np.arange(len(seq_lens))
    ax.bar(x - 0.2, bs_16, 0.4, label='BLOCK_SIZE=16', color='#2ecc71')
    ax.bar(x + 0.2, bs_128, 0.4, label='BLOCK_SIZE=128', color='#e74c3c')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Compute Savings (%)')
    ax.set_title('(b) Causal Mask Compute Savings\n(BS=16 vs BS=128)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(sl) for sl in seq_lens])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('BLOCK_SIZE Impact on FlexAttention Performance', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'block_size_impact.png'), bbox_inches='tight')
    plt.close()
    print("Chart 6: block_size_impact.png done")


if __name__ == '__main__':
    print("Generating FlexAttention Pipeline Analysis Charts...")
    print(f"Output directory: {FIG_DIR}")
    chart_pipeline_overview()
    chart_blockmask_structure()
    chart_triton_kernel_flow()
    chart_path_comparison()
    chart_autotune_configs()
    chart_block_size_impact()
    print("\nAll 6 charts generated!")
