"""
FlexAttention → CATLASS Integration Design: Chart Generation

Generates analysis charts for the FlexAttention CATLASS integration design report.
All labels in English to avoid CJK font issues on headless servers.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os
import json

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Color palette
COLORS = {
    'blue': '#2563EB',
    'green': '#16A34A',
    'orange': '#EA580C',
    'red': '#DC2626',
    'purple': '#9333EA',
    'teal': '#0D9488',
    'gray': '#6B7280',
    'light_blue': '#DBEAFE',
    'light_green': '#DCFCE7',
    'light_orange': '#FFF7ED',
    'light_purple': '#F3E8FF',
    'light_gray': '#F3F4F6',
}


def fig1_integration_architecture():
    """Figure 1: Integration Architecture Overview - Python API to CATLASS kernel"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('FlexAttention → CATLASS Integration Architecture', fontsize=14, fontweight='bold', pad=20)

    layers = [
        (5.0, 9.0, 'User API Layer', 'flex_attention(score_mod, mask_mod)\ncreate_block_mask()', COLORS['light_blue'], COLORS['blue']),
        (5.0, 7.5, 'Dynamo + HOP Tracing', 'FlexAttentionHigherOrderVariable\nTraces score_mod/mask_mod into graph', COLORS['light_green'], COLORS['green']),
        (5.0, 6.0, 'Inductor Lowering', 'register_lowering(flex_attention)\nPattern matching + backend dispatch', COLORS['light_orange'], COLORS['orange']),
        (2.0, 4.2, 'CATLASS Path\n(Proposed)', 'CATLASSFATemplate\nFAInferKernel C++\nAscendC codegen', COLORS['light_purple'], COLORS['purple']),
        (8.0, 4.2, 'Triton Path\n(Existing)', 'NPUTritonTemplate\nTriton kernel\nJIT compilation', COLORS['light_blue'], COLORS['blue']),
        (2.0, 2.2, 'CATLASS Runtime', 'AscendC AICore\nFAInferKernel\nOnline Softmax + RescaleO', '#FCE7F3', '#DB2777'),
        (8.0, 2.2, 'Triton Runtime', 'NPU Triton VM\nload_checked_2d\nVector/Cube ops', COLORS['light_gray'], COLORS['gray']),
    ]

    for (x, y, title, desc, bg, border) in layers:
        box = FancyBboxPatch((x - 2.2, y - 0.6), 4.4, 1.2,
                             boxstyle="round,pad=0.1", facecolor=bg,
                             edgecolor=border, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y + 0.15, title, ha='center', va='center',
                fontsize=10, fontweight='bold', color=border)
        ax.text(x, y - 0.25, desc, ha='center', va='center',
                fontsize=7, color='#374151', family='monospace')

    # Arrows between layers
    arrow_props = dict(arrowstyle='->', color=COLORS['gray'], lw=1.5, mutation_scale=15)
    ax.annotate('', xy=(5.0, 8.4), xytext=(5.0, 8.9), arrowprops=arrow_props)
    ax.annotate('', xy=(5.0, 6.9), xytext=(5.0, 7.4), arrowprops=arrow_props)
    ax.annotate('', xy=(5.0, 5.4), xytext=(5.0, 5.9), arrowprops=arrow_props)
    # Split to two paths
    ax.annotate('', xy=(2.0, 4.9), xytext=(4.0, 5.4), arrowprops=arrow_props)
    ax.annotate('', xy=(8.0, 4.9), xytext=(6.0, 5.4), arrowprops=arrow_props)
    # To runtime
    ax.annotate('', xy=(2.0, 2.9), xytext=(2.0, 3.5), arrowprops=arrow_props)
    ax.annotate('', xy=(8.0, 2.9), xytext=(8.0, 3.5), arrowprops=arrow_props)

    # autotune_select_algorithm box
    ax.text(5.0, 5.5, 'autotune_select_algorithm', ha='center', va='center',
            fontsize=7, fontstyle='italic', color=COLORS['orange'])

    # GPU label
    ax.text(7.0, 1.2, 'Ascend NPU (910B3 / Atlas A2)', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLORS['red'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEE2E2', edgecolor=COLORS['red']))

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'integration_architecture.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Generated: integration_architecture.png")


def fig2_pattern_matching():
    """Figure 2: Pattern matching for score_mod/mask_mod → CATLASS mask types"""
    fig, ax = plt.subplots(1, 1, figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('score_mod/mask_mod Pattern Matching → CATLASS MaskType', fontsize=14, fontweight='bold', pad=20)

    # Input patterns (left)
    patterns = [
        (2.0, 7.0, 'causal_mask\n(mask_mod)', COLORS['light_blue'], COLORS['blue']),
        (2.0, 5.8, 'sliding_window\n(mask_mod)', COLORS['light_blue'], COLORS['blue']),
        (2.0, 4.6, 'alibi_score\n(score_mod)', COLORS['light_green'], COLORS['green']),
        (2.0, 3.4, 'softcap_score\n(score_mod)', COLORS['light_green'], COLORS['green']),
        (2.0, 2.2, 'prefix_lm\n(mask_mod)', COLORS['light_blue'], COLORS['blue']),
        (2.0, 1.0, 'custom\n(score/mask_mod)', COLORS['light_orange'], COLORS['orange']),
    ]

    for (x, y, label, bg, border) in patterns:
        box = FancyBboxPatch((x - 1.8, y - 0.4), 3.6, 0.8,
                             boxstyle="round,pad=0.05", facecolor=bg,
                             edgecolor=border, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold', color=border)

    # Pattern matcher (center)
    box = FancyBboxPatch((5.3, 2.5), 2.4, 5.0,
                         boxstyle="round,pad=0.1", facecolor='#FFF7ED',
                         edgecolor=COLORS['orange'], linewidth=2)
    ax.add_patch(box)
    ax.text(6.5, 7.0, 'Pattern', ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['orange'])
    ax.text(6.5, 6.5, 'Matcher', ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['orange'])

    # Pattern matching rules
    rules = [
        '1. Identify mask_mod type',
        '2. Identify score_mod type',
        '3. Check combinability',
        '4. Map to maskType',
        '5. Set scaleValue',
        '6. Generate tiling data',
    ]
    for i, rule in enumerate(rules):
        ax.text(6.5, 5.5 - i * 0.55, rule, ha='center', va='center', fontsize=8, color='#374151')

    # CATLASS outputs (right)
    outputs = [
        (10.5, 6.5, 'CATLASS: MASK_CAUSUAL\nmaskType=2, no extra mask buf', '#FCE7F3', '#DB2777'),
        (10.5, 5.0, 'CATLASS: MASK_SPEC\nmaskType=1, custom mask buf', '#FCE7F3', '#DB2777'),
        (10.5, 3.5, 'CATLASS: NO_MASK\nmaskType=0, modify scaleValue', '#DCFCE7', COLORS['green']),
        (10.5, 2.0, 'FALLBACK to Triton\nNPUTritonTemplate', COLORS['light_gray'], COLORS['gray']),
    ]

    for (x, y, label, bg, border) in outputs:
        box = FancyBboxPatch((x - 2.0, y - 0.4), 4.0, 0.8,
                             boxstyle="round,pad=0.05", facecolor=bg,
                             edgecolor=border, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold', color=border)

    # Arrows from patterns to matcher
    arrow_props = dict(arrowstyle='->', color=COLORS['gray'], lw=1.0, mutation_scale=12)
    for (x, y, _, _, _) in patterns:
        ax.annotate('', xy=(5.3, y), xytext=(3.8, y), arrowprops=arrow_props)

    # Arrows from matcher to outputs
    ax.annotate('', xy=(8.5, 6.5), xytext=(7.7, 6.5), arrowprops=arrow_props)
    ax.annotate('', xy=(8.5, 5.0), xytext=(7.7, 5.5), arrowprops=arrow_props)
    ax.annotate('', xy=(8.5, 3.5), xytext=(7.7, 4.0), arrowprops=arrow_props)
    ax.annotate('', xy=(8.5, 2.0), xytext=(7.7, 2.8), arrowprops=arrow_props)

    # Legend
    legend_items = [
        mpatches.Patch(facecolor=COLORS['light_blue'], edgecolor=COLORS['blue'], label='mask_mod (boolean)'),
        mpatches.Patch(facecolor=COLORS['light_green'], edgecolor=COLORS['green'], label='score_mod (float)'),
        mpatches.Patch(facecolor=COLORS['light_orange'], edgecolor=COLORS['orange'], label='Custom (unsupported)'),
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'pattern_matching.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Generated: pattern_matching.png")


def fig3_blockmask_translation():
    """Figure 3: BlockMask BCSR → CATLASS paged KV mapping"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: BlockMask BCSR structure
    ax1 = axes[0]
    ax1.set_title('PyTorch BlockMask (BCSR Format)', fontsize=11, fontweight='bold')

    # Create a sparse attention pattern
    np.random.seed(42)
    n = 16
    mask = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j <= i:  # causal
                mask[i][j] = 1
    # Add some sparsity
    mask[12][3] = 0
    mask[13][4] = 0
    mask[14][5] = 0
    mask[15][6] = 0

    # Show block-level view (4x4 blocks)
    block_mask = np.zeros((4, 4))
    for bi in range(4):
        for bj in range(4):
            block = mask[bi*4:(bi+1)*4, bj*4:(bj+1)*4]
            if block.any():
                block_mask[bi][bj] = 1

    cmap = plt.cm.colors.ListedColormap(['#FEE2E2', '#DCFCE7'])
    ax1.imshow(block_mask, cmap=cmap, vmin=0, vmax=1, aspect='equal')

    # Annotate blocks
    for i in range(4):
        for j in range(4):
            if block_mask[i][j]:
                ax1.text(j, i, '1', ha='center', va='center', fontsize=14, fontweight='bold', color='green')
            else:
                ax1.text(j, i, '0', ha='center', va='center', fontsize=14, fontweight='bold', color='red')

    ax1.set_xlabel('KV Block Index', fontsize=10)
    ax1.set_ylabel('Q Block Index', fontsize=10)
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))

    # Add BCSR data annotation
    ax1.text(0.5, -0.3, 'kv_num_blocks: [4,4,4,3]\nkv_indices: [[0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,7]]',
             ha='center', va='top', fontsize=7, family='monospace', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor=COLORS['light_blue'], alpha=0.8))

    # Right: CATLASS paged KV
    ax2 = axes[1]
    ax2.set_title('CATLASS Paged KV + mask', fontsize=11, fontweight='bold')

    # Show paged KV layout
    blocks = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    page_table = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    # Draw KV blocks
    for i, b in enumerate(blocks):
        color = COLORS['light_green'] if i <= 4 else COLORS['light_gray']
        rect = FancyBboxPatch((i * 1.1, 0.5), 1.0, 0.8,
                              boxstyle="round,pad=0.05", facecolor=color,
                              edgecolor=COLORS['green'], linewidth=1.5)
        ax2.add_patch(rect)
        ax2.text(i * 1.1 + 0.5, 0.9, b, ha='center', va='center', fontsize=8, fontweight='bold')
        ax2.text(i * 1.1 + 0.5, 0.6, f'K{i}', ha='center', va='center', fontsize=6, color='gray')

    ax2.text(4.4, 1.7, 'Paged KV Cache (blockTables)', ha='center', va='center',
             fontsize=9, fontweight='bold', color=COLORS['green'])

    # Mask application
    ax2.text(4.4, 0.0, 'maskType: MASK_CAUSUAL (2)\nnoSkipKvS / noMaskKvS computed\nfrom qSeqlen vs kvSeqlen',
             ha='center', va='center', fontsize=8, color='#374151',
             bbox=dict(boxstyle='round', facecolor=COLORS['light_orange'], alpha=0.8))

    # Kernel params
    params_text = (
        'FAIKernelParams:\n'
        '  q, k, v, mask, blockTables\n'
        '  actualQseqlen, actualKvseqlen\n'
        '  o, s, p, oTemp, oUpdate, tiling'
    )
    ax2.text(4.4, -0.8, params_text, ha='center', va='center', fontsize=7,
             family='monospace', color='#374151',
             bbox=dict(boxstyle='round', facecolor='#F3E8FF', alpha=0.8))

    ax2.set_xlim(-0.5, 9.3)
    ax2.set_ylim(-1.5, 2.2)
    ax2.axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'blockmask_translation.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Generated: blockmask_translation.png")


def fig4_kernel_code_generation():
    """Figure 4: CATLASS FA kernel code generation pipeline"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('CATLASS FA Kernel Code Generation Pipeline', fontsize=14, fontweight='bold', pad=20)

    stages = [
        (2.0, 5.5, 'CATLASSFATemplate\n(Python)', 'Template class with\nFA C++ template string', COLORS['light_blue'], COLORS['blue']),
        (5.5, 5.5, 'render()\n(Jinja2-like)', 'Substitute {{placeholders}}\nwith kernel params', COLORS['light_green'], COLORS['green']),
        (9.0, 5.5, 'CATLASSBenchmark\nRequest', 'C++ source + tensor meta\nfor autotuning', COLORS['light_orange'], COLORS['orange']),
        (12.0, 5.5, 'autotune\nselect_algorithm', 'Benchmark CATLASS\nvs Triton on NPU', COLORS['light_purple'], COLORS['purple']),
        (2.0, 3.0, 'CATLASSKernel\nCode', 'FAInferKernel<...>\nAscendC AIC operator()', '#FCE7F3', '#DB2777'),
        (5.5, 3.0, 'Tiling Data\n(FATilingData)', 'numHeads, embed, blockSize\nmaskType, scaleValue', COLORS['light_green'], COLORS['green']),
        (9.0, 3.0, 'AscendC\nCompiler', 'AICore binary\nCube+Vector ops', '#FEE2E2', COLORS['red']),
        (12.0, 3.0, 'NPU\nExecution', 'aclrtLaunch kernel\nstream execution', COLORS['light_gray'], COLORS['gray']),
    ]

    for (x, y, title, desc, bg, border) in stages:
        box = FancyBboxPatch((x - 1.5, y - 0.6), 3.0, 1.2,
                             boxstyle="round,pad=0.08", facecolor=bg,
                             edgecolor=border, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y + 0.2, title, ha='center', va='center', fontsize=9, fontweight='bold', color=border)
        ax.text(x, y - 0.2, desc, ha='center', va='center', fontsize=7, color='#374151', family='monospace')

    # Arrows
    arrow_props = dict(arrowstyle='->', color=COLORS['gray'], lw=1.5, mutation_scale=15)
    # Top row
    for i in range(3):
        ax.annotate('', xy=(stages[i+1][0] - 1.5, stages[i+1][1]), xytext=(stages[i][0] + 1.5, stages[i][1]),
                     arrowprops=arrow_props)
    # Down from stage 4
    ax.annotate('', xy=(12.0, 3.7), xytext=(12.0, 4.8), arrowprops=arrow_props)
    # Bottom row
    for i in range(4, 7):
        ax.annotate('', xy=(stages[i+1][0] - 1.5, stages[i+1][1]), xytext=(stages[i][0] + 1.5, stages[i][1]),
                     arrowprops=arrow_props)
    # Down from top to bottom
    ax.annotate('', xy=(2.0, 3.7), xytext=(2.0, 4.8), arrowprops=arrow_props)
    ax.annotate('', xy=(5.5, 3.7), xytext=(5.5, 4.8), arrowprops=arrow_props)
    ax.annotate('', xy=(9.0, 3.7), xytext=(9.0, 4.8), arrowprops=arrow_props)

    # Template snippet
    template_text = (
        'CATLASS_FA_TEMPLATE = r"""\n'
        '{{header}} {{globals}}\n'
        'extern "C" PT_EXPORT kernel_call {\n'
        '  using FAInferKernel = FAInferKernel<\n'
        '    BlockMmadQK, BlockMmadPV,\n'
        '    EpilogueOnlineSoftmax,\n'
        '    EpilogueRescaleO,\n'
        '    {{paged_cache_flag}}>;\n'
        '  FAInferKernel kernel;\n'
        '  kernel(params);\n'
        '}\n'
        '"""'
    )
    ax.text(7.0, 1.2, template_text, ha='center', va='center', fontsize=6,
            family='monospace', color='#374151',
            bbox=dict(boxstyle='round', facecolor=COLORS['light_blue'], alpha=0.6))

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'kernel_codegen_pipeline.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Generated: kernel_codegen_pipeline.png")


def fig5_path_comparison():
    """Figure 5: Triton vs CATLASS path feature comparison"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.axis('off')
    ax.set_title('Triton Path vs CATLASS Path: Feature Comparison', fontsize=14, fontweight='bold', pad=20)

    features = [
        'Causal Mask',
        'Sliding Window',
        'Prefix LM',
        'ALiBi (score_mod)',
        'Softcapping (score_mod)',
        'Block Sparse (BlockMask)',
        'Paged KV Cache',
        'Online Softmax',
        'GQA/MQA Support',
        'Varlen (variable seq len)',
        'FP16/BF16',
        'Dynamic score_mod',
        'Autotuning',
        'Training (backward)',
    ]

    triton_support = [1, 1, 1, 1, 1, 1, 0.5, 0, 0.5, 1, 1, 1, 0.5, 0.5]
    catlass_current = [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    catlass_proposed = [1, 1, 1, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 0, 1, 0]

    x = np.arange(len(features))
    width = 0.25

    def support_color(val):
        if val >= 1:
            return COLORS['green']
        elif val >= 0.5:
            return COLORS['orange']
        else:
            return COLORS['red']

    # Draw bars
    for i, (t, c, p) in enumerate(zip(triton_support, catlass_current, catlass_proposed)):
        ax.barh(i + 0.25, t, height=0.2, color=support_color(t), alpha=0.8, edgecolor='white')
        ax.barh(i, c, height=0.2, color=support_color(c), alpha=0.8, edgecolor='white')
        ax.barh(i - 0.25, p, height=0.2, color=support_color(p), alpha=0.8, edgecolor='white')

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(['No', 'Partial', 'Full'], fontsize=9)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['green'], label='Triton (existing)'),
        mpatches.Patch(facecolor=COLORS['orange'], label='CATLASS (current)'),
        mpatches.Patch(facecolor=COLORS['red'], label='CATLASS (proposed)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'path_feature_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Generated: path_feature_comparison.png")


def fig6_implementation_roadmap():
    """Figure 6: Implementation roadmap / phases"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Implementation Roadmap: 4 Phases', fontsize=14, fontweight='bold', pad=20)

    phases = [
        {
            'x': 1.75, 'y': 5.5, 'title': 'Phase 1: Foundation',
            'items': [
                'CATLASSFATemplate class',
                'CATLASS_FA_TEMPLATE C++ string',
                'register_lowering() hook',
                'FATilingData generation',
            ],
            'color': COLORS['blue'], 'bg': COLORS['light_blue']
        },
        {
            'x': 5.25, 'y': 5.5, 'title': 'Phase 2: Causal',
            'items': [
                'Causal mask pattern detection',
                'maskType=MASK_CAUSUAL mapping',
                'Basic forward kernel integration',
                'Correctness validation',
            ],
            'color': COLORS['green'], 'bg': COLORS['light_green']
        },
        {
            'x': 8.75, 'y': 5.5, 'title': 'Phase 3: Masks',
            'items': [
                'Sliding window / Prefix LM',
                'MASK_SPEC + mask buffer',
                'BlockMask → mask tensor',
                'Autotuning (910B3 configs)',
            ],
            'color': COLORS['orange'], 'bg': COLORS['light_orange']
        },
        {
            'x': 12.25, 'y': 5.5, 'title': 'Phase 4: Advanced',
            'items': [
                'score_mod preprocessing',
                'ALiBi scaleValue injection',
                'Paged KV + FlexAttention',
                'Performance optimization',
            ],
            'color': COLORS['purple'], 'bg': COLORS['light_purple']
        },
    ]

    for phase in phases:
        x, y = phase['x'], phase['y']
        box = FancyBboxPatch((x - 1.5, y - 2.0), 3.0, 4.0,
                             boxstyle="round,pad=0.1", facecolor=phase['bg'],
                             edgecolor=phase['color'], linewidth=2)
        ax.add_patch(box)
        ax.text(x, y + 1.5, phase['title'], ha='center', va='center',
                fontsize=10, fontweight='bold', color=phase['color'])
        for i, item in enumerate(phase['items']):
            ax.text(x, y + 0.6 - i * 0.5, f'• {item}', ha='center', va='center',
                    fontsize=8, color='#374151')

    # Phase arrows
    arrow_props = dict(arrowstyle='->', color=COLORS['gray'], lw=2, mutation_scale=20)
    for i in range(3):
        ax.annotate('', xy=(phases[i+1]['x'] - 1.5, phases[i+1]['y']),
                     xytext=(phases[i]['x'] + 1.5, phases[i]['y']),
                     arrowprops=arrow_props)

    # Timeline
    ax.text(7.0, 0.8, 'Estimated Timeline: Phase 1 (2w) → Phase 2 (2w) → Phase 3 (3w) → Phase 4 (3w)',
            ha='center', va='center', fontsize=10, color=COLORS['gray'],
            bbox=dict(boxstyle='round', facecolor=COLORS['light_gray'], alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'implementation_roadmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Generated: implementation_roadmap.png")


if __name__ == '__main__':
    print("Generating FlexAttention → CATLASS integration design charts...")
    fig1_integration_architecture()
    fig2_pattern_matching()
    fig3_blockmask_translation()
    fig4_kernel_code_generation()
    fig5_path_comparison()
    fig6_implementation_roadmap()
    print(f"\nAll charts saved to: {FIGURES_DIR}")
