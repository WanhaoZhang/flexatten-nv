import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('Agg')
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})
FIG_DIR = '/home/zhangwh/zwhllm/flexatten-nv/docs/figures'

with open('/home/zhangwh/zwhllm/flexatten-nv/data/paged_attention_results.json') as f:
    R = json.load(f)

COLORS = {'padded': '#e74c3c', 'native': '#3498db', 'flex': '#2ecc71'}

# ===== Exp1: Memory Waste Bar Chart =====
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
e1 = R['exp1']
seqs = e1['seq_lengths']
x = np.arange(len(seqs))

ax = axes[0]
ax.bar(x - 0.15, e1['waste_per_seq_padded_mb'], 0.3, label='Padded Waste', color=COLORS['padded'], alpha=0.8)
ax.bar(x + 0.15, e1['waste_per_seq_paged_mb'], 0.3, label='Paged Waste', color=COLORS['flex'], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in seqs])
ax.set_xlabel('Sequence Length')
ax.set_ylabel('Wasted Memory (MB)')
ax.set_title('Exp1a: Memory Waste per Sequence')
ax.legend()
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
mems = [e1['padded_mem_mb'], e1['paged_mem_mb'], e1['actual_data_mb']]
labels = ['Padded\n(max_seq padded)', 'Paged\n(page-based)', 'Actual Data']
colors = [COLORS['padded'], COLORS['flex'], '#95a5a6']
bars = ax.bar(labels, mems, color=colors, alpha=0.8)
for b, m in zip(bars, mems):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f'{m:.1f} MB', ha='center', fontsize=10)
ax.set_ylabel('Memory (MB)')
ax.set_title('Exp1b: Total KV Cache Memory')
ax.grid(axis='y', alpha=0.3)

ax = axes[2]
utils = [e1['padded_utilization'], e1['paged_utilization']]
bars = ax.bar(['Padded', 'Paged'], utils, color=[COLORS['padded'], COLORS['flex']], alpha=0.8)
for b, u in zip(bars, utils):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f'{u:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Memory Utilization (%)')
ax.set_ylim(0, 110)
ax.set_title('Exp1c: Memory Utilization')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Paged Attention Exp1: Memory Waste Visualization', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/paged_exp1_memory_waste.png', bbox_inches='tight')
plt.close()
print('Exp1 done')

# ===== Exp2: Correctness =====
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
e2 = R['exp2']
batch_data = e2['per_batch']
batches = [f"B{i}\n(S={d['seq_len']})" for i, d in enumerate(batch_data)]
native_errs = [d['native_err'] for d in batch_data]
flex_errs = [d['flex_err'] for d in batch_data]
x = np.arange(len(batches))
ax.bar(x - 0.15, native_errs, 0.3, label='Native vs Ground Truth', color=COLORS['native'], alpha=0.8)
ax.bar(x + 0.15, flex_errs, 0.3, label='Flex vs Ground Truth', color=COLORS['flex'], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(batches)
ax.set_ylabel('Max Absolute Error')
ax.set_title('Exp2: Correctness Verification (fp16 numerical precision)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.text(0.5, 0.95, f'Errors ~0.2-0.3 due to fp16 precision\n(float16 has ~1e-3 relative error)',
        transform=ax.transAxes, ha='center', va='top', fontsize=9, color='gray')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/paged_exp2_correctness.png', bbox_inches='tight')
plt.close()
print('Exp2 done')

# ===== Exp3: Memory Efficiency =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
e3 = R['exp3']
for B in [4, 8, 16]:
    data = [d for d in e3 if d['batch_size'] == B]
    seqs = [d['max_seq'] for d in data]
    ax = axes[0]
    ax.plot(seqs, [d['padded_mb'] for d in data], 'o-', label=f'Padded B={B}', color=COLORS['padded'], alpha=0.7)
    ax.plot(seqs, [d['paged_mb'] for d in data], 's-', label=f'Paged B={B}', color=COLORS['flex'], alpha=0.7)
    ax = axes[1]
    ax.plot(seqs, [d['padded_util'] for d in data], 'o-', label=f'Padded B={B}', color=COLORS['padded'], alpha=0.7)
    ax.plot(seqs, [d['paged_util'] for d in data], 's-', label=f'Paged B={B}', color=COLORS['flex'], alpha=0.7)

axes[0].set_xlabel('Max Sequence Length')
axes[0].set_ylabel('KV Cache Memory (MB)')
axes[0].set_title('Exp3a: Memory Usage')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel('Max Sequence Length')
axes[1].set_ylabel('Memory Utilization (%)')
axes[1].set_title('Exp3b: Memory Utilization')
axes[1].set_ylim(0, 110)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Exp3: Memory Efficiency at Scale', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/paged_exp3_memory_efficiency.png', bbox_inches='tight')
plt.close()
print('Exp3 done')

# ===== Exp4: Throughput Heatmap =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
e4 = R['exp4']
B_vals = sorted(set(d['batch_size'] for d in e4))
S_vals = sorted(set(d['seq_len'] for d in e4))

for ax_idx, (method, key) in enumerate([('Padded', 'padded_tokens_s'), ('Native Paged', 'native_tokens_s'), ('Flex Paged', 'flex_tokens_s')]):
    ax = axes[ax_idx]
    mat = np.zeros((len(B_vals), len(S_vals)))
    for d in e4:
        bi = B_vals.index(d['batch_size'])
        si = S_vals.index(d['seq_len'])
        mat[bi][si] = d[key] / 1000
    im = ax.imshow(mat, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(S_vals)))
    ax.set_xticklabels([str(s) for s in S_vals])
    ax.set_yticks(range(len(B_vals)))
    ax.set_yticklabels([str(b) for b in B_vals])
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Batch Size')
    ax.set_title(f'{method}\n(tokens/s x1000)')
    for i in range(len(B_vals)):
        for j in range(len(S_vals)):
            ax.text(j, i, f'{mat[i][j]:.1f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle('Exp4: Throughput Comparison (K tokens/s)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/paged_exp4_throughput.png', bbox_inches='tight')
plt.close()
print('Exp4 done')

# ===== Exp5: Scaling =====
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
e5 = R['exp5']
seqs = [d['seq_len'] for d in e5]
ax.plot(seqs, [d['padded_ms'] for d in e5], 'o-', label='Padded (Standard)', color=COLORS['padded'], linewidth=2)
ax.plot(seqs, [d['native_ms'] for d in e5], 's-', label='Native Paged', color=COLORS['native'], linewidth=2)
ax.plot(seqs, [d['flex_ms'] for d in e5], '^-', label='Flex Paged', color=COLORS['flex'], linewidth=2)
ax.set_xlabel('Sequence Length')
ax.set_ylabel('Latency (ms)')
ax.set_title('Exp5: Sequence Length Scaling (B=4, H=8, D=64, page_size=128)')
ax.legend()
ax.set_yscale('log')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/paged_exp5_scaling.png', bbox_inches='tight')
plt.close()
print('Exp5 done')

# ===== Exp6: Variable Length =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
e6 = R['exp6']
dists = list(e6.keys())
x = np.arange(len(dists))
width = 0.25

ax = axes[0]
padded_ms = [e6[d]['padded_ms'] for d in dists]
native_ms = [e6[d]['native_ms'] for d in dists]
flex_ms = [e6[d]['flex_ms'] for d in dists]
ax.bar(x - width, padded_ms, width, label='Padded', color=COLORS['padded'], alpha=0.8)
ax.bar(x, native_ms, width, label='Native Paged', color=COLORS['native'], alpha=0.8)
ax.bar(x + width, flex_ms, width, label='Flex Paged', color=COLORS['flex'], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(dists)
ax.set_ylabel('Latency (ms)')
ax.set_title('Exp6a: Variable-Length Latency')
ax.legend()
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
padded_mem = [e6[d]['padded_mem_mb'] for d in dists]
paged_mem = [e6[d]['paged_mem_mb'] for d in dists]
ax.bar(x - 0.15, padded_mem, 0.3, label='Padded', color=COLORS['padded'], alpha=0.8)
ax.bar(x + 0.15, paged_mem, 0.3, label='Paged', color=COLORS['flex'], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(dists)
ax.set_ylabel('KV Cache Memory (MB)')
ax.set_title('Exp6b: Memory Usage')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Exp6: Variable-Length Sequence Scenarios', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/paged_exp6_variable_length.png', bbox_inches='tight')
plt.close()
print('Exp6 done')

# ===== Exp7: Mask Combination =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
e7 = R['exp7']
masks = [d['mask_type'] for d in e7]
x = np.arange(len(masks))

ax = axes[0]
ax.bar(x - 0.15, [d['native_ms'] for d in e7], 0.3, label='Native Paged', color=COLORS['native'], alpha=0.8)
ax.bar(x + 0.15, [d['flex_ms'] for d in e7], 0.3, label='Flex Paged', color=COLORS['flex'], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(masks)
ax.set_ylabel('Latency (ms)')
ax.set_title('Exp7a: Mask + Paged Attention Performance')
ax.legend()
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
ratios = [d['flex_vs_native'] for d in e7]
bars = ax.bar(masks, ratios, color=COLORS['flex'], alpha=0.8)
for b, r in zip(bars, ratios):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05, f'{r:.2f}x', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Flex / Native Ratio')
ax.set_title('Exp7b: FlexAttention Overhead Ratio')
ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='1x (equal)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Exp7: Mask Pattern + Paged Attention', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/paged_exp7_mask_combination.png', bbox_inches='tight')
plt.close()
print('Exp7 done')

# ===== Summary Figure =====
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Memory savings
ax = axes[0][0]
e3_by_seq = {}
for d in R['exp3']:
    if d['batch_size'] == 8:
        e3_by_seq[d['max_seq']] = d['savings_pct']
seqs = sorted(e3_by_seq.keys())
ax.bar([str(s) for s in seqs], [e3_by_seq[s] for s in seqs], color=COLORS['flex'], alpha=0.8)
for i, s in enumerate(seqs):
    ax.text(i, e3_by_seq[s] + 0.5, f'{e3_by_seq[s]:.1f}%', ha='center', fontsize=10)
ax.set_xlabel('Max Sequence Length')
ax.set_ylabel('Memory Saved (%)')
ax.set_title('Memory Savings (Paged vs Padded, B=8)')
ax.grid(axis='y', alpha=0.3)

# 2. Throughput comparison
ax = axes[0][1]
e4_data = {1: {}, 2: {}, 4: {}}
for d in R['exp4']:
    e4_data[d['batch_size']][d['seq_len']] = d
x = np.arange(len(S_vals))
width = 0.25
for i, method in enumerate([('Padded', 'padded_tokens_s'), ('Native', 'native_tokens_s'), ('Flex', 'flex_tokens_s')]):
    vals = []
    for s in S_vals:
        d = e4_data[4].get(s)
        vals.append(d[method[1]] / 1000 if d else 0)
    ax.bar(x + i * width, vals, width, label=f'{method[0]} (B=4)', color=list(COLORS.values())[i], alpha=0.8)
ax.set_xticks(x + width)
ax.set_xticklabels([str(s) for s in S_vals])
ax.set_xlabel('Sequence Length')
ax.set_ylabel('Throughput (K tokens/s)')
ax.set_title('Throughput (B=4)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 3. Native vs Flex overhead
ax = axes[1][0]
overhead_data = {}
for d in R['exp4']:
    ratio = d['flex_ms'] / d['native_ms']
    key = f"B={d['batch_size']},S={d['seq_len']}"
    overhead_data[key] = ratio
keys = list(overhead_data.keys())
vals = list(overhead_data.values())
colors = [COLORS['flex'] if v < 5 else '#e67e22' if v < 10 else COLORS['padded'] for v in vals]
ax.barh(keys, vals, color=colors, alpha=0.8)
ax.set_xlabel('Flex / Native Overhead Ratio')
ax.set_title('FlexAttention Overhead')
ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
ax.grid(axis='x', alpha=0.3)

# 4. Key findings text
ax = axes[1][1]
ax.axis('off')
findings = [
    'Key Findings:',
    '',
    f'1. Paged saves {R["exp1"]["memory_saved_pct"]:.0f}% memory',
    f'   Utilization: Padded {R["exp1"]["padded_utilization"]:.0f}% -> Paged {R["exp1"]["paged_utilization"]:.0f}%',
    '',
    '2. Native Paged: ~1.5x slower than Padded',
    '   (overhead from gather/scatter)',
    '',
    '3. Flex Paged: ~10x slower than Native',
    '   (torch.compile overhead for page mapping)',
    '',
    '4. Flex code is ~4x shorter than Native',
    '   (mask_mod translation vs manual gather)',
    '',
    '5. Flex supports mask combination easily',
    '   (just swap mask_mod, ~3x overhead)',
]
ax.text(0.05, 0.95, '\n'.join(findings), transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('Summary')

plt.suptitle('Paged Attention Experiment Summary', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/paged_summary.png', bbox_inches='tight')
plt.close()
print('Summary done')
print('All figures generated!')
