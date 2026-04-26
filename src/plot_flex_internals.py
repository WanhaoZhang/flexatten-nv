import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.dpi'] = 150

FIG = '../docs/figures'
DATA = '../data'

with open(f'{DATA}/flex_internals_results.json') as f:
    R = json.load(f)


def fig1_blockmask_anatomy():
    d = R['exp1']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1a: Block size sweep
    bs_data = d['block_size_sweep']
    reqs = [x['requested'] for x in bs_data]
    actuals = [x.get('actual', 0) for x in bs_data]
    sparsities = [x.get('sparsity', 0) for x in bs_data]

    ax = axes[0]
    ax.bar([str(r) for r in reqs], sparsities, color='#3498db', edgecolor='white')
    ax.set_xlabel('Requested BLOCK_SIZE')
    ax.set_ylabel('Causal Sparsity (%)')
    ax.set_title('PyTorch 2.5.1: BLOCK_SIZE Ignored!\n(Always forced to 128)')
    for i, (r, s) in enumerate(zip(reqs, sparsities)):
        ax.text(i, s + 1, f'Actual=128\nSp={s:.0f}%', ha='center', fontsize=8)

    # 1b: Conceptual block mask visualization
    ax = axes[1]
    S = 8
    mask = np.tril(np.ones((S, S)))
    ax.imshow(mask, cmap='Blues', aspect='equal')
    ax.set_xlabel('KV Index')
    ax.set_ylabel('Query Index')
    ax.set_title('Causal Block Mask Concept\n(8x8 with BLOCK_SIZE=2)')

    # Draw block grid
    for i in range(0, S + 1, 2):
        ax.axhline(i - 0.5, color='red', lw=1, alpha=0.5)
        ax.axvline(i - 0.5, color='red', lw=1, alpha=0.5)

    # 1c: Sparsity vs seq_len for different patterns
    ax = axes[2]
    seq_lens = [x['seq_len'] for x in R['exp3']]
    causal_sp = [x['causal_sparsity'] for x in R['exp3']]
    sw_sp = [x['sw_sparsity'] for x in R['exp3']]
    prefix_sp = [x['prefix_sparsity'] for x in R['exp3']]
    ax.plot(seq_lens, causal_sp, 'o-', color='#e74c3c', lw=2, label='Causal')
    ax.plot(seq_lens, sw_sp, 's-', color='#3498db', lw=2, label='Sliding(64)')
    ax.plot(seq_lens, prefix_sp, '^-', color='#2ecc71', lw=2, label='Prefix(64)+Causal')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Block Sparsity (%)')
    ax.set_title('Sparsity vs Sequence Length')
    ax.legend()

    plt.suptitle('Exp1: BlockMask Internal Structure', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{FIG}/flex_fig1_blockmask.png')
    plt.close()
    print("  fig1 saved")


def fig2_score_mod_latency():
    d = R['exp2']
    names = list(d.keys())
    latencies = [d[n]['latency_ms'] for n in names]
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, latencies, color=colors[:len(names)], edgecolor='white')
    for b, v in zip(bars, latencies):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.2f}ms', ha='center', fontsize=10)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Exp2: score_mod Compilation & Execution Latency\n(B=1, H=1, S=64, D=32, cached after first compile)')
    plt.tight_layout()
    plt.savefig(f'{FIG}/flex_fig2_score_mod.png')
    plt.close()
    print("  fig2 saved")


def fig3_sparsity_vs_perf():
    d = R['exp3']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    seq_lens = [x['seq_len'] for x in d]

    ax1.plot(seq_lens, [x['sdpa_ms'] for x in d], 'D-', color='black', lw=2, ms=7, label='SDPA dense')
    ax1.plot(seq_lens, [x['fa_dense_ms'] for x in d], 'o-', color='#3498db', lw=2, label='FA dense')
    ax1.plot(seq_lens, [x['fa_causal_ms'] for x in d], 's-', color='#e74c3c', lw=2, label='FA causal')
    ax1.plot(seq_lens, [x['fa_sw_ms'] for x in d], '^-', color='#2ecc71', lw=2, label='FA sw64')
    ax1.plot(seq_lens, [x['fa_prefix_ms'] for x in d], 'v-', color='#f39c12', lw=2, label='FA prefix')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Latency Comparison')
    ax1.legend()
    ax1.set_yscale('log')

    overheads = [x['fa_dense_ms'] / x['sdpa_ms'] for x in d]
    ax2.bar([str(s) for s in seq_lens], overheads, color='#e74c3c', edgecolor='white')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('FlexAttention / SDPA (x)')
    ax2.set_title('FlexAttention Overhead vs SDPA Dense')
    for i, v in enumerate(overheads):
        ax2.text(i, v + 1, f'{v:.0f}x', ha='center', fontsize=9)
    ax2.axhline(y=1.0, color='green', ls='--', alpha=0.5, label='1x (parity)')
    ax2.legend()

    plt.suptitle('Exp3: BlockMask Sparsity vs Performance (B=2, H=8, D=64)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{FIG}/flex_fig3_sparsity_perf.png')
    plt.close()
    print("  fig3 saved")


def fig4_compile_overhead():
    d = R['exp5']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    configs = {}
    for x in d:
        name = x['name']
        if name not in configs:
            configs[name] = {'seq_lens': [], 'first': [], 'cached': [], 'sdpa': []}
        configs[name]['seq_lens'].append(x['seq_len'])
        configs[name]['first'].append(x['first_call_ms'])
        configs[name]['cached'].append(x['cached_ms'])
        configs[name]['sdpa'].append(x['sdpa_ms'])

    colors = {'no_mod': '#3498db', 'causal_score': '#e74c3c', 'causal_block': '#2ecc71'}
    for name, data in configs.items():
        ax1.plot(data['seq_lens'], data['first'], 'o-', color=colors[name], lw=2, label=f'{name} (first)')
        ax1.plot(data['seq_lens'], data['cached'], 's--', color=colors[name], lw=1.5, alpha=0.7, label=f'{name} (cached)')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('First Call (compile) vs Cached')
    ax1.legend(fontsize=9)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')

    compile_overheads = {}
    for x in d:
        name = x['name']
        if name not in compile_overheads:
            compile_overheads[name] = {'seq_lens': [], 'overheads': []}
        compile_overheads[name]['seq_lens'].append(x['seq_len'])
        compile_overheads[name]['overheads'].append(x['compile_overhead_ms'])

    for name, data in compile_overheads.items():
        ax2.bar([f"{name}\nS={s}" for s in data['seq_lens']], data['overheads'],
                color=colors[name], edgecolor='white', alpha=0.8)
    ax2.set_ylabel('Compile Overhead (ms)')
    ax2.set_title('torch.compile Overhead')
    ax2.tick_params(axis='x', rotation=45)

    plt.suptitle('Exp5: torch.compile Compilation Overhead Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{FIG}/flex_fig4_compile.png')
    plt.close()
    print("  fig4 saved")


def fig5_latency_showdown():
    d = R['exp6']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    seq_lens = [x['seq_len'] for x in d]

    ax1.plot(seq_lens, [x['sdpa_dense'] for x in d], 'D-', color='black', lw=2, ms=7, label='SDPA dense')
    ax1.plot(seq_lens, [x['sdpa_causal'] for x in d], 'D--', color='gray', lw=2, ms=7, label='SDPA causal')
    ax1.plot(seq_lens, [x['flex_dense'] for x in d], 'o-', color='#3498db', lw=2, label='FA dense')
    ax1.plot(seq_lens, [x['flex_causal_bm'] for x in d], 's-', color='#e74c3c', lw=2, label='FA causal(block)')
    ax1.plot(seq_lens, [x['flex_causal_sm'] for x in d], '^-', color='#f39c12', lw=2, label='FA causal(score)')
    ax1.plot(seq_lens, [x['flex_sw64'] for x in d], 'v-', color='#2ecc71', lw=2, label='FA sw64')
    ax1.plot(seq_lens, [x['flex_gqa'] for x in d], 'p-', color='#9b59b6', lw=2, label='FA GQA')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Absolute Latency')
    ax1.legend(fontsize=9)
    ax1.set_xscale('log', base=2)

    ratios = [x['flex_dense'] / x['sdpa_dense'] for x in d]
    ax2.bar([str(s) for s in seq_lens], ratios, color='#e74c3c', edgecolor='white')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('FA_dense / SDPA_dense (x)')
    ax2.set_title('FlexAttention Overhead Factor')
    ax2.axhline(y=1.0, color='green', ls='--', alpha=0.5)
    for i, v in enumerate(ratios):
        ax2.text(i, v + 1, f'{v:.0f}x', ha='center', fontsize=9)

    plt.suptitle('Exp6: FlexAttention vs SDPA Latency Showdown (B=2, H=8, D=64)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{FIG}/flex_fig5_showdown.png')
    plt.close()
    print("  fig5 saved")


def fig6_pattern_analysis():
    d = R['exp8']
    patterns = d['patterns']
    sdpa_dense = d['sdpa_dense_ms']

    names = list(patterns.keys())
    lats = [patterns[n]['ms'] for n in names]
    spars = [patterns[n]['sparsity'] for n in names]
    overheads = [l / sdpa_dense for l in lats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6', '#1abc9c', '#e67e22', '#95a5a6']
    bars = ax1.barh(names, lats, color=colors[:len(names)], edgecolor='white')
    ax1.axvline(x=sdpa_dense, color='black', ls='--', lw=2, label=f'SDPA dense ({sdpa_dense:.2f}ms)')
    ax1.set_xlabel('Latency (ms)')
    ax1.set_title('Latency by Pattern (S=1024)')
    ax1.legend()

    scatter = ax2.scatter(spars, lats, s=200, c=colors[:len(names)], edgecolors='black', lw=1, zorder=5)
    ax2.axhline(y=sdpa_dense, color='black', ls='--', lw=1.5, alpha=0.5)
    for i, name in enumerate(names):
        ax2.annotate(name, (spars[i], lats[i]), textcoords="offset points",
                     xytext=(10, 5 + i * 3), fontsize=9, color=colors[i])
    ax2.set_xlabel('Block Sparsity (%)')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Sparsity vs Latency\n(Does more sparsity = faster?)')

    plt.suptitle('Exp8: Attention Pattern Performance Analysis (B=2, H=8, S=1024, D=64)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{FIG}/flex_fig6_patterns.png')
    plt.close()
    print("  fig6 saved")


if __name__ == '__main__':
    print("Generating FlexAttention internals figures...")
    fig1_blockmask_anatomy()
    fig2_score_mod_latency()
    fig3_sparsity_vs_perf()
    fig4_compile_overhead()
    fig5_latency_showdown()
    fig6_pattern_analysis()
    print("All figures generated!")
