import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams['font.family'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.dpi'] = 150

FIG = '../docs/figures'
DATA = '../data'

with open(f'{DATA}/mla_results.json') as f:
    R = json.load(f)

COLORS = {
    'MHA': '#e74c3c', 'MQA': '#3498db', 'GQA4': '#2ecc71', 'GQA8': '#f39c12',
    'MLA': '#9b59b6', 'Vanilla': '#2ecc71', 'Absorb': '#e67e22',
    'blue': '#3498db', 'green': '#2ecc71', 'orange': '#e67e22',
    'red': '#e74c3c', 'purple': '#9b59b6', 'gray': '#95a5a6',
}


def fig1_kv_cache_memory():
    d = R['exp1']
    seq_lens = [x['seq_len'] for x in d]
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, [x['mha_mb'] for x in d], 'o-', color=COLORS['MHA'], lw=2, ms=6, label='MHA (128 heads)')
    plt.plot(seq_lens, [x['gqa8_mb'] for x in d], 's-', color=COLORS['GQA8'], lw=2, ms=6, label='GQA-8')
    plt.plot(seq_lens, [x['mqa_mb'] for x in d], '^-', color=COLORS['MQA'], lw=2, ms=6, label='MQA')
    plt.plot(seq_lens, [x['mla_mb'] for x in d], 'D-', color=COLORS['MLA'], lw=2, ms=6, label='MLA (rank=512)')
    plt.xlabel('Sequence Length')
    plt.ylabel('KV Cache Memory (MB)')
    plt.title('Exp1: KV Cache Memory Comparison')
    plt.legend()
    plt.xscale('log', base=2)
    plt.yscale('log')
    ax = plt.gca()
    ax.set_xticks(seq_lens)
    ax.set_xticklabels([str(s) for s in seq_lens])
    plt.tight_layout()
    plt.savefig(f'{FIG}/mla_fig1_kv_cache_memory.png')
    plt.close()
    print("  fig1 saved")


def fig2_per_token_bytes():
    types = ['MHA', 'GQA-4', 'GQA-8', 'MQA', 'MLA']
    bytes_per_token = [16384, 2048, 4096, 512, 1152]
    colors = [COLORS['MHA'], COLORS['GQA4'], COLORS['GQA8'], COLORS['MQA'], COLORS['MLA']]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(types, bytes_per_token, color=colors, edgecolor='white', lw=1.5)
    for b, v in zip(bars, bytes_per_token):
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 200, f'{v:,}B',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.ylabel('KV Cache per Token (Bytes)')
    plt.title('Exp1: KV Cache per Token Comparison')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{FIG}/mla_fig2_per_token_bytes.png')
    plt.close()
    print("  fig2 saved")


def fig3_correctness():
    d = R['exp2']
    batches = [x['batch'] for x in d['per_batch']]
    cos_sims = [x['cos_sim'] for x in d['per_batch']]
    max_errs = [x['max_err'] for x in d['per_batch']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar([f'B{i}' for i in batches], cos_sims, color=COLORS['green'], edgecolor='white')
    ax1.set_ylim(0.998, 1.001)
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Cosine Similarity per Batch')
    ax1.axhline(y=1.0, color='gray', ls='--', alpha=0.5)
    for i, v in enumerate(cos_sims):
        ax1.text(i, v + 0.0001, f'{v:.6f}', ha='center', fontsize=9)

    ax2.bar([f'B{i}' for i in batches], max_errs, color=COLORS['orange'], edgecolor='white')
    ax2.set_ylabel('Max Absolute Error')
    ax2.set_title('Max Absolute Error per Batch')
    for i, v in enumerate(max_errs):
        ax2.text(i, v * 1.05, f'{v:.2e}', ha='center', fontsize=9)

    fig.suptitle(f'Exp2: Vanilla MLA vs MatAbsorb MLA (Overall cos_sim={d["overall_cos_sim"]:.6f})', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{FIG}/mla_fig3_correctness.png')
    plt.close()
    print("  fig3 saved")


def fig4_lora_rank_tradeoff():
    d = R['exp3']
    ranks = [x['kv_lora_rank'] for x in d]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(ranks, [x['cos_sim'] for x in d], 'o-', color=COLORS['blue'], lw=2, ms=7)
    axes[0].set_xlabel('kv_lora_rank')
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Correctness')
    axes[0].set_ylim(0.995, 1.005)
    axes[0].axhline(y=1.0, color='gray', ls='--', alpha=0.5)

    axes[1].plot(ranks, [x['vanilla_ms'] for x in d], 'o-', color=COLORS['Vanilla'], lw=2, ms=7, label='Vanilla')
    axes[1].plot(ranks, [x['absorb_ms'] for x in d], 's-', color=COLORS['Absorb'], lw=2, ms=7, label='MatAbsorb')
    axes[1].set_xlabel('kv_lora_rank')
    axes[1].set_ylabel('Latency (ms)')
    axes[1].set_title('Decode Latency')
    axes[1].legend()

    axes[2].bar([str(r) for r in ranks], [x['kv_cache_ratio_vs_mha'] for x in d],
                color=COLORS['purple'], edgecolor='white')
    axes[2].set_xlabel('kv_lora_rank')
    axes[2].set_ylabel('KV Cache / MHA KV Cache (%)')
    axes[2].set_title('Memory Ratio')

    fig.suptitle('Exp3: kv_lora_rank Capacity Sweep', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{FIG}/mla_fig4_lora_rank_tradeoff.png')
    plt.close()
    print("  fig4 saved")


def fig5_decode_latency():
    d = R['exp4']
    kv_lens = [x['kv_len'] for x in d]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(kv_lens, [x['vanilla_ms'] for x in d], 'o-', color=COLORS['Vanilla'], lw=2, ms=7, label='Vanilla MLA')
    ax1.plot(kv_lens, [x['absorb_ms'] for x in d], 's-', color=COLORS['Absorb'], lw=2, ms=7, label='MatAbsorb MLA')
    ax1.set_xlabel('KV Cache Length')
    ax1.set_ylabel('Decode Latency (ms)')
    ax1.set_title('Decode Latency vs KV Length')
    ax1.legend()
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(kv_lens)
    ax1.set_xticklabels([str(s) for s in kv_lens])

    ax2.bar([str(k) for k in kv_lens], [x['speedup'] for x in d],
            color=[COLORS['green'] if s >= 1 else COLORS['red'] for s in [x['speedup'] for x in d]],
            edgecolor='white')
    ax2.set_xlabel('KV Cache Length')
    ax2.set_ylabel('Speedup (Vanilla / Absorb)')
    ax2.set_title('MatAbsorb Speedup')
    ax2.axhline(y=1.0, color='gray', ls='--', alpha=0.5)

    fig.suptitle('Exp4: Decode Latency Benchmark (B=1)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{FIG}/mla_fig5_decode_latency.png')
    plt.close()
    print("  fig5 saved")


def fig6_seq_scaling():
    d = R['exp5']
    kv_lens = [x['kv_len'] for x in d]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(kv_lens, [x['vanilla_ms'] for x in d], 'o-', color=COLORS['Vanilla'], lw=2, ms=7, label='Vanilla MLA')
    ax1.plot(kv_lens, [x['absorb_ms'] for x in d], 's-', color=COLORS['Absorb'], lw=2, ms=7, label='MatAbsorb MLA')
    ax1.set_xlabel('KV Cache Length')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Latency vs Sequence Length')
    ax1.legend()
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(kv_lens)
    ax1.set_xticklabels([str(s) for s in kv_lens])

    ax2b = ax2.twinx()
    ax2.bar([str(k) for k in kv_lens], [x['mla_kv_mb'] for x in d], alpha=0.6,
            color=COLORS['MLA'], label='MLA KV Mem', edgecolor='white')
    ax2b.plot([str(k) for k in kv_lens], [x['mha_kv_mb'] for x in d], 'o-',
              color=COLORS['MHA'], lw=2, ms=7, label='MHA KV Mem')
    ax2.set_ylabel('MLA KV Cache (MB)', color=COLORS['MLA'])
    ax2b.set_ylabel('MHA KV Cache (MB)', color=COLORS['MHA'])
    ax2.set_title('KV Cache Memory (B=2)')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    fig.suptitle('Exp5: Sequence Length Scaling (B=2)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{FIG}/mla_fig6_seq_scaling.png')
    plt.close()
    print("  fig6 saved")


def fig7_batch_scaling():
    d = R['exp6']
    batches = [x['batch_size'] for x in d]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(batches, [x['vanilla_ms'] for x in d], 'o-', color=COLORS['Vanilla'], lw=2, ms=7, label='Vanilla MLA')
    ax1.plot(batches, [x['absorb_ms'] for x in d], 's-', color=COLORS['Absorb'], lw=2, ms=7, label='MatAbsorb MLA')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Latency vs Batch Size')
    ax1.legend()

    ax1_twin = ax1.twinx()
    speedups = [x['speedup'] for x in d]
    ax1_twin.plot(batches, speedups, 'D--', color=COLORS['purple'], lw=1.5, ms=6, alpha=0.7, label='Speedup')
    ax1_twin.set_ylabel('Speedup', color=COLORS['purple'])
    ax1_twin.axhline(y=1.0, color='gray', ls=':', alpha=0.5)

    tp_vanilla = [x['vanilla_tokens_s'] for x in d]
    tp_absorb = [x['absorb_tokens_s'] for x in d]
    ax2.bar([f'B={b}' for b in batches], tp_vanilla, width=0.35, color=COLORS['Vanilla'], label='Vanilla', edgecolor='white')
    ax2.bar([i + 0.35 for i in range(len(batches))], tp_absorb, width=0.35, color=COLORS['Absorb'], label='Absorb', edgecolor='white')
    ax2.set_ylabel('Throughput (tokens/s)')
    ax2.set_title('Throughput')
    ax2.legend()
    ax2.set_xticks([i + 0.175 for i in range(len(batches))])
    ax2.set_xticklabels([f'B={b}' for b in batches])

    fig.suptitle('Exp6: Batch Size Scaling (kv_len=512)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{FIG}/mla_fig7_batch_scaling.png')
    plt.close()
    print("  fig7 saved")


def fig8_attention_comparison():
    d = R['exp7']
    names = [x['name'] for x in d]
    mems = [x['total_mem_mb'] for x in d]
    lats = [x['latency_ms'] for x in d]
    clrs = [COLORS.get(n, COLORS['gray']) for n in names]
    clrs = []
    color_map = {'MHA': COLORS['MHA'], 'GQA-4': COLORS['GQA4'], 'GQA-8': COLORS['GQA8'],
                 'MQA': COLORS['MQA'], 'MLA': COLORS['MLA'], 'MLA_Vanilla': COLORS['Vanilla'],
                 'MLA_MatAbsorb': COLORS['Absorb']}
    for n in names:
        clrs.append(color_map.get(n, COLORS['gray']))

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (name, mem, lat, c) in enumerate(zip(names, mems, lats, clrs)):
        ax.scatter(lat, mem, s=200, c=c, edgecolors='black', lw=1, zorder=5)
        offset_y = mem * 0.08 if mem > 2 else 0.3
        ax.annotate(name, (lat, mem), textcoords="offset points", xytext=(10, 5 + i * 2),
                    fontsize=11, fontweight='bold', color=c)

    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('KV Cache Memory (MB)')
    ax.set_title('Exp7: Attention Mechanism Comparison\nMemory vs Latency Tradeoff (B=4, kv_len=512)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(f'{FIG}/mla_fig8_attention_comparison.png')
    plt.close()
    print("  fig8 saved")


if __name__ == '__main__':
    print("Generating MLA figures...")
    fig1_kv_cache_memory()
    fig2_per_token_bytes()
    fig3_correctness()
    fig4_lora_rank_tradeoff()
    fig5_decode_latency()
    fig6_seq_scaling()
    fig7_batch_scaling()
    fig8_attention_comparison()
    print("All figures generated!")
