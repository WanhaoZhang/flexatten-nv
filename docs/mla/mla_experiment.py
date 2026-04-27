import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import os
import gc

torch.manual_seed(42)
torch.cuda.manual_seed(42)

DTYPE = torch.float16
DEVICE = 'cuda'
RESULTS = {}
DATA_DIR = '../data'
FIGURE_DIR = '../docs/figures'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * out).to(x.dtype)


class MLAConfig:
    def __init__(self, hidden_size=2048, num_heads=32, q_lora_rank=512,
                 kv_lora_rank=256, qk_nope_head_dim=64, qk_rope_head_dim=32,
                 v_head_dim=64):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim


class VanillaMLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.q_a_proj = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(cfg.q_lora_rank)
        self.q_b_proj = nn.Linear(cfg.q_lora_rank, cfg.num_heads * cfg.q_head_dim, bias=False)
        self.kv_b_proj = nn.Linear(cfg.kv_lora_rank, cfg.num_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.v_head_dim, cfg.hidden_size, bias=False)
        self.softmax_scale = cfg.q_head_dim ** (-0.5)

    def forward(self, hidden_states, c_kv_cache, k_pe_cache):
        cfg = self.cfg
        B = hidden_states.size(0)
        q_len = hidden_states.size(1)
        kv_len = c_kv_cache.size(1)

        c_Q = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(c_Q).view(B, q_len, cfg.num_heads, cfg.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1)

        kv = self.kv_b_proj(c_kv_cache).view(B, kv_len, cfg.num_heads, cfg.qk_nope_head_dim + cfg.v_head_dim).transpose(1, 2)
        k_nope, v = torch.split(kv, [cfg.qk_nope_head_dim, cfg.v_head_dim], dim=-1)

        k_pe = k_pe_cache.unsqueeze(2).expand(-1, -1, cfg.num_heads, -1).transpose(1, 2)
        rope_scores = torch.matmul(q_pe, k_pe.transpose(-2, -1)) * cfg.qk_rope_head_dim ** (-0.5)

        scores_nope = torch.matmul(q_nope, k_nope.transpose(-2, -1)) * self.softmax_scale
        scores = scores_nope + rope_scores

        attn = F.softmax(scores.float(), dim=-1).to(DTYPE)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(B, q_len, -1)
        return self.o_proj(output)


class MatAbsorbMLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.q_a_proj = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(cfg.q_lora_rank)
        self.W_UQ_UK = nn.Parameter(torch.randn(cfg.q_lora_rank, cfg.num_heads * cfg.kv_lora_rank) * 0.02)
        self.W_QR = nn.Parameter(torch.randn(cfg.q_lora_rank, cfg.num_heads * cfg.qk_rope_head_dim) * 0.02)
        self.W_UV_O = nn.Parameter(torch.randn(cfg.num_heads * cfg.kv_lora_rank, cfg.hidden_size) * 0.02)
        self.softmax_scale = cfg.q_head_dim ** (-0.5)

    @classmethod
    def from_vanilla(cls, vanilla):
        cfg = vanilla.cfg
        m = cls(cfg)
        m.q_a_proj.weight.data.copy_(vanilla.q_a_proj.weight.data)
        m.q_a_norm.weight.data.copy_(vanilla.q_a_norm.weight.data)

        with torch.no_grad():
            W_qb = vanilla.q_b_proj.weight.data.T.view(cfg.q_lora_rank, cfg.num_heads, cfg.q_head_dim)
            W_UQ, W_QR = torch.split(W_qb, [cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1)
            m.W_QR.data.copy_(W_QR.reshape(cfg.q_lora_rank, cfg.num_heads * cfg.qk_rope_head_dim))

            W_kvb = vanilla.kv_b_proj.weight.data.T.view(cfg.kv_lora_rank, cfg.num_heads, cfg.qk_nope_head_dim + cfg.v_head_dim)
            W_UK, W_UV = torch.split(W_kvb, [cfg.qk_nope_head_dim, cfg.v_head_dim], dim=-1)
            m.W_UQ_UK.data.copy_(torch.einsum('qnd,lnd->qnl', W_UQ, W_UK).flatten(1, 2))

            W_O = vanilla.o_proj.weight.data.view(cfg.hidden_size, cfg.num_heads, cfg.v_head_dim).permute(1, 2, 0)
            m.W_UV_O.data.copy_(torch.einsum('lnd,ndh->nlh', W_UV, W_O).flatten(0, 1))

        return m

    def forward(self, hidden_states, c_kv_cache, k_pe_cache):
        cfg = self.cfg
        B = hidden_states.size(0)
        q_len = hidden_states.size(1)
        kv_len = c_kv_cache.size(1)

        c_Q = self.q_a_norm(self.q_a_proj(hidden_states))

        q_nope = torch.matmul(c_Q, self.W_UQ_UK).view(B, q_len, cfg.num_heads, cfg.kv_lora_rank).transpose(1, 2)
        q_pe = torch.matmul(c_Q, self.W_QR).view(B, q_len, cfg.num_heads, cfg.qk_rope_head_dim).transpose(1, 2)

        k_pe = k_pe_cache.unsqueeze(1).expand(-1, cfg.num_heads, -1, -1)
        rope_scores = torch.matmul(q_pe, k_pe.transpose(-2, -1)) * cfg.qk_rope_head_dim ** (-0.5)

        c_kv_t = c_kv_cache.transpose(-2, -1).unsqueeze(1).expand(-1, cfg.num_heads, -1, -1)
        scores_nope = torch.matmul(q_nope, c_kv_t) * self.softmax_scale

        scores = scores_nope + rope_scores

        attn = F.softmax(scores.float(), dim=-1).to(DTYPE)
        c_kv_exp = c_kv_cache.unsqueeze(1).expand(-1, cfg.num_heads, -1, -1)
        attn_output = torch.matmul(attn, c_kv_exp)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, q_len, cfg.num_heads * cfg.kv_lora_rank)
        return torch.matmul(attn_output, self.W_UV_O)


def make_cache(B, kv_len, cfg, device='cuda'):
    c_kv = torch.randn(B, kv_len, cfg.kv_lora_rank, device=device, dtype=DTYPE)
    k_pe = torch.randn(B, kv_len, cfg.qk_rope_head_dim, device=device, dtype=DTYPE)
    return c_kv, k_pe


def warmup():
    cfg = MLAConfig()
    c_kv, k_pe = make_cache(1, 64, cfg)
    vanilla = VanillaMLA(cfg).cuda().half()
    absorb = MatAbsorbMLA.from_vanilla(vanilla).cuda().half()
    x = torch.randn(1, 1, cfg.hidden_size, device=DEVICE, dtype=DTYPE)
    for _ in range(3):
        _ = vanilla(x, c_kv, k_pe)
        _ = absorb(x, c_kv, k_pe)
    torch.cuda.synchronize()


def exp1_memory():
    print("=" * 60)
    print("Exp1: KV Cache Memory Comparison")
    print("=" * 60)
    num_heads, head_dim, rope_dim = 128, 128, 64
    seq_lens = [512, 1024, 2048, 4096, 8192, 16384]
    kv_lora_rank = 512
    results = []
    for S in seq_lens:
        mha = 2 * num_heads * head_dim * S * 2
        mqa = 2 * 1 * head_dim * S * 2
        gqa4 = 2 * 4 * head_dim * S * 2
        gqa8 = 2 * 8 * head_dim * S * 2
        mla = (kv_lora_rank + rope_dim) * S * 2
        results.append({
            'seq_len': S,
            'mha_mb': mha / 1024**2,
            'mqa_mb': mqa / 1024**2,
            'gqa4_mb': gqa4 / 1024**2,
            'gqa8_mb': gqa8 / 1024**2,
            'mla_mb': mla / 1024**2,
            'mha_per_token': 2 * num_heads * head_dim * 2,
            'mqa_per_token': 2 * 1 * head_dim * 2,
            'gqa8_per_token': 2 * 8 * head_dim * 2,
            'mla_per_token': (kv_lora_rank + rope_dim) * 2,
            'mla_vs_mha': mla / mha * 100,
            'mla_vs_gqa8': mla / gqa8 * 100,
        })
        print(f"  S={S}: MHA {mha/1024**2:.1f}MB | GQA8 {gqa8/1024**2:.1f}MB | MQA {mqa/1024**2:.1f}MB | MLA {mla/1024**2:.1f}MB ({mla/mha*100:.1f}% of MHA)")
    RESULTS['exp1'] = results


def exp2_correctness():
    print("=" * 60)
    print("Exp2: Correctness Verification")
    print("=" * 60)
    torch.manual_seed(42)
    cfg = MLAConfig()
    B, kv_len = 4, 256

    vanilla = VanillaMLA(cfg).cuda().half().eval()
    absorb = MatAbsorbMLA.from_vanilla(vanilla).cuda().half().eval()

    c_kv, k_pe = make_cache(B, kv_len, cfg)
    x = torch.randn(B, 1, cfg.hidden_size, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        out_vanilla = vanilla(x, c_kv, k_pe)
        out_absorb = absorb(x, c_kv, k_pe)

    cos_sim = F.cosine_similarity(out_vanilla.view(-1), out_absorb.view(-1), dim=0).item()
    max_err = (out_vanilla - out_absorb).abs().max().item()
    mean_err = (out_vanilla - out_absorb).abs().mean().item()
    print(f"  Vanilla vs MatAbsorb: cos_sim={cos_sim:.6f}, max_err={max_err:.6e}, mean_err={mean_err:.6e}")

    per_batch = []
    for i in range(B):
        cs = F.cosine_similarity(out_vanilla[i].view(-1), out_absorb[i].view(-1), dim=0).item()
        me = (out_vanilla[i] - out_absorb[i]).abs().max().item()
        per_batch.append({'batch': i, 'cos_sim': cs, 'max_err': me})
        print(f"    B{i}: cos_sim={cs:.6f}, max_err={me:.6e}")

    RESULTS['exp2'] = {
        'overall_cos_sim': cos_sim,
        'overall_max_err': max_err,
        'overall_mean_err': mean_err,
        'per_batch': per_batch,
        'flex_note': 'FlexAttention score_mod with dynamic tensor indexing requires PyTorch 2.6+; PyTorch 2.5.1 torch.compile does not support data-dependent indexing in score_mod closures.',
    }
    del vanilla, absorb, c_kv, k_pe, x, out_vanilla, out_absorb
    gc.collect(); torch.cuda.empty_cache()


def exp3_lora_rank():
    print("=" * 60)
    print("Exp3: kv_lora_rank Capacity Sweep")
    print("=" * 60)
    ranks = [64, 128, 256, 512, 768, 1024]
    B, kv_len, n_trials = 4, 256, 5
    results = []

    for rank in ranks:
        cfg = MLAConfig(kv_lora_rank=rank)
        vanilla = VanillaMLA(cfg).cuda().half().eval()
        absorb = MatAbsorbMLA.from_vanilla(vanilla).cuda().half().eval()
        c_kv, k_pe = make_cache(B, kv_len, cfg)
        x = torch.randn(B, 1, cfg.hidden_size, device=DEVICE, dtype=DTYPE)

        with torch.no_grad():
            out_v = vanilla(x, c_kv, k_pe)
            out_a = absorb(x, c_kv, k_pe)
        cs = F.cosine_similarity(out_v.view(-1), out_a.view(-1), dim=0).item()

        kv_mem = (rank + cfg.qk_rope_head_dim) * kv_len * 2
        mha_mem = 2 * cfg.num_heads * cfg.qk_nope_head_dim * kv_len * 2

        for _ in range(3):
            _ = absorb(x, c_kv, k_pe)
        torch.cuda.synchronize()
        times = []
        for _ in range(n_trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = absorb(x, c_kv, k_pe)
            torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        ms = sum(times) / len(times) * 1000

        for _ in range(3):
            _ = vanilla(x, c_kv, k_pe)
        torch.cuda.synchronize()
        times_v = []
        for _ in range(n_trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = vanilla(x, c_kv, k_pe)
            torch.cuda.synchronize(); times_v.append(time.perf_counter() - t0)
        ms_v = sum(times_v) / len(times_v) * 1000

        results.append({
            'kv_lora_rank': rank,
            'kv_cache_per_token_bytes': (rank + cfg.qk_rope_head_dim) * 2,
            'kv_cache_ratio_vs_mha': kv_mem / mha_mem * 100,
            'cos_sim': cs,
            'vanilla_ms': ms_v,
            'absorb_ms': ms,
            'speedup': ms_v / ms if ms > 0 else 0,
        })
        print(f"  rank={rank}: cos_sim={cs:.4f}, vanilla={ms_v:.2f}ms, absorb={ms:.2f}ms, speedup={ms_v/ms:.2f}x, mem={kv_mem/mha_mem*100:.1f}% of MHA")
        del vanilla, absorb, c_kv, k_pe, x; gc.collect(); torch.cuda.empty_cache()

    RESULTS['exp3'] = results


def exp4_decode_latency():
    print("=" * 60)
    print("Exp4: Decode Latency Benchmark")
    print("=" * 60)
    cfg = MLAConfig()
    kv_lens = [64, 128, 256, 512, 1024, 2048]
    B, n_warmup, n_trials = 1, 3, 10
    results = []

    for kv_len in kv_lens:
        vanilla = VanillaMLA(cfg).cuda().half().eval()
        absorb = MatAbsorbMLA.from_vanilla(vanilla).cuda().half().eval()
        c_kv, k_pe = make_cache(B, kv_len, cfg)
        x = torch.randn(B, 1, cfg.hidden_size, device=DEVICE, dtype=DTYPE)

        for _ in range(n_warmup):
            _ = vanilla(x, c_kv, k_pe)
        torch.cuda.synchronize()
        tv = []
        for _ in range(n_trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = vanilla(x, c_kv, k_pe)
            torch.cuda.synchronize(); tv.append(time.perf_counter() - t0)
        ms_v = sum(tv) / len(tv) * 1000

        for _ in range(n_warmup):
            _ = absorb(x, c_kv, k_pe)
        torch.cuda.synchronize()
        ta = []
        for _ in range(n_trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = absorb(x, c_kv, k_pe)
            torch.cuda.synchronize(); ta.append(time.perf_counter() - t0)
        ms_a = sum(ta) / len(ta) * 1000

        kv_mem_mla = (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * kv_len * 2 / 1024**2
        kv_mem_mha = 2 * cfg.num_heads * cfg.qk_nope_head_dim * kv_len * 2 / 1024**2

        results.append({
            'kv_len': kv_len,
            'vanilla_ms': ms_v, 'absorb_ms': ms_a,
            'speedup': ms_v / ms_a,
            'mla_kv_mb': kv_mem_mla, 'mha_kv_mb': kv_mem_mha,
            'mem_saving': (1 - kv_mem_mla / kv_mem_mha) * 100,
        })
        print(f"  kv={kv_len}: Vanilla {ms_v:.2f}ms | Absorb {ms_a:.2f}ms | Speedup {ms_v/ms_a:.2f}x | Mem saved {(1-kv_mem_mla/kv_mem_mha)*100:.1f}%")
        del vanilla, absorb, c_kv, k_pe, x; gc.collect(); torch.cuda.empty_cache()

    RESULTS['exp4'] = results


def exp5_seq_scaling():
    print("=" * 60)
    print("Exp5: Sequence Length Scaling")
    print("=" * 60)
    cfg = MLAConfig()
    kv_lens = [64, 128, 256, 512, 1024, 2048, 4096]
    B, n_trials = 2, 5
    results = []

    for kv_len in kv_lens:
        vanilla = VanillaMLA(cfg).cuda().half().eval()
        absorb = MatAbsorbMLA.from_vanilla(vanilla).cuda().half().eval()
        c_kv, k_pe = make_cache(B, kv_len, cfg)
        x = torch.randn(B, 1, cfg.hidden_size, device=DEVICE, dtype=DTYPE)

        for _ in range(3):
            _ = vanilla(x, c_kv, k_pe)
        torch.cuda.synchronize()
        tv = []
        for _ in range(n_trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = vanilla(x, c_kv, k_pe)
            torch.cuda.synchronize(); tv.append(time.perf_counter() - t0)
        ms_v = sum(tv) / len(tv) * 1000

        for _ in range(3):
            _ = absorb(x, c_kv, k_pe)
        torch.cuda.synchronize()
        ta = []
        for _ in range(n_trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = absorb(x, c_kv, k_pe)
            torch.cuda.synchronize(); ta.append(time.perf_counter() - t0)
        ms_a = sum(ta) / len(ta) * 1000

        mla_mem = (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * kv_len * B * 2 / 1024**2
        mha_mem = 2 * cfg.num_heads * cfg.qk_nope_head_dim * kv_len * B * 2 / 1024**2

        results.append({
            'kv_len': kv_len,
            'vanilla_ms': ms_v, 'absorb_ms': ms_a,
            'speedup': ms_v / ms_a,
            'mla_kv_mb': mla_mem, 'mha_kv_mb': mha_mem,
        })
        print(f"  kv={kv_len}: Vanilla {ms_v:.2f}ms | Absorb {ms_a:.2f}ms | Speedup {ms_v/ms_a:.2f}x | MHA mem {mha_mem:.1f}MB | MLA mem {mla_mem:.1f}MB")
        del vanilla, absorb, c_kv, k_pe, x; gc.collect(); torch.cuda.empty_cache()

    RESULTS['exp5'] = results


def exp6_batch_scaling():
    print("=" * 60)
    print("Exp6: Batch Size Scaling")
    print("=" * 60)
    cfg = MLAConfig()
    kv_len = 512
    batch_sizes = [1, 2, 4, 8]
    n_trials = 5
    results = []

    for B in batch_sizes:
        vanilla = VanillaMLA(cfg).cuda().half().eval()
        absorb = MatAbsorbMLA.from_vanilla(vanilla).cuda().half().eval()
        c_kv, k_pe = make_cache(B, kv_len, cfg)
        x = torch.randn(B, 1, cfg.hidden_size, device=DEVICE, dtype=DTYPE)

        for _ in range(3):
            _ = vanilla(x, c_kv, k_pe)
        torch.cuda.synchronize()
        tv = []
        for _ in range(n_trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = vanilla(x, c_kv, k_pe)
            torch.cuda.synchronize(); tv.append(time.perf_counter() - t0)
        ms_v = sum(tv) / len(tv) * 1000

        for _ in range(3):
            _ = absorb(x, c_kv, k_pe)
        torch.cuda.synchronize()
        ta = []
        for _ in range(n_trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = absorb(x, c_kv, k_pe)
            torch.cuda.synchronize(); ta.append(time.perf_counter() - t0)
        ms_a = sum(ta) / len(ta) * 1000

        results.append({
            'batch_size': B,
            'vanilla_ms': ms_v, 'absorb_ms': ms_a,
            'speedup': ms_v / ms_a,
            'vanilla_tokens_s': B / (ms_v / 1000),
            'absorb_tokens_s': B / (ms_a / 1000),
        })
        print(f"  B={B}: Vanilla {ms_v:.2f}ms | Absorb {ms_a:.2f}ms | Speedup {ms_v/ms_a:.2f}x")
        del vanilla, absorb, c_kv, k_pe, x; gc.collect(); torch.cuda.empty_cache()

    RESULTS['exp6'] = results


def exp7_attention_comparison():
    print("=" * 60)
    print("Exp7: MHA vs MQA vs GQA vs MLA Full Comparison")
    print("=" * 60)
    num_heads, head_dim, rope_dim, kv_lora_rank = 32, 64, 32, 256
    B, kv_len, n_trials = 4, 512, 5
    hidden = 2048

    def gqa_attention(q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = F.softmax(scores.float(), dim=-1).to(DTYPE)
        return torch.matmul(attn, v)

    attn_types = {
        'MHA': {'kv_heads': num_heads, 'per_token_bytes': 2 * num_heads * head_dim * 2},
        'GQA-4': {'kv_heads': 4, 'per_token_bytes': 2 * 4 * head_dim * 2},
        'GQA-8': {'kv_heads': 8, 'per_token_bytes': 2 * 8 * head_dim * 2},
        'MQA': {'kv_heads': 1, 'per_token_bytes': 2 * 1 * head_dim * 2},
        'MLA': {'kv_heads': 1, 'per_token_bytes': (kv_lora_rank + rope_dim) * 2},
    }

    results = []
    for name, info in attn_types.items():
        q = torch.randn(B, num_heads, kv_len, head_dim, device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, info['kv_heads'], kv_len, head_dim, device=DEVICE, dtype=DTYPE)
        v = torch.randn(B, info['kv_heads'], kv_len, head_dim, device=DEVICE, dtype=DTYPE)

        if info['kv_heads'] < num_heads:
            n_rep = num_heads // info['kv_heads']
            k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(B, num_heads, kv_len, head_dim)
            v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(B, num_heads, kv_len, head_dim)

        for _ in range(3):
            _ = gqa_attention(q, k, v)
        torch.cuda.synchronize()
        times = []
        for _ in range(n_trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = gqa_attention(q, k, v)
            torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        ms = sum(times) / len(times) * 1000

        total_mem = info['per_token_bytes'] * kv_len * B / 1024**2
        results.append({
            'name': name,
            'kv_heads': info['kv_heads'],
            'per_token_bytes': info['per_token_bytes'],
            'total_mem_mb': total_mem,
            'latency_ms': ms,
            'tokens_s': B * kv_len / (ms / 1000),
        })
        print(f"  {name}: {ms:.2f}ms, mem={total_mem:.1f}MB, per_token={info['per_token_bytes']}B")
        del q, k, v; gc.collect(); torch.cuda.empty_cache()

    cfg = MLAConfig()
    vanilla = VanillaMLA(cfg).cuda().half().eval()
    absorb = MatAbsorbMLA.from_vanilla(vanilla).cuda().half().eval()
    c_kv, k_pe = make_cache(B, kv_len, cfg)
    x = torch.randn(B, kv_len, cfg.hidden_size, device=DEVICE, dtype=DTYPE)

    for impl_name, model in [('MLA_Vanilla', vanilla), ('MLA_MatAbsorb', absorb)]:
        for _ in range(3):
            _ = model(x, c_kv, k_pe)
        torch.cuda.synchronize()
        times = []
        for _ in range(n_trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = model(x, c_kv, k_pe)
            torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        ms = sum(times) / len(times) * 1000
        mla_mem = (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * kv_len * B * 2 / 1024**2

        results.append({
            'name': impl_name,
            'per_token_bytes': (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * 2,
            'total_mem_mb': mla_mem,
            'latency_ms': ms,
            'tokens_s': B * kv_len / (ms / 1000),
        })
        print(f"  {impl_name}: {ms:.2f}ms, mem={mla_mem:.1f}MB")

    del vanilla, absorb, c_kv, k_pe, x; gc.collect(); torch.cuda.empty_cache()
    RESULTS['exp7'] = results


if __name__ == '__main__':
    print("MLA Experiment Suite")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"Dtype: {DTYPE}\n")
    warmup()
    exp1_memory()
    exp2_correctness()
    exp3_lora_rank()
    exp4_decode_latency()
    exp5_seq_scaling()
    exp6_batch_scaling()
    exp7_attention_comparison()
    out_path = os.path.join(DATA_DIR, 'mla_results.json')
    with open(out_path, 'w') as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print("Done!")
