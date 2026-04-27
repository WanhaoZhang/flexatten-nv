import torch
import torch.nn.functional as F
import math
import time
import json
import os
import gc
from typing import Optional, List, Tuple

from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    BlockMask,
    _identity,
    noop_mask,
)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

DTYPE = torch.float16
DEVICE = 'cuda'
RESULTS = {}
FIGURE_DIR = '../docs/figures'
os.makedirs(FIGURE_DIR, exist_ok=True)

DATA_DIR = '../data'
os.makedirs(DATA_DIR, exist_ok=True)

def _cdiv(x, multiple):
    return (x + multiple - 1) // multiple


class PagedAttentionManager:
    def __init__(self, n_pages, page_size, max_batch_size, n_heads, head_dim, device='cuda'):
        self.n_pages = n_pages
        self.page_size = page_size
        self.max_batch_size = max_batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device

        self.page_table = -torch.ones(
            (max_batch_size, n_pages), dtype=torch.int64, device=device
        )
        self.capacity = torch.zeros(max_batch_size, dtype=torch.int64, device=device)
        self.empty_pages = list(range(n_pages - 1, -1, -1))
        self.physical_to_logical = -torch.ones(
            (max_batch_size, n_pages), dtype=torch.int64, device=device
        )

        self.k_cache = torch.zeros(
            (1, n_heads, n_pages * page_size, head_dim), dtype=DTYPE, device=device
        )
        self.v_cache = torch.zeros(
            (1, n_heads, n_pages * page_size, head_dim), dtype=DTYPE, device=device
        )

    def reserve(self, batch_idx, seq_len):
        if isinstance(batch_idx, int):
            batch_idx = torch.tensor([batch_idx], device=self.device)
        if isinstance(seq_len, int):
            seq_len = torch.tensor([seq_len], device=self.device)
        bi = batch_idx[0]
        sl = seq_len[0]
        if sl <= self.capacity[bi]:
            return
        num_pages = _cdiv(sl - self.capacity[bi], self.page_size)
        assert len(self.empty_pages) >= num_pages, \
            f"requested {num_pages} pages but only {len(self.empty_pages)} available"
        start_page = self.capacity[bi] // self.page_size
        end_page = start_page + num_pages
        allocated = torch.tensor(self.empty_pages[-num_pages:], device=self.device)
        self.empty_pages = self.empty_pages[:-num_pages]
        self.page_table[bi, start_page:end_page] = allocated
        self.physical_to_logical[bi, allocated] = torch.arange(
            start_page.item(), end_page.item(), device=self.device
        )
        self.capacity[bi] += num_pages * self.page_size

    def erase(self, batch_idx):
        if isinstance(batch_idx, int):
            batch_idx = torch.tensor(batch_idx, device=self.device)
        allocated_mask = self.page_table[batch_idx] != -1
        allocated_pages = self.page_table[batch_idx][allocated_mask]
        self.capacity[batch_idx] = 0
        self.empty_pages += allocated_pages.tolist()
        self.physical_to_logical[batch_idx][:, allocated_pages] = -1
        self.page_table[batch_idx] = -1

    def assign(self, batch_idx, input_pos, k_val, v_val):
        if isinstance(batch_idx, int):
            batch_idx = torch.tensor([batch_idx], device=self.device)
        B, H, S, D = k_val.shape
        logical_block_idx = input_pos // self.page_size
        logical_block_offset = input_pos % self.page_size
        physical_block_idx = torch.gather(
            self.page_table[batch_idx], 1, logical_block_idx.to(torch.int64)
        ).to(torch.int32)
        addr = (physical_block_idx * self.page_size + logical_block_offset).view(-1)
        k_flat = k_val.permute(1, 0, 2, 3).contiguous().view(1, H, B * S, D)
        v_flat = v_val.permute(1, 0, 2, 3).contiguous().view(1, H, B * S, D)
        self.k_cache[:, :, addr, :] = k_flat
        self.v_cache[:, :, addr, :] = v_flat

    def convert_logical_block_mask(self, block_mask, batch_idx=None):
        B, H, ROWS, MAX_BLOCKS = block_mask.kv_indices.shape
        if block_mask.BLOCK_SIZE[1] != self.page_size:
            raise RuntimeError(
                f"block_mask column size {block_mask.BLOCK_SIZE[1]} != page_size {self.page_size}"
            )
        device = block_mask.kv_indices.device
        if batch_idx is None:
            batch_idx = torch.arange(B, device=device)
        page_table = self.page_table[batch_idx]
        new_kv_num_blocks = block_mask.kv_num_blocks.clone()
        new_kv_indices = torch.zeros((B, H, ROWS, self.n_pages), dtype=torch.int32, device=device)
        new_kv_indices[:, :, :, :MAX_BLOCKS] = (
            torch.gather(page_table, 1, block_mask.kv_indices.view(B, -1).to(torch.int64))
            .view(block_mask.kv_indices.shape)
            .to(torch.int32)
        )
        new_full_kv_num_blocks = block_mask.full_kv_num_blocks.clone() if block_mask.full_kv_num_blocks is not None else None
        new_full_kv_indices = None
        if block_mask.full_kv_indices is not None:
            new_full_kv_indices = torch.zeros(
                (B, H, ROWS, self.n_pages), dtype=torch.int32, device=device
            )
            new_full_kv_indices[:, :, :, :MAX_BLOCKS] = (
                torch.gather(page_table, 1, block_mask.full_kv_indices.view(B, -1).to(torch.int64))
                .view(block_mask.full_kv_indices.shape)
                .to(torch.int32)
            )
        new_mask_mod = self._get_mask_mod(block_mask.mask_mod, batch_idx)
        return BlockMask.from_kv_blocks(
            new_kv_num_blocks, new_kv_indices,
            new_full_kv_num_blocks, new_full_kv_indices,
            block_mask.BLOCK_SIZE, new_mask_mod,
        )

    def _get_mask_mod(self, mask_mod=None, batch_idx=None):
        if mask_mod is None:
            mask_mod = noop_mask
        ptl = self.physical_to_logical[batch_idx] if batch_idx is not None else self.physical_to_logical
        def new_mask_mod(b, h, q_idx, physical_kv_idx):
            physical_kv_block = physical_kv_idx // self.page_size
            physical_kv_offset = physical_kv_idx % self.page_size
            logical_block_idx = ptl[b, physical_kv_block]
            logical_kv_idx = logical_block_idx * self.page_size + physical_kv_offset
            return torch.where(logical_block_idx >= 0, mask_mod(b, h, q_idx, logical_kv_idx), False)
        return new_mask_mod

    def _get_score_mod(self, score_mod=None, batch_idx=None):
        if score_mod is None:
            score_mod = _identity
        ptl = self.physical_to_logical[batch_idx] if batch_idx is not None else self.physical_to_logical
        def new_score_mod(score, b, h, q_idx, physical_kv_idx):
            physical_kv_block = physical_kv_idx // self.page_size
            physical_kv_offset = physical_kv_idx % self.page_size
            logical_block_idx = ptl[b, physical_kv_block]
            logical_kv_idx = logical_block_idx * self.page_size + physical_kv_offset
            return torch.where(
                logical_block_idx >= 0,
                score_mod(score, b, h, q_idx, logical_kv_idx),
                float('-inf'),
            )
        return new_score_mod

    def get_physical_kv(self):
        return self.k_cache, self.v_cache


def native_paged_attention(query, mgr, batch_indices, seq_lengths, mask_mod=None):
    B = len(batch_indices)
    D = query.shape[3]
    output = torch.zeros_like(query)
    for i, bi in enumerate(batch_indices):
        s = seq_lengths[i]
        q_i = query[i:i+1, :, :s, :]
        logical_blocks = _cdiv(s, mgr.page_size)
        phys_blocks = mgr.page_table[bi, :logical_blocks].tolist()
        k_parts, v_parts = [], []
        for pb in phys_blocks:
            start = pb * mgr.page_size
            end = start + mgr.page_size
            k_parts.append(mgr.k_cache[0, :, start:end, :])
            v_parts.append(mgr.v_cache[0, :, start:end, :])
        k_i = torch.cat(k_parts, dim=1)[:, :s, :]
        v_i = torch.cat(v_parts, dim=1)[:, :s, :]
        scores = torch.matmul(q_i, k_i.transpose(-2, -1)) / math.sqrt(D)
        if mask_mod is not None:
            q_indices = torch.arange(s, device=DEVICE).unsqueeze(1).expand(s, s)
            kv_indices = torch.arange(s, device=DEVICE).unsqueeze(0).expand(s, s)
            b_t = torch.zeros(s, s, dtype=torch.long, device=DEVICE)
            h_t = torch.zeros(s, s, dtype=torch.long, device=DEVICE)
            mask = mask_mod(b_t, h_t, q_indices, kv_indices)
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            causal = torch.ones(s, s, device=DEVICE, dtype=torch.bool).tril_()
            scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores.float(), dim=-1).to(DTYPE)
        output[i:i+1, :, :s, :] = torch.matmul(attn, v_i)
    return output


def flex_paged_attention(query, mgr, batch_indices, seq_lengths, mask_mod=None, score_mod=None):
    B = len(batch_indices)
    S_q = query.shape[2]
    page_size = mgr.page_size
    actual_mask_mod = mask_mod if mask_mod is not None else noop_mask
    actual_score_mod = score_mod if score_mod is not None else _identity
    def causal_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    final_mask_mod = causal_mod if mask_mod is None else actual_mask_mod
    batch_idx_tensor = torch.tensor(batch_indices, device=DEVICE)
    logical_bm = create_block_mask(
        final_mask_mod, B, 1, S_q, max(seq_lengths),
        device=DEVICE, BLOCK_SIZE=(page_size, page_size),
    )
    physical_bm = mgr.convert_logical_block_mask(logical_bm, batch_idx=batch_idx_tensor)
    new_score_mod = mgr._get_score_mod(actual_score_mod, batch_idx=batch_idx_tensor)
    k_p, v_p = mgr.get_physical_kv()
    k_exp = k_p.expand(B, -1, -1, -1)
    v_exp = v_p.expand(B, -1, -1, -1)
    return flex_attention(query, k_exp, v_exp, score_mod=new_score_mod, block_mask=physical_bm)


def padded_attention(q, k, v, mask_mod=None):
    B, H, S, D = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    if mask_mod is None:
        causal = torch.ones(S, S, device=DEVICE, dtype=torch.bool).tril_()
        scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))
    else:
        q_idx = torch.arange(S, device=DEVICE).unsqueeze(1).expand(S, S)
        kv_idx = torch.arange(S, device=DEVICE).unsqueeze(0).expand(S, S)
        b_t = torch.zeros(S, S, dtype=torch.long, device=DEVICE)
        h_t = torch.zeros(S, S, dtype=torch.long, device=DEVICE)
        mask = mask_mod(b_t, h_t, q_idx, kv_idx)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    attn = F.softmax(scores.float(), dim=-1).to(DTYPE)
    return torch.matmul(attn, v)


def warmup():
    q = torch.randn(1, 4, 64, 64, device=DEVICE, dtype=DTYPE)
    k = torch.randn(1, 4, 64, 64, device=DEVICE, dtype=DTYPE)
    v = torch.randn(1, 4, 64, 64, device=DEVICE, dtype=DTYPE)
    for _ in range(3):
        _ = padded_attention(q, k, v)
    torch.cuda.synchronize()


def exp1_memory_waste():
    print("=" * 60)
    print("Exp1: Memory Waste Visualization")
    print("=" * 60)
    seq_lengths = [100, 500, 2000, 8000]
    max_seq = max(seq_lengths)
    B = len(seq_lengths)
    H, D = 8, 64
    padded_mem = B * H * max_seq * D * 2 * 2
    actual_data = sum(H * s * D * 2 * 2 for s in seq_lengths)
    page_size = 128
    paged_mem = sum(H * _cdiv(s, page_size) * page_size * D * 2 * 2 for s in seq_lengths)
    padded_util = actual_data / padded_mem * 100
    paged_util = actual_data / paged_mem * 100
    print(f"  Seq lens: {seq_lengths}")
    print(f"  Padded: {padded_mem/1024**2:.2f} MB ({padded_util:.1f}%)")
    print(f"  Paged:  {paged_mem/1024**2:.2f} MB ({paged_util:.1f}%)")
    print(f"  Saved: {(1 - paged_mem/padded_mem)*100:.1f}%")
    RESULTS['exp1'] = {
        'seq_lengths': seq_lengths,
        'max_seq': max_seq,
        'padded_mem_mb': padded_mem / 1024**2,
        'paged_mem_mb': paged_mem / 1024**2,
        'actual_data_mb': actual_data / 1024**2,
        'padded_utilization': padded_util,
        'paged_utilization': paged_util,
        'memory_saved_pct': (1 - paged_mem / padded_mem) * 100,
        'page_size': page_size,
        'pages_per_seq': [_cdiv(s, page_size) for s in seq_lengths],
        'waste_per_seq_padded_mb': [(max_seq - s) * H * D * 2 * 2 / 1024**2 for s in seq_lengths],
        'waste_per_seq_paged_mb': [(_cdiv(s, page_size) * page_size - s) * H * D * 2 * 2 / 1024**2 for s in seq_lengths],
    }
    gc.collect(); torch.cuda.empty_cache()


def exp2_correctness():
    print("=" * 60)
    print("Exp2: Correctness (Native vs Flex)")
    print("=" * 60)
    B, H, D, page_size = 4, 4, 64, 128
    seq_lengths = [128, 256, 384, 512]
    max_kv = max(seq_lengths)
    n_pages = sum(_cdiv(s, page_size) for s in seq_lengths) + 10
    mgr = PagedAttentionManager(n_pages, page_size, B, H, D)
    q_list = []
    for i in range(B):
        s = seq_lengths[i]
        mgr.reserve(i, s)
        ip = torch.arange(s, device=DEVICE).unsqueeze(0)
        ki = torch.randn(1, H, s, D, device=DEVICE, dtype=DTYPE)
        vi = torch.randn(1, H, s, D, device=DEVICE, dtype=DTYPE)
        mgr.assign(torch.tensor([i], device=DEVICE), ip, ki, vi)
        qi = torch.randn(1, H, s, D, device=DEVICE, dtype=DTYPE)
        padded_qi = torch.zeros(1, H, max_kv, D, device=DEVICE, dtype=DTYPE)
        padded_qi[:, :, :s, :] = qi
        q_list.append(padded_qi)
    query = torch.cat(q_list, dim=0)
    batch_indices = list(range(B))

    # Reconstruct full K,V for ground truth
    k_full = torch.zeros(B, H, max_kv, D, device=DEVICE, dtype=DTYPE)
    v_full = torch.zeros(B, H, max_kv, D, device=DEVICE, dtype=DTYPE)
    for i in range(B):
        s = seq_lengths[i]
        logical_blocks = _cdiv(s, page_size)
        phys_blocks = mgr.page_table[i, :logical_blocks].tolist()
        offset = 0
        for pb in phys_blocks:
            start = pb * page_size
            end = start + page_size
            take = min(page_size, s - offset)
            k_full[i, :, offset:offset+take, :] = mgr.k_cache[0, :, start:start+take, :]
            v_full[i, :, offset:offset+take, :] = mgr.v_cache[0, :, start:start+take, :]
            offset += page_size

    output_padded = padded_attention(query, k_full, v_full)
    output_native = native_paged_attention(query, mgr, batch_indices, seq_lengths)
    output_flex = flex_paged_attention(query, mgr, batch_indices, seq_lengths)

    err_pn = (output_padded - output_native).abs().max().item()
    err_pf = (output_padded - output_flex).abs().max().item()
    err_nf = (output_native - output_flex).abs().max().item()
    print(f"  Padded vs Native: {err_pn:.6e}")
    print(f"  Padded vs Flex:   {err_pf:.6e}")
    print(f"  Native vs Flex:   {err_nf:.6e}")

    per_batch = []
    for i in range(B):
        e_n = (output_padded[i] - output_native[i]).abs().max().item()
        e_f = (output_padded[i] - output_flex[i]).abs().max().item()
        per_batch.append({'batch': i, 'seq_len': seq_lengths[i], 'native_err': e_n, 'flex_err': e_f})

    RESULTS['exp2'] = {
        'seq_lengths': seq_lengths,
        'padded_vs_native': err_pn,
        'padded_vs_flex': err_pf,
        'native_vs_flex': err_nf,
        'per_batch': per_batch,
    }
    del mgr, query, q_list, k_full, v_full, output_padded, output_native, output_flex
    gc.collect(); torch.cuda.empty_cache()


def exp3_memory_efficiency():
    print("=" * 60)
    print("Exp3: Memory Efficiency")
    print("=" * 60)
    H, D, page_size = 8, 64, 128
    max_seq_list = [512, 1024, 2048, 4096, 8192]
    results = []
    for max_seq in max_seq_list:
        for B in [4, 8, 16]:
            seq_lengths = [max_seq * (i + 1) // (B + 1) for i in range(B)]
            padded_mem = B * H * max_seq * D * 2 * 2
            actual_data = sum(H * s * D * 2 * 2 for s in seq_lengths)
            paged_mem = sum(H * _cdiv(s, page_size) * page_size * D * 2 * 2 for s in seq_lengths)
            padded_util = actual_data / padded_mem * 100
            paged_util = actual_data / paged_mem * 100
            results.append({
                'max_seq': max_seq, 'batch_size': B,
                'padded_mb': padded_mem / 1024**2,
                'paged_mb': paged_mem / 1024**2,
                'actual_mb': actual_data / 1024**2,
                'padded_util': padded_util,
                'paged_util': paged_util,
                'savings_pct': (1 - paged_mem / padded_mem) * 100,
            })
            print(f"  S={max_seq}, B={B}: Padded {padded_mem/1024**2:.1f}MB({padded_util:.0f}%) "
                  f"Paged {paged_mem/1024**2:.1f}MB({paged_util:.0f}%) Saved {(1-paged_mem/padded_mem)*100:.1f}%")
    RESULTS['exp3'] = results
    gc.collect(); torch.cuda.empty_cache()


def exp4_throughput():
    print("=" * 60)
    print("Exp4: Throughput Benchmark")
    print("=" * 60)
    H, D, page_size = 8, 64, 128
    batch_sizes = [1, 2, 4]
    seq_lengths_list = [256, 512, 1024]
    n_warmup, n_trials = 3, 10
    results = []

    for B in batch_sizes:
        for S in seq_lengths_list:
            if B * H * S * D * 4 * 2 > 16 * 1024**3:
                continue
            seq_lens = [S] * B

            # Padded
            q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            for _ in range(n_warmup):
                _ = padded_attention(q, k, v)
            torch.cuda.synchronize()
            tp = []
            for _ in range(n_trials):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = padded_attention(q, k, v)
                torch.cuda.synchronize()
                tp.append(time.perf_counter() - t0)
            ms_p = sum(tp) / len(tp) * 1000
            del q, k, v; gc.collect(); torch.cuda.empty_cache()

            # Native paged
            np_total = B * _cdiv(S, page_size) + 10
            mgr = PagedAttentionManager(np_total, page_size, B, H, D)
            ql = []
            for i in range(B):
                mgr.reserve(i, S)
                ip = torch.arange(S, device=DEVICE).unsqueeze(0)
                ki = torch.randn(1, H, S, D, device=DEVICE, dtype=DTYPE)
                vi = torch.randn(1, H, S, D, device=DEVICE, dtype=DTYPE)
                mgr.assign(torch.tensor([i], device=DEVICE), ip, ki, vi)
                ql.append(torch.randn(1, H, S, D, device=DEVICE, dtype=DTYPE))
            query = torch.cat(ql, dim=0)
            bi = list(range(B))
            for _ in range(n_warmup):
                _ = native_paged_attention(query, mgr, bi, seq_lens)
            torch.cuda.synchronize()
            tn = []
            for _ in range(n_trials):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = native_paged_attention(query, mgr, bi, seq_lens)
                torch.cuda.synchronize()
                tn.append(time.perf_counter() - t0)
            ms_n = sum(tn) / len(tn) * 1000

            # Flex paged
            for _ in range(n_warmup):
                _ = flex_paged_attention(query, mgr, bi, seq_lens)
            torch.cuda.synchronize()
            tf = []
            for _ in range(n_trials):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = flex_paged_attention(query, mgr, bi, seq_lens)
                torch.cuda.synchronize()
                tf.append(time.perf_counter() - t0)
            ms_f = sum(tf) / len(tf) * 1000

            results.append({
                'batch_size': B, 'seq_len': S,
                'padded_ms': ms_p, 'padded_tokens_s': B * S / (ms_p / 1000),
                'native_ms': ms_n, 'native_tokens_s': B * S / (ms_n / 1000),
                'flex_ms': ms_f, 'flex_tokens_s': B * S / (ms_f / 1000),
                'native_vs_padded': ms_n / ms_p, 'flex_vs_padded': ms_f / ms_p,
            })
            print(f"  B={B}, S={S}: Padded {ms_p:.1f}ms | Native {ms_n:.1f}ms | Flex {ms_f:.1f}ms")
            del mgr, query, ql; gc.collect(); torch.cuda.empty_cache()

    RESULTS['exp4'] = results


def exp5_page_size():
    print("=" * 60)
    print("Exp5: Sequence Length Scaling (page_size=128 fixed)")
    print("=" * 60)
    B, H, D, page_size = 4, 8, 64, 128
    seq_len_list = [128, 256, 512, 1024, 2048]
    n_warmup, n_trials = 3, 10
    results = []
    for S in seq_len_list:
        seq_lens = [S] * B
        np_total = B * _cdiv(S, page_size) + 10
        mgr = PagedAttentionManager(np_total, page_size, B, H, D)
        ql = []
        for i in range(B):
            mgr.reserve(i, S)
            ip = torch.arange(S, device=DEVICE).unsqueeze(0)
            ki = torch.randn(1, H, S, D, device=DEVICE, dtype=DTYPE)
            vi = torch.randn(1, H, S, D, device=DEVICE, dtype=DTYPE)
            mgr.assign(torch.tensor([i], device=DEVICE), ip, ki, vi)
            ql.append(torch.randn(1, H, S, D, device=DEVICE, dtype=DTYPE))
        query = torch.cat(ql, dim=0)
        bi = list(range(B))

        # Padded baseline
        q_pad = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
        k_pad = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
        v_pad = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
        for _ in range(n_warmup):
            _ = padded_attention(q_pad, k_pad, v_pad)
        torch.cuda.synchronize()
        tp = []
        for _ in range(n_trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = padded_attention(q_pad, k_pad, v_pad)
            torch.cuda.synchronize(); tp.append(time.perf_counter() - t0)
        ms_p = sum(tp) / len(tp) * 1000
        del q_pad, k_pad, v_pad; gc.collect(); torch.cuda.empty_cache()

        # Native paged
        for _ in range(n_warmup):
            _ = native_paged_attention(query, mgr, bi, seq_lens)
        torch.cuda.synchronize()
        tn = []
        for _ in range(n_trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = native_paged_attention(query, mgr, bi, seq_lens)
            torch.cuda.synchronize(); tn.append(time.perf_counter() - t0)
        ms_n = sum(tn) / len(tn) * 1000

        # Flex paged
        for _ in range(n_warmup):
            _ = flex_paged_attention(query, mgr, bi, seq_lens)
        torch.cuda.synchronize()
        tf = []
        for _ in range(n_trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = flex_paged_attention(query, mgr, bi, seq_lens)
            torch.cuda.synchronize(); tf.append(time.perf_counter() - t0)
        ms_f = sum(tf) / len(tf) * 1000

        results.append({
            'seq_len': S, 'pages_per_seq': _cdiv(S, page_size),
            'padded_ms': ms_p, 'native_ms': ms_n, 'flex_ms': ms_f,
            'padded_tokens_s': B * S / (ms_p / 1000),
            'native_tokens_s': B * S / (ms_n / 1000),
            'flex_tokens_s': B * S / (ms_f / 1000),
        })
        print(f"  S={S}: Padded {ms_p:.1f}ms | Native {ms_n:.1f}ms | Flex {ms_f:.1f}ms")
        del mgr, query, ql; gc.collect(); torch.cuda.empty_cache()

    RESULTS['exp5'] = results


def exp6_variable_length():
    print("=" * 60)
    print("Exp6: Variable-Length Sequence Latency")
    print("=" * 60)
    H, D, page_size = 8, 64, 128
    n_trials = 5
    distributions = {
        'uniform': [128, 256, 384, 512, 640, 768, 896, 1024],
        'long_tail': [128, 128, 256, 256, 384, 512, 512, 1024],
        'bimodal': [128, 128, 256, 256, 768, 896, 896, 1024],
    }
    results = {}
    for dist_name, seq_lens in distributions.items():
        B = len(seq_lens)
        max_s = max(seq_lens)
        # Padded
        q = torch.randn(B, H, max_s, D, device=DEVICE, dtype=DTYPE)
        k_f = torch.zeros(B, H, max_s, D, device=DEVICE, dtype=DTYPE)
        v_f = torch.zeros(B, H, max_s, D, device=DEVICE, dtype=DTYPE)
        for i in range(B):
            s = seq_lens[i]
            k_f[i, :, :s, :] = torch.randn(H, s, D, device=DEVICE, dtype=DTYPE)
            v_f[i, :, :s, :] = torch.randn(H, s, D, device=DEVICE, dtype=DTYPE)

        def padded_per_seq(q, k, v, sl):
            out = torch.zeros_like(q)
            for i in range(len(sl)):
                s = sl[i]
                out[i:i+1, :, :s, :] = padded_attention(
                    q[i:i+1, :, :s, :], k[i:i+1, :, :s, :], v[i:i+1, :, :s, :])
            return out

        for _ in range(3): _ = padded_per_seq(q, k_f, v_f, seq_lens)
        torch.cuda.synchronize()
        tp = []
        for _ in range(n_trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = padded_per_seq(q, k_f, v_f, seq_lens)
            torch.cuda.synchronize(); tp.append(time.perf_counter() - t0)
        ms_p = sum(tp) / len(tp) * 1000
        del q, k_f, v_f; gc.collect(); torch.cuda.empty_cache()

        # Paged
        np_total = sum(_cdiv(s, page_size) for s in seq_lens) + 10
        mgr = PagedAttentionManager(np_total, page_size, B, H, D)
        ql = []
        bi = list(range(B))
        max_s_local = max(seq_lens)
        for i in range(B):
            s = seq_lens[i]
            mgr.reserve(i, s)
            ip = torch.arange(s, device=DEVICE).unsqueeze(0)
            ki = torch.randn(1, H, s, D, device=DEVICE, dtype=DTYPE)
            vi = torch.randn(1, H, s, D, device=DEVICE, dtype=DTYPE)
            mgr.assign(torch.tensor([i], device=DEVICE), ip, ki, vi)
            qi = torch.randn(1, H, s, D, device=DEVICE, dtype=DTYPE)
            padded_qi = torch.zeros(1, H, max_s_local, D, device=DEVICE, dtype=DTYPE)
            padded_qi[:, :, :s, :] = qi
            ql.append(padded_qi)
        query = torch.cat(ql, dim=0)

        for _ in range(3): _ = native_paged_attention(query, mgr, bi, seq_lens)
        torch.cuda.synchronize()
        tn = []
        for _ in range(n_trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = native_paged_attention(query, mgr, bi, seq_lens)
            torch.cuda.synchronize(); tn.append(time.perf_counter() - t0)
        ms_n = sum(tn) / len(tn) * 1000

        for _ in range(3): _ = flex_paged_attention(query, mgr, bi, seq_lens)
        torch.cuda.synchronize()
        tf = []
        for _ in range(n_trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = flex_paged_attention(query, mgr, bi, seq_lens)
            torch.cuda.synchronize(); tf.append(time.perf_counter() - t0)
        ms_f = sum(tf) / len(tf) * 1000

        results[dist_name] = {
            'seq_lengths': seq_lens,
            'padded_ms': ms_p, 'native_ms': ms_n, 'flex_ms': ms_f,
            'padded_mem_mb': B * H * max_s * D * 2 * 2 / 1024**2,
            'paged_mem_mb': sum(H * _cdiv(s, page_size) * page_size * D * 2 * 2 for s in seq_lens) / 1024**2,
        }
        print(f"  {dist_name}: Padded {ms_p:.1f}ms | Native {ms_n:.1f}ms | Flex {ms_f:.1f}ms")
        del mgr, query, ql; gc.collect(); torch.cuda.empty_cache()

    RESULTS['exp6'] = results


def exp7_mask_combination():
    print("=" * 60)
    print("Exp7: Mask + Paged Attention")
    print("=" * 60)
    B, H, D, S, page_size = 4, 4, 64, 256, 128
    seq_lens = [S] * B
    n_trials = 5

    def causal_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    def sliding_window_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx <= 128)
    def prefix_lm_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) | (kv_idx < 64)

    masks = {'causal': causal_mod, 'sliding_window': sliding_window_mod, 'prefix_lm': prefix_lm_mod}
    results = []
    for mask_name, mask_mod in masks.items():
        np_total = B * _cdiv(S, page_size) + 10
        mgr = PagedAttentionManager(np_total, page_size, B, H, D)
        ql = []
        for i in range(B):
            mgr.reserve(i, S)
            ip = torch.arange(S, device=DEVICE).unsqueeze(0)
            ki = torch.randn(1, H, S, D, device=DEVICE, dtype=DTYPE)
            vi = torch.randn(1, H, S, D, device=DEVICE, dtype=DTYPE)
            mgr.assign(torch.tensor([i], device=DEVICE), ip, ki, vi)
            ql.append(torch.randn(1, H, S, D, device=DEVICE, dtype=DTYPE))
        query = torch.cat(ql, dim=0)
        bi = list(range(B))

        for _ in range(3): _ = native_paged_attention(query, mgr, bi, seq_lens, mask_mod=mask_mod)
        torch.cuda.synchronize()
        tn = []
        for _ in range(n_trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = native_paged_attention(query, mgr, bi, seq_lens, mask_mod=mask_mod)
            torch.cuda.synchronize(); tn.append(time.perf_counter() - t0)
        ms_n = sum(tn) / len(tn) * 1000

        for _ in range(3): _ = flex_paged_attention(query, mgr, bi, seq_lens, mask_mod=mask_mod)
        torch.cuda.synchronize()
        tf = []
        for _ in range(n_trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            _ = flex_paged_attention(query, mgr, bi, seq_lens, mask_mod=mask_mod)
            torch.cuda.synchronize(); tf.append(time.perf_counter() - t0)
        ms_f = sum(tf) / len(tf) * 1000

        results.append({
            'mask_type': mask_name, 'native_ms': ms_n, 'flex_ms': ms_f,
            'flex_vs_native': ms_f / ms_n,
        })
        print(f"  {mask_name}: Native {ms_n:.1f}ms | Flex {ms_f:.1f}ms | ratio {ms_f/ms_n:.2f}x")
        del mgr, query, ql; gc.collect(); torch.cuda.empty_cache()

    RESULTS['exp7'] = results


if __name__ == '__main__':
    print("Paged Attention Experiment Suite")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"Dtype: {DTYPE}\n")
    warmup()
    exp1_memory_waste()
    exp2_correctness()
    exp3_memory_efficiency()
    exp4_throughput()
    exp5_page_size()
    exp6_variable_length()
    exp7_mask_combination()
    out_path = os.path.join(DATA_DIR, 'paged_attention_results.json')
    with open(out_path, 'w') as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print("Done!")
