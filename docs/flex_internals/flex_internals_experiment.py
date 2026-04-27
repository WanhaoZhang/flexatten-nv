import torch
import torch.nn.functional as F
import time
import json
import os
import gc
import math
import sys

torch.manual_seed(42)
torch.cuda.manual_seed(42)

DTYPE = torch.float16
DEVICE = 'cuda'
RESULTS = {}
DATA_DIR = '../data'
FIGURE_DIR = '../docs/figures'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


def timed(fn, n_warmup=3, n_trials=10):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return out, sum(times) / len(times) * 1000


# ============================================================
# Exp1: BlockMask 深度解剖
# ============================================================
def exp1_block_mask_anatomy():
    print("=" * 60)
    print("Exp1: BlockMask 内部结构解剖")
    print("=" * 60)
    from torch.nn.attention.flex_attention import create_block_mask

    results = {}

    # 1a: Causal mask on small example (S=16)
    print("\n--- 1a: Causal Mask (S=16) ---")
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    bm = create_block_mask(causal_mask, 1, 1, 16, 16, device=DEVICE)
    print(f"  BLOCK_SIZE: {bm.BLOCK_SIZE}")
    print(f"  mask_mod: {bm.mask_mod.__name__}")

    dense = bm.to_dense()
    print(f"  dense shape: {dense.shape}")
    print(f"  dense:\n{dense[0,0].int().cpu()}")

    # Show internal representation
    print(f"  kv_num_blocks shape: {bm.kv_num_blocks.shape}, values: {bm.kv_num_blocks[0,0].tolist()}")
    print(f"  kv_indices shape: {bm.kv_indices.shape}")
    print(f"  kv_indices[0,0]: {bm.kv_indices[0,0].tolist()}")
    if bm.full_kv_num_blocks is not None:
        print(f"  full_kv_num_blocks: {bm.full_kv_num_blocks[0,0].tolist()}")
        print(f"  full_kv_indices[0,0]: {bm.full_kv_indices[0,0].tolist()}")

    sparsity = bm.sparsity()
    print(f"  sparsity: {sparsity:.1f}%")

    results['causal_s16'] = {
        'seq_len': 16,
        'block_size': bm.BLOCK_SIZE,
        'sparsity': sparsity,
        'dense': dense[0, 0].cpu().tolist(),
        'kv_num_blocks': bm.kv_num_blocks[0, 0].tolist(),
        'kv_indices_shape': list(bm.kv_indices.shape),
    }

    # 1b: Sliding window mask (S=32, window=4)
    print("\n--- 1b: Sliding Window Mask (S=32, window=4) ---")
    def sliding_window(window):
        def mask_fn(b, h, q_idx, kv_idx):
            return q_idx - kv_idx < window
        return mask_fn

    bm_sw = create_block_mask(sliding_window(4), 1, 1, 32, 32, device=DEVICE)
    dense_sw = bm_sw.to_dense()
    print(f"  BLOCK_SIZE: {bm_sw.BLOCK_SIZE}")
    print(f"  sparsity: {bm_sw.sparsity():.1f}%")

    results['sliding_s32'] = {
        'seq_len': 32,
        'window': 4,
        'block_size': bm_sw.BLOCK_SIZE,
        'sparsity': bm_sw.sparsity(),
    }

    # 1c: Block size experiment
    print("\n--- 1c: Block Size Investigation ---")
    seq_len = 256
    block_sizes_to_test = [16, 32, 64, 128]
    bs_results = []
    for bs in block_sizes_to_test:
        try:
            bm_bs = create_block_mask(causal_mask, 1, 1, seq_len, seq_len, device=DEVICE, BLOCK_SIZE=bs)
            actual_bs = bm_bs.BLOCK_SIZE
            print(f"  Requested BLOCK_SIZE={bs}, Actual={actual_bs}, Sparsity={bm_bs.sparsity():.1f}%")
            bs_results.append({
                'requested': bs, 'actual': actual_bs[0] if isinstance(actual_bs, tuple) else actual_bs,
                'sparsity': bm_bs.sparsity()
            })
        except Exception as e:
            print(f"  BLOCK_SIZE={bs}: Error: {e}")
            bs_results.append({'requested': bs, 'error': str(e)})

    results['block_size_sweep'] = bs_results

    RESULTS['exp1'] = results


# ============================================================
# Exp2: score_mod 编译追踪
# ============================================================
def exp2_score_mod_tracing():
    print("=" * 60)
    print("Exp2: score_mod 编译过程追踪")
    print("=" * 60)
    from torch.nn.attention.flex_attention import flex_attention

    B, H, S, D = 1, 1, 64, 32
    q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)

    results = {}

    # 2a: Identity (no modification)
    print("\n--- 2a: Identity score_mod ---")
    def identity_mod(score, b, h, q_idx, kv_idx):
        return score

    _, ms_identity = timed(lambda: flex_attention(q, k, v, score_mod=identity_mod))
    ref = F.scaled_dot_product_attention(q, k, v)
    out_fa = flex_attention(q, k, v, score_mod=identity_mod)
    err = (ref - out_fa).abs().max().item()
    print(f"  Identity: {ms_identity:.2f}ms, max_err vs SDPA: {err:.6e}")
    results['identity'] = {'latency_ms': ms_identity, 'max_err_vs_sdpa': err}

    # 2b: Causal via score_mod
    print("\n--- 2b: Causal score_mod ---")
    def causal_score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(q_idx >= kv_idx, score, float('-inf'))

    _, ms_causal_sm = timed(lambda: flex_attention(q, k, v, score_mod=causal_score_mod))
    causal_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out_causal = flex_attention(q, k, v, score_mod=causal_score_mod)
    err_c = (causal_ref - out_causal).abs().max().item()
    print(f"  Causal score_mod: {ms_causal_sm:.2f}ms, max_err vs SDPA_causal: {err_c:.6e}")
    results['causal_score_mod'] = {'latency_ms': ms_causal_sm, 'max_err_vs_sdpa': err_c}

    # 2c: Softcap
    print("\n--- 2c: Tanh softcap score_mod ---")
    def softcap_mod(cap):
        def mod(score, b, h, q_idx, kv_idx):
            return cap * torch.tanh(score / cap)
        return mod

    out_sc = flex_attention(q, k, v, score_mod=softcap_mod(50.0))
    _, ms_sc = timed(lambda: flex_attention(q, k, v, score_mod=softcap_mod(50.0)))
    print(f"  Softcap(50): {ms_sc:.2f}ms, output_range=[{out_sc.min().item():.4f}, {out_sc.max().item():.4f}]")
    results['softcap'] = {'latency_ms': ms_sc, 'cap': 50.0}

    # 2d: Relative position bias
    print("\n--- 2d: Relative position bias ---")
    def rel_pos_bias(score, b, h, q_idx, kv_idx):
        return score + 0.5 * (q_idx - kv_idx).float()

    _, ms_rpb = timed(lambda: flex_attention(q, k, v, score_mod=rel_pos_bias))
    print(f"  RelPosBias: {ms_rpb:.2f}ms")
    results['rel_pos_bias'] = {'latency_ms': ms_rpb}

    # 2e: ALiBi bias
    print("\n--- 2e: ALiBi bias ---")
    def alibi_mod(score, b, h, q_idx, kv_idx):
        slope = 2.0 ** (-8.0 * (h + 1) / H)
        return score - slope * (q_idx - kv_idx).float()

    _, ms_alibi = timed(lambda: flex_attention(q, k, v, score_mod=alibi_mod))
    print(f"  ALiBi: {ms_alibi:.2f}ms")
    results['alibi'] = {'latency_ms': ms_alibi}

    RESULTS['exp2'] = results


# ============================================================
# Exp3: BlockMask 稀疏性 vs 性能
# ============================================================
def exp3_sparsity_perf():
    print("=" * 60)
    print("Exp3: BlockMask 稀疏性 vs 性能")
    print("=" * 60)
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    B, H, D = 2, 8, 64
    seq_lens = [256, 512, 1024, 2048]
    results = []

    for S in seq_lens:
        q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
        v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)

        row = {'seq_len': S}

        # Dense attention (SDPA baseline)
        _, ms_sdpa = timed(lambda: F.scaled_dot_product_attention(q, k, v))
        row['sdpa_ms'] = ms_sdpa

        # Dense FlexAttention (no mask)
        _, ms_fa_dense = timed(lambda: flex_attention(q, k, v))
        row['fa_dense_ms'] = ms_fa_dense

        # Causal
        def causal_mask_fn(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        bm_causal = create_block_mask(causal_mask_fn, B, H, S, S, device=DEVICE)
        row['causal_sparsity'] = bm_causal.sparsity()
        _, ms_fa_causal = timed(lambda: flex_attention(q, k, v, block_mask=bm_causal))
        row['fa_causal_ms'] = ms_fa_causal

        _, ms_sdpa_causal = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        row['sdpa_causal_ms'] = ms_sdpa_causal

        # Sliding window (w=64)
        def sw_mask(b, h, q_idx, kv_idx):
            return q_idx - kv_idx < 64
        bm_sw = create_block_mask(sw_mask, B, H, S, S, device=DEVICE)
        row['sw_sparsity'] = bm_sw.sparsity()
        _, ms_fa_sw = timed(lambda: flex_attention(q, k, v, block_mask=bm_sw))
        row['fa_sw_ms'] = ms_fa_sw

        # Sliding window (w=128)
        def sw128_mask(b, h, q_idx, kv_idx):
            return q_idx - kv_idx < 128
        bm_sw128 = create_block_mask(sw128_mask, B, H, S, S, device=DEVICE)
        row['sw128_sparsity'] = bm_sw128.sparsity()
        _, ms_fa_sw128 = timed(lambda: flex_attention(q, k, v, block_mask=bm_sw128))
        row['fa_sw128_ms'] = ms_fa_sw128

        # Prefix (first 64 tokens visible to all)
        def prefix_mask(b, h, q_idx, kv_idx):
            return torch.logical_or(q_idx >= kv_idx, kv_idx < 64)
        bm_prefix = create_block_mask(prefix_mask, B, H, S, S, device=DEVICE)
        row['prefix_sparsity'] = bm_prefix.sparsity()
        _, ms_fa_prefix = timed(lambda: flex_attention(q, k, v, block_mask=bm_prefix))
        row['fa_prefix_ms'] = ms_fa_prefix

        results.append(row)
        print(f"  S={S}: SDPA={ms_sdpa:.2f}ms, FA_dense={ms_fa_dense:.2f}ms, "
              f"FA_causal={ms_fa_causal:.2f}ms (sp={bm_causal.sparsity():.0f}%), "
              f"FA_sw64={ms_fa_sw:.2f}ms (sp={bm_sw.sparsity():.0f}%), "
              f"FA_prefix={ms_fa_prefix:.2f}ms (sp={bm_prefix.sparsity():.0f}%)")

        del q, k, v, bm_causal, bm_sw, bm_sw128, bm_prefix
        gc.collect(); torch.cuda.empty_cache()

    RESULTS['exp3'] = results


# ============================================================
# Exp4: mask_mod + score_mod 组合追踪
# ============================================================
def exp4_mask_plus_score():
    print("=" * 60)
    print("Exp4: mask_mod + score_mod 组合效果")
    print("=" * 60)
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    B, H, S, D = 2, 4, 256, 64
    q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    results = []

    configs = [
        ('no_mod', None, None),
        ('causal_mask_only', None, lambda b, h, q_idx, kv_idx: q_idx >= kv_idx),
        ('causal_mask+softcap', softcap_mod(50.0), lambda b, h, q_idx, kv_idx: q_idx >= kv_idx),
        ('causal_mask+alibi', alibi_mod_gen(H), lambda b, h, q_idx, kv_idx: q_idx >= kv_idx),
        ('sw_mask_only', None, lambda b, h, q_idx, kv_idx: q_idx - kv_idx < 64),
        ('sw_mask+softcap', softcap_mod(50.0), lambda b, h, q_idx, kv_idx: q_idx - kv_idx < 64),
        ('prefix_mask', None, lambda b, h, q_idx, kv_idx: torch.logical_or(q_idx >= kv_idx, kv_idx < 32)),
    ]

    for name, smod, mmod in configs:
        bm = None
        if mmod is not None:
            bm = create_block_mask(mmod, B, H, S, S, device=DEVICE)

        sp = bm.sparsity() if bm else 0.0

        _, ms = timed(lambda: flex_attention(q, k, v, score_mod=smod, block_mask=bm))

        # Reference: SDPA dense
        _, ms_ref = timed(lambda: F.scaled_dot_product_attention(q, k, v))

        results.append({
            'name': name,
            'has_score_mod': smod is not None,
            'has_block_mask': mmod is not None,
            'sparsity': sp,
            'flex_attn_ms': ms,
            'overhead_vs_sdpa': ms / ms_ref,
        })
        print(f"  {name:30s}: {ms:.2f}ms, sparsity={sp:.0f}%, overhead={ms/ms_ref:.2f}x")

    RESULTS['exp4'] = results


def softcap_mod(cap):
    def mod(score, b, h, q_idx, kv_idx):
        return cap * torch.tanh(score / cap)
    return mod


def alibi_mod_gen(num_heads):
    def mod(score, b, h, q_idx, kv_idx):
        slope = 2.0 ** (-8.0 * (h + 1) / num_heads)
        return score - slope * (q_idx - kv_idx).float()
    return mod


# ============================================================
# Exp5: torch.compile 追踪 - 编译开销分析
# ============================================================
def exp5_compile_overhead():
    print("=" * 60)
    print("Exp5: torch.compile 编译开销分析")
    print("=" * 60)
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    B, H, D = 1, 4, 64

    configs = [
        ('no_mod', None, None),
        ('causal_score', lambda s, b, h, qi, ki: torch.where(qi >= ki, s, float('-inf')), None),
        ('causal_block', None, lambda b, h, qi, ki: qi >= ki),
    ]

    results = []
    for name, smod, mmod in configs:
        for S in [64, 256, 1024]:
            q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
            v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)

            bm = None
            if mmod is not None:
                bm = create_block_mask(mmod, B, H, S, S, device=DEVICE)

            # First call = compilation + execution
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out1 = flex_attention(q, k, v, score_mod=smod, block_mask=bm)
            torch.cuda.synchronize()
            first_call_ms = (time.perf_counter() - t0) * 1000

            # Subsequent calls = cached
            _, cached_ms = timed(lambda: flex_attention(q, k, v, score_mod=smod, block_mask=bm))

            # SDPA reference
            _, sdpa_ms = timed(lambda: F.scaled_dot_product_attention(q, k, v))

            results.append({
                'name': name, 'seq_len': S,
                'first_call_ms': first_call_ms,
                'cached_ms': cached_ms,
                'compile_overhead_ms': first_call_ms - cached_ms,
                'sdpa_ms': sdpa_ms,
                'overhead_ratio': first_call_ms / sdpa_ms,
                'cached_ratio': cached_ms / sdpa_ms,
            })
            print(f"  {name:20s} S={S:5d}: first={first_call_ms:.1f}ms, cached={cached_ms:.2f}ms, "
                  f"SDPA={sdpa_ms:.2f}ms, compile_overhead={first_call_ms-cached_ms:.1f}ms")

            del q, k, v, bm, out1
            gc.collect(); torch.cuda.empty_cache()

    RESULTS['exp5'] = results


# ============================================================
# Exp6: FlexAttention vs 手写 Triton vs SDPA 延迟对比
# ============================================================
def exp6_latency_showdown():
    print("=" * 60)
    print("Exp6: FlexAttention vs SDPA 延迟全对比")
    print("=" * 60)
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    B, H, D = 2, 8, 64
    seq_lens = [64, 128, 256, 512, 1024, 2048]
    results = []

    for S in seq_lens:
        q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
        v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)

        row = {'seq_len': S}

        # SDPA dense
        _, ms = timed(lambda: F.scaled_dot_product_attention(q, k, v))
        row['sdpa_dense'] = ms

        # SDPA causal
        _, ms = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        row['sdpa_causal'] = ms

        # Flex dense
        _, ms = timed(lambda: flex_attention(q, k, v))
        row['flex_dense'] = ms

        # Flex causal (score_mod)
        def causal_sm(s, b, h, qi, ki):
            return torch.where(qi >= ki, s, float('-inf'))
        _, ms = timed(lambda: flex_attention(q, k, v, score_mod=causal_sm))
        row['flex_causal_sm'] = ms

        # Flex causal (block_mask)
        def causal_mm(b, h, qi, ki):
            return qi >= ki
        bm = create_block_mask(causal_mm, B, H, S, S, device=DEVICE)
        _, ms = timed(lambda: flex_attention(q, k, v, block_mask=bm))
        row['flex_causal_bm'] = ms

        # Flex causal (both)
        _, ms = timed(lambda: flex_attention(q, k, v, score_mod=causal_sm, block_mask=bm))
        row['flex_causal_both'] = ms

        # Flex softcap
        def sc50(s, b, h, qi, ki):
            return 50.0 * torch.tanh(s / 50.0)
        _, ms = timed(lambda: flex_attention(q, k, v, score_mod=sc50))
        row['flex_softcap'] = ms

        # Flex sliding window (block_mask)
        def sw64(b, h, qi, ki):
            return qi - ki < 64
        bm_sw = create_block_mask(sw64, B, H, S, S, device=DEVICE)
        row['sw64_sparsity'] = bm_sw.sparsity()
        _, ms = timed(lambda: flex_attention(q, k, v, block_mask=bm_sw))
        row['flex_sw64'] = ms

        # Flex GQA
        q_gqa = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
        k_gqa = torch.randn(B, 2, S, D, device=DEVICE, dtype=DTYPE)
        v_gqa = torch.randn(B, 2, S, D, device=DEVICE, dtype=DTYPE)
        _, ms = timed(lambda: flex_attention(q_gqa, k_gqa, v_gqa, enable_gqa=True))
        row['flex_gqa'] = ms

        results.append(row)
        print(f"  S={S:5d}: SDPA={row['sdpa_dense']:.2f}ms, Flex_dense={row['flex_dense']:.2f}ms, "
              f"Flex_causal_bm={row['flex_causal_bm']:.2f}ms, Flex_sw64={row['flex_sw64']:.2f}ms, "
              f"Flex_gqa={row['flex_gqa']:.2f}ms")

        del q, k, v, bm, bm_sw, q_gqa, k_gqa, v_gqa
        gc.collect(); torch.cuda.empty_cache()

    RESULTS['exp6'] = results


# ============================================================
# Exp7: 逐步追踪 - 用小例子展示完整计算
# ============================================================
def exp7_step_by_step_trace():
    print("=" * 60)
    print("Exp7: 逐步计算追踪 (S=4, D=4)")
    print("=" * 60)
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    B, H, S, D = 1, 1, 4, 4
    torch.manual_seed(42)

    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)

    print(f"\n  Q:\n{q[0,0]}")
    print(f"\n  K:\n{k[0,0]}")
    print(f"\n  V:\n{v[0,0]}")

    # Step 1: Raw scores
    scale = 1.0 / math.sqrt(D)
    raw_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    print(f"\n  Step 1 - Raw QK^T scores (scaled by 1/sqrt({D})={scale:.4f}):")
    print(f"  {raw_scores[0,0]}")

    # Step 2: Causal mask
    causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool))
    print(f"\n  Step 2 - Causal mask:")
    print(f"  {causal_mask.int()}")

    masked_scores = raw_scores.masked_fill(~causal_mask, float('-inf'))
    print(f"\n  Step 3 - Masked scores:")
    print(f"  {masked_scores[0,0]}")

    # Step 3: Softmax
    attn_weights = F.softmax(masked_scores, dim=-1)
    print(f"\n  Step 4 - Attention weights (softmax):")
    print(f"  {attn_weights[0,0]}")

    # Step 4: Weighted sum
    output = torch.matmul(attn_weights, v)
    print(f"\n  Step 5 - Output (Attn @ V):")
    print(f"  {output[0,0]}")

    # Compare with FlexAttention
    def causal_mm(b, h, qi, ki):
        return qi >= ki

    bm = create_block_mask(causal_mm, B, H, S, S, device='cpu')

    q_cuda = q.cuda().half()
    k_cuda = k.cuda().half()
    v_cuda = v.cuda().half()

    out_flex = flex_attention(q_cuda, k_cuda, v_cuda, block_mask=bm.to('cuda'))
    print(f"\n  FlexAttention output (FP16):")
    print(f"  {out_flex[0,0].float()}")

    out_sdpa = F.scaled_dot_product_attention(q_cuda, k_cuda, v_cuda, is_causal=True)
    print(f"\n  SDPA output (FP16):")
    print(f"  {out_sdpa[0,0].float()}")

    # Also trace with score_mod for softcap
    def softcap_2(s, b, h, qi, ki):
        return 2.0 * torch.tanh(s / 2.0)

    out_sc = flex_attention(q_cuda, k_cuda, v_cuda, score_mod=softcap_2, block_mask=bm.to('cuda'))

    # Manual softcap trace
    print(f"\n  Step 2b - Softcap(2.0) applied to scores:")
    sc_scores = 2.0 * torch.tanh(raw_scores / 2.0)
    print(f"  Before cap: {raw_scores[0,0]}")
    print(f"  After  cap: {sc_scores[0,0]}")

    sc_masked = sc_scores.masked_fill(~causal_mask, float('-inf'))
    sc_attn = F.softmax(sc_masked, dim=-1)
    sc_output = torch.matmul(sc_attn, v)
    print(f"\n  Manual softcap output:")
    print(f"  {sc_output[0,0]}")
    print(f"\n  FlexAttention softcap output (FP16):")
    print(f"  {out_sc[0,0].float()}")

    RESULTS['exp7'] = {
        'seq_len': S, 'head_dim': D,
        'note': 'Step-by-step trace with S=4, D=4. See console output for detailed values.'
    }


# ============================================================
# Exp8: 不同注意力模式的性能剖析
# ============================================================
def exp8_pattern_perf_analysis():
    print("=" * 60)
    print("Exp8: 不同注意力模式性能剖析")
    print("=" * 60)
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    B, H, S, D = 2, 8, 1024, 64
    q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)

    patterns = {}

    # Dense
    _, ms = timed(lambda: flex_attention(q, k, v))
    patterns['dense'] = {'ms': ms, 'sparsity': 0, 'note': 'No mask, no score_mod'}

    # Causal
    def causal(b, h, qi, ki): return qi >= ki
    bm = create_block_mask(causal, B, H, S, S, device=DEVICE)
    _, ms = timed(lambda: flex_attention(q, k, v, block_mask=bm))
    patterns['causal'] = {'ms': ms, 'sparsity': bm.sparsity(), 'note': 'Lower triangular'}

    # Causal + Softcap
    def sc(s, b, h, qi, ki): return 50.0 * torch.tanh(s / 50.0)
    _, ms = timed(lambda: flex_attention(q, k, v, score_mod=sc, block_mask=bm))
    patterns['causal+softcap'] = {'ms': ms, 'sparsity': bm.sparsity()}

    # Sliding window 64
    def sw64(b, h, qi, ki): return qi - ki < 64
    bm64 = create_block_mask(sw64, B, H, S, S, device=DEVICE)
    _, ms = timed(lambda: flex_attention(q, k, v, block_mask=bm64))
    patterns['sliding_64'] = {'ms': ms, 'sparsity': bm64.sparsity()}

    # Sliding window 256
    def sw256(b, h, qi, ki): return qi - ki < 256
    bm256 = create_block_mask(sw256, B, H, S, S, device=DEVICE)
    _, ms = timed(lambda: flex_attention(q, k, v, block_mask=bm256))
    patterns['sliding_256'] = {'ms': ms, 'sparsity': bm256.sparsity()}

    # Prefix (32) + causal
    def prefix32(b, h, qi, ki): return torch.logical_or(qi >= ki, ki < 32)
    bm_p32 = create_block_mask(prefix32, B, H, S, S, device=DEVICE)
    _, ms = timed(lambda: flex_attention(q, k, v, block_mask=bm_p32))
    patterns['prefix32+causal'] = {'ms': ms, 'sparsity': bm_p32.sparsity()}

    # Prefix (128) + causal
    def prefix128(b, h, qi, ki): return torch.logical_or(qi >= ki, ki < 128)
    bm_p128 = create_block_mask(prefix128, B, H, S, S, device=DEVICE)
    _, ms = timed(lambda: flex_attention(q, k, v, block_mask=bm_p128))
    patterns['prefix128+causal'] = {'ms': ms, 'sparsity': bm_p128.sparsity()}

    # Dilated (every 4th)
    def dilated4(b, h, qi, ki): return (qi - ki) % 4 == 0
    bm_d4 = create_block_mask(dilated4, B, H, S, S, device=DEVICE)
    _, ms = timed(lambda: flex_attention(q, k, v, block_mask=bm_d4))
    patterns['dilated_4'] = {'ms': ms, 'sparsity': bm_d4.sparsity()}

    # SDPA references
    _, ms_sdpa_dense = timed(lambda: F.scaled_dot_product_attention(q, k, v))
    _, ms_sdpa_causal = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))

    RESULTS['exp8'] = {
        'patterns': patterns,
        'sdpa_dense_ms': ms_sdpa_dense,
        'sdpa_causal_ms': ms_sdpa_causal,
        'config': {'B': B, 'H': H, 'S': S, 'D': D}
    }

    print(f"\n  SDPA dense: {ms_sdpa_dense:.2f}ms | SDPA causal: {ms_sdpa_causal:.2f}ms")
    for name, info in patterns.items():
        print(f"  {name:20s}: {info['ms']:.2f}ms (sparsity={info['sparsity']:.0f}%, "
              f"overhead={info['ms']/ms_sdpa_dense:.2f}x vs SDPA_dense)")


if __name__ == '__main__':
    print("FlexAttention Internals Experiment Suite")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"Dtype: {DTYPE}\n")

    exp1_block_mask_anatomy()
    exp2_score_mod_tracing()
    exp3_sparsity_perf()
    exp4_mask_plus_score()
    exp5_compile_overhead()
    exp6_latency_showdown()
    exp7_step_by_step_trace()
    exp8_pattern_perf_analysis()

    out_path = os.path.join(DATA_DIR, 'flex_internals_results.json')
    with open(out_path, 'w') as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print("Done!")
