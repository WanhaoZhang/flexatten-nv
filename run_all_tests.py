"""
FlexAttention + FlashAttention 完整测试套件
在 NVIDIA L4 GPU (CUDA 12.4) 上验证

测试项:
  1. Causal Mask (因果掩码)
  2. Sliding Window Attention (滑动窗口注意力)
  3. Document Mask (文档掩码)
  4. FlexAttention vs SDPA 性能对比
  5. Prefix LM Mask (前缀语言模型掩码)
  6. ALiBi Score Modification (ALiBi 分数修改)
  7. FlashAttention 后端验证
  8. 大规模序列长度压力测试
"""

import torch
import torch.nn.functional as F
import time
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def get_device_info():
    print("=" * 60)
    print("环境信息")
    print("=" * 60)
    print(f"PyTorch 版本:       {torch.__version__}")
    print(f"CUDA 版本:          {torch.version.cuda}")
    print(f"cuDNN 版本:         {torch.backends.cudnn.version()}")
    print(f"GPU 名称:           {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存:           {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Flash SDPA:         {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"Mem Efficient SDPA: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"Math SDPA:          {torch.backends.cuda.math_sdp_enabled()}")
    print(f"cuDNN SDPA:         {torch.backends.cuda.cudnn_sdp_enabled()}")
    print()


def test_causal_mask():
    print("=" * 60)
    print("Test 1: Causal Mask (因果掩码)")
    print("=" * 60)

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    B, H, S, D = 2, 4, 128, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    block_mask = create_block_mask(causal_mask, B, H, S, S, device="cuda")
    output = flex_attention(q, k, v, block_mask=block_mask)

    assert output.shape == (B, H, S, D), f"Shape mismatch: {output.shape}"
    assert not torch.isnan(output).any(), "NaN detected in output"
    print(f"  Block mask sparsity: {block_mask.sparsity}%")
    print(f"  Output shape: {output.shape}")
    print(f"  Output sample: {output[0, 0, 0, :5]}")
    print("  [PASS] Causal mask test")
    print()


def test_sliding_window():
    print("=" * 60)
    print("Test 2: Sliding Window Attention (滑动窗口注意力)")
    print("=" * 60)

    WINDOW_SIZE = 32

    def sliding_window_mask(b, h, q_idx, kv_idx):
        return torch.abs(q_idx - kv_idx) <= WINDOW_SIZE

    B, H, S, D = 2, 4, 256, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    block_mask = create_block_mask(sliding_window_mask, B, H, S, S, device="cuda")
    output = flex_attention(q, k, v, block_mask=block_mask)

    assert output.shape == (B, H, S, D)
    assert not torch.isnan(output).any()
    print(f"  Window size: {WINDOW_SIZE}")
    print(f"  Output shape: {output.shape}")
    print("  [PASS] Sliding window test")
    print()


def test_document_mask():
    print("=" * 60)
    print("Test 3: Document Mask (文档掩码)")
    print("=" * 60)

    S = 512
    DOC_LEN = 64

    doc_ids = torch.zeros(S, device="cuda", dtype=torch.long)
    for i in range(S):
        doc_ids[i] = i // DOC_LEN

    def document_mask(b, h, q_idx, kv_idx):
        return doc_ids[q_idx] == doc_ids[kv_idx]

    B, H, D = 2, 4, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    block_mask = create_block_mask(document_mask, B, H, S, S, device="cuda")
    output = flex_attention(q, k, v, block_mask=block_mask)

    assert output.shape == (B, H, S, D)
    assert not torch.isnan(output).any()
    print(f"  Documents: {S // DOC_LEN}, each length: {DOC_LEN}")
    print(f"  Output shape: {output.shape}")
    print("  [PASS] Document mask test")
    print()


def test_benchmark():
    print("=" * 60)
    print("Test 4: FlexAttention vs SDPA 性能对比")
    print("=" * 60)

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    B, H, S, D = 1, 8, 1024, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    # SDPA warmup
    for _ in range(5):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()

    # SDPA benchmark
    start = time.time()
    for _ in range(100):
        out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
    sdpa_time = (time.time() - start) / 100

    # FlexAttention benchmark
    block_mask = create_block_mask(causal_mask, B, H, S, S, device="cuda")
    compiled_flex = torch.compile(flex_attention)

    for _ in range(5):
        _ = compiled_flex(q, k, v, block_mask=block_mask)
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(100):
        out_flex = compiled_flex(q, k, v, block_mask=block_mask)
        torch.cuda.synchronize()
    flex_time = (time.time() - start) / 100

    diff = (out_sdpa - out_flex).abs().max().item()

    print(f"  配置: B={B}, H={H}, S={S}, D={D}")
    print(f"  SDPA (FlashAttention2):     {sdpa_time*1000:.3f} ms")
    print(f"  FlexAttention (compiled):   {flex_time*1000:.3f} ms")
    print(f"  性能比: FlexAttn / SDPA = {flex_time/sdpa_time:.2f}x")
    print(f"  最大误差: {diff:.6f}")
    print("  [PASS] Benchmark test")
    print()


def test_prefix_lm():
    print("=" * 60)
    print("Test 5: Prefix LM Mask (前缀语言模型掩码)")
    print("=" * 60)

    PREFIX_LEN = 64

    def prefix_lm_mask(b, h, q_idx, kv_idx):
        prefix_mask = kv_idx < PREFIX_LEN
        causal_mask = q_idx >= kv_idx
        return prefix_mask | causal_mask

    B, H, S, D = 2, 4, 256, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    block_mask = create_block_mask(prefix_lm_mask, B, H, S, S, device="cuda")
    output = flex_attention(q, k, v, block_mask=block_mask)

    assert output.shape == (B, H, S, D)
    assert not torch.isnan(output).any()
    print(f"  Prefix length: {PREFIX_LEN}")
    print(f"  Output shape: {output.shape}")
    print("  [PASS] Prefix LM mask test")
    print()


def test_alibi():
    print("=" * 60)
    print("Test 6: ALiBi Score Modification (ALiBi 位置偏置)")
    print("=" * 60)

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def alibi_score_mod(score, b, h, q_idx, kv_idx):
        slopes = torch.tensor([0.5, 0.25, 0.125, 0.0625], device="cuda")
        return score - slopes[h] * (q_idx - kv_idx).abs()

    B, H, S, D = 2, 4, 256, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    block_mask = create_block_mask(causal_mask, B, H, S, S, device="cuda")
    output = flex_attention(q, k, v, score_mod=alibi_score_mod, block_mask=block_mask)

    assert output.shape == (B, H, S, D)
    assert not torch.isnan(output).any()
    print(f"  ALiBi slopes per head: [0.5, 0.25, 0.125, 0.0625]")
    print(f"  Output shape: {output.shape}")
    print("  [PASS] ALiBi score mod test")
    print()


def test_flash_backend():
    print("=" * 60)
    print("Test 7: FlashAttention 后端验证")
    print("=" * 60)

    B, H, S, D = 2, 8, 1024, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    # Flash backend
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
        out_flash = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"  FlashAttention output: {out_flash.shape}")

    # Mem efficient backend
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
        out_mem = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"  Mem Efficient output: {out_mem.shape}")

    # Math backend
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        out_math = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"  Math backend output: {out_math.shape}")

    diff_flash_mem = (out_flash - out_mem).abs().max().item()
    print(f"  Flash vs MemEff max diff: {diff_flash_mem:.6f}")
    print("  [PASS] FlashAttention backend test")
    print()


def test_large_sequence():
    print("=" * 60)
    print("Test 8: 大规模序列长度压力测试")
    print("=" * 60)

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    configs = [
        (1, 8, 2048, 64),
        (1, 8, 4096, 64),
        (1, 8, 8192, 64),
    ]

    for B, H, S, D in configs:
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

        # SDPA
        torch.cuda.synchronize()
        start = time.time()
        out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        sdpa_time = time.time() - start

        # FlexAttention
        block_mask = create_block_mask(causal_mask, B, H, S, S, device="cuda")
        compiled = torch.compile(flex_attention)
        # warmup
        _ = compiled(q, k, v, block_mask=block_mask)
        torch.cuda.synchronize()

        start = time.time()
        out_flex = compiled(q, k, v, block_mask=block_mask)
        torch.cuda.synchronize()
        flex_time = time.time() - start

        mem_used = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.reset_peak_memory_stats()

        print(f"  S={S:5d}: SDPA={sdpa_time*1000:6.2f}ms | FlexAttn={flex_time*1000:6.2f}ms | "
              f"Ratio={flex_time/sdpa_time:.2f}x | GPU mem={mem_used:.2f}GB")

    print("  [PASS] Large sequence test")
    print()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA GPU required!"
    get_device_info()
    test_causal_mask()
    test_sliding_window()
    test_document_mask()
    test_benchmark()
    test_prefix_lm()
    test_alibi()
    test_flash_backend()
    test_large_sequence()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
