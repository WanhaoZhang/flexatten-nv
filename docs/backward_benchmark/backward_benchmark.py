"""
Project 1: FlexAttention Backward Benchmark
============================================
Measures forward/backward latency, peak memory, and compilation overhead
for 4 mask patterns across multiple sequence lengths.

Environment: NVIDIA L4 (24GB) | PyTorch 2.6.0+cu124 | Triton 3.2.0
"""

import torch
import torch.nn.functional as F
import time
import json
import os
import sys
import gc
from collections import defaultdict
from contextlib import contextmanager

# Check FlexAttention availability
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

torch.set_default_device('cuda')
torch.manual_seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# Mask Definitions
# ============================================================

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def sliding_window_mask(window_size):
    def _mask(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx < window_size)
    return _mask

def prefix_lm_mask(prefix_len):
    def _mask(b, h, q_idx, kv_idx):
        # Prefix tokens can attend to all prefix tokens (bidirectional)
        # Non-prefix tokens attend causally + can attend to all prefix tokens
        is_prefix_kv = kv_idx < prefix_len
        causal = q_idx >= kv_idx
        return is_prefix_kv | causal
    return _mask

def doc_packing_mask(doc_boundary):
    """Document packing: tokens in same doc can attend to each other + causal within doc."""
    def _mask(b, h, q_idx, kv_idx):
        # Same document (same boundary segment)
        same_doc = doc_boundary[q_idx] == doc_boundary[kv_idx]
        causal = q_idx >= kv_idx
        return same_doc & causal
    return _mask

# Score mods for additional analysis
def softcap_score(cap):
    def _score(score, b, h, q_idx, kv_idx):
        return cap * torch.tanh(score / cap)
    return _score

# ============================================================
# Benchmark Utilities
# ============================================================

@contextmanager
def cuda_timer():
    """Context manager for precise CUDA timing."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield start, end
    end.record()
    torch.cuda.synchronize()

def measure_compilation_overhead(fn, *args, warmup=1, **kwargs):
    """Measure first-call (compilation) vs steady-state latency."""
    # First call - includes JIT compilation
    torch.cuda.synchronize()
    compile_start = time.perf_counter()
    result = fn(*args, **kwargs)
    if isinstance(result, torch.Tensor) and result.requires_grad:
        result.backward(torch.ones_like(result))
    torch.cuda.synchronize()
    compile_time = time.perf_counter() - compile_start

    # Warmup
    for _ in range(warmup):
        r = fn(*args, **kwargs)
        if isinstance(r, torch.Tensor) and r.requires_grad:
            r.backward(torch.ones_like(r))

    # Steady state measurement
    torch.cuda.synchronize()
    steady_start = time.perf_counter()
    for _ in range(5):
        r = fn(*args, **kwargs)
        if isinstance(r, torch.Tensor) and r.requires_grad:
            r.backward(torch.ones_like(r))
    torch.cuda.synchronize()
    steady_time = (time.perf_counter() - steady_start) / 5

    return compile_time, steady_time

def get_peak_memory():
    """Get current peak memory in MB."""
    return torch.cuda.max_memory_allocated() / 1024 / 1024

def reset_memory_stats():
    """Reset CUDA memory stats."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

# ============================================================
# Experiment 1: Forward + Backward Latency by Mask Type
# ============================================================

def experiment1_latency_by_mask():
    """Exp1: Measure forward + backward latency for 4 mask types."""
    print("\n" + "="*60)
    print("Experiment 1: Forward + Backward Latency by Mask Type")
    print("="*60)

    seq_lengths = [512, 1024, 2048, 4096]
    batch_size = 1
    num_heads = 32
    head_dim = 64
    embed_dim = num_heads * head_dim

    mask_configs = [
        ("Causal", causal_mask, {}),
        ("SlidingWindow_256", sliding_window_mask(256), {}),
        ("SlidingWindow_512", sliding_window_mask(512), {}),
        ("PrefixLM_128", prefix_lm_mask(128), {}),
        ("PrefixLM_256", prefix_lm_mask(256), {}),
    ]

    results = []

    for seq_len in seq_lengths:
        print(f"\n--- seq_len={seq_len} ---")
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda',
                        dtype=torch.float16, requires_grad=True)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda',
                        dtype=torch.float16, requires_grad=True)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda',
                        dtype=torch.float16, requires_grad=True)

        for mask_name, mask_fn, mask_kwargs in mask_configs:
            reset_memory_stats()

            try:
                block_mask = create_block_mask(mask_fn, batch_size, num_heads, seq_len, seq_len,
                                               device='cuda', _compile=True)

                # Measure forward
                torch.cuda.synchronize()
                fwd_start = torch.cuda.Event(enable_timing=True)
                fwd_end = torch.cuda.Event(enable_timing=True)

                fwd_start.record()
                out = flex_attention(q, k, v, block_mask=block_mask)
                fwd_end.record()
                torch.cuda.synchronize()
                fwd_ms = fwd_start.elapsed_time(fwd_end)

                fwd_peak = get_peak_memory()

                # Measure backward
                grad_out = torch.ones_like(out)
                reset_memory_stats()

                bwd_start = torch.cuda.Event(enable_timing=True)
                bwd_end = torch.cuda.Event(enable_timing=True)
                bwd_start.record()
                out.backward(grad_out)
                bwd_end.record()
                torch.cuda.synchronize()
                bwd_ms = bwd_start.elapsed_time(bwd_end)

                bwd_peak = get_peak_memory()
                total_peak = max(fwd_peak, bwd_peak)

                result = {
                    "mask": mask_name,
                    "seq_len": seq_len,
                    "forward_ms": round(fwd_ms, 2),
                    "backward_ms": round(bwd_ms, 2),
                    "total_ms": round(fwd_ms + bwd_ms, 2),
                    "fwd_peak_mb": round(fwd_peak, 1),
                    "bwd_peak_mb": round(bwd_peak, 1),
                    "total_peak_mb": round(total_peak, 1),
                }
                results.append(result)
                print(f"  {mask_name:25s} | fwd={fwd_ms:7.2f}ms | bwd={bwd_ms:7.2f}ms | peak={total_peak:8.1f}MB")

                del out, block_mask
                q.grad = None; k.grad = None; v.grad = None

            except torch.cuda.OutOfMemoryError:
                print(f"  {mask_name:25s} | OOM at seq_len={seq_len}")
                results.append({
                    "mask": mask_name, "seq_len": seq_len,
                    "forward_ms": None, "backward_ms": None, "total_ms": None,
                    "fwd_peak_mb": None, "bwd_peak_mb": None, "total_peak_mb": None,
                    "oom": True
                })
                reset_memory_stats()
            except Exception as e:
                print(f"  {mask_name:25s} | Error: {e}")
                results.append({
                    "mask": mask_name, "seq_len": seq_len,
                    "error": str(e)
                })

        del q, k, v
        reset_memory_stats()

    return results

# ============================================================
# Experiment 2: Compilation Overhead Analysis
# ============================================================

def experiment2_compilation_overhead():
    """Exp2: Measure JIT compilation overhead vs steady-state."""
    print("\n" + "="*60)
    print("Experiment 2: Compilation Overhead Analysis")
    print("="*60)

    seq_len = 1024
    batch_size = 1
    num_heads = 32
    head_dim = 64

    results = []

    for mask_name, mask_fn in [
        ("Causal", causal_mask),
        ("SlidingWindow_256", sliding_window_mask(256)),
    ]:
        print(f"\n  {mask_name}:")

        reset_memory_stats()
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda',
                        dtype=torch.float16, requires_grad=True)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda',
                        dtype=torch.float16, requires_grad=True)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda',
                        dtype=torch.float16, requires_grad=True)

        block_mask = create_block_mask(mask_fn, batch_size, num_heads, seq_len, seq_len,
                                       device='cuda', _compile=True)

        # First call (includes compile)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = flex_attention(q, k, v, block_mask=block_mask)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        first_call_ms = (time.perf_counter() - t0) * 1000

        q.grad = None; k.grad = None; v.grad = None

        # Warmup 3
        for _ in range(3):
            out = flex_attention(q, k, v, block_mask=block_mask)
            out.sum().backward()
            q.grad = None; k.grad = None; v.grad = None

        # Steady state (10 iterations)
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = flex_attention(q, k, v, block_mask=block_mask)
            out.sum().backward()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            q.grad = None; k.grad = None; v.grad = None

        steady_ms = sum(times) / len(times)
        compile_overhead_ms = first_call_ms - steady_ms

        result = {
            "mask": mask_name,
            "seq_len": seq_len,
            "first_call_ms": round(first_call_ms, 2),
            "steady_state_ms": round(steady_ms, 2),
            "compile_overhead_ms": round(compile_overhead_ms, 2),
            "overhead_ratio": round(compile_overhead_ms / steady_ms, 2),
            "steady_fwd_bwd_ms": [round(t, 2) for t in times],
        }
        results.append(result)
        print(f"    First call: {first_call_ms:.1f}ms | Steady: {steady_ms:.1f}ms | Overhead: {compile_overhead_ms:.1f}ms ({compile_overhead_ms/steady_ms:.1f}x)")

        del q, k, v, out, block_mask
        reset_memory_stats()

    return results

# ============================================================
# Experiment 3: Document Packing Mask with varied segment counts
# ============================================================

def experiment3_doc_packing():
    """Exp3: Document packing with different segment counts and seq lengths."""
    print("\n" + "="*60)
    print("Experiment 3: Document Packing Performance")
    print("="*60)

    seq_lengths = [512, 1024, 2048]
    batch_size = 1
    num_heads = 32
    head_dim = 64

    # Vary number of documents packed into same sequence
    doc_counts = [2, 4, 8, 16]

    results = []

    for seq_len in seq_lengths:
        for n_docs in doc_counts:
            if seq_len < n_docs * 32:
                continue  # skip if too many docs for the seq length

            doc_len = seq_len // n_docs
            # Create document boundary tensor
            boundary = torch.zeros(seq_len, dtype=torch.int32, device='cuda')
            for i in range(n_docs):
                boundary[i * doc_len : (i + 1) * doc_len] = i
            # Handle remainder
            boundary[n_docs * doc_len:] = n_docs - 1

            mask_fn = doc_packing_mask(boundary)

            reset_memory_stats()

            try:
                q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda',
                                dtype=torch.float16, requires_grad=True)
                k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda',
                                dtype=torch.float16, requires_grad=True)
                v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda',
                                dtype=torch.float16, requires_grad=True)

                block_mask = create_block_mask(mask_fn, batch_size, num_heads, seq_len, seq_len,
                                               device='cuda', _compile=True)

                # Warmup
                out = flex_attention(q, k, v, block_mask=block_mask)
                out.sum().backward()
                q.grad = None; k.grad = None; v.grad = None

                # Measure
                fwd_times = []
                bwd_times = []
                for _ in range(5):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    out = flex_attention(q, k, v, block_mask=block_mask)
                    torch.cuda.synchronize()
                    fwd_times.append((time.perf_counter() - t0) * 1000)

                    reset_memory_stats()
                    t0 = time.perf_counter()
                    out.sum().backward()
                    torch.cuda.synchronize()
                    bwd_times.append((time.perf_counter() - t0) * 1000)
                    peak_mb = get_peak_memory()

                    q.grad = None; k.grad = None; v.grad = None

                # Calculate theoretical sparsity
                total_blocks = (seq_len // 128) ** 2  # approximate
                attended_blocks = n_docs * (doc_len // 128) ** 2  # each doc attends to itself
                sparsity = 1 - attended_blocks / max(total_blocks, 1)

                result = {
                    "seq_len": seq_len,
                    "n_docs": n_docs,
                    "doc_len": doc_len,
                    "forward_ms": round(sum(fwd_times)/len(fwd_times), 2),
                    "backward_ms": round(sum(bwd_times)/len(bwd_times), 2),
                    "total_ms": round(sum(fwd_times)/len(fwd_times) + sum(bwd_times)/len(bwd_times), 2),
                    "peak_mb": round(peak_mb, 1),
                    "theoretical_sparsity": round(sparsity, 4),
                }
                results.append(result)
                print(f"  seq={seq_len:4d} docs={n_docs:2d} | fwd={result['forward_ms']:7.2f}ms | bwd={result['backward_ms']:7.2f}ms | peak={result['peak_mb']:8.1f}MB | sparse={sparsity:.1%}")

                del q, k, v, out, block_mask, boundary
                reset_memory_stats()

            except torch.cuda.OutOfMemoryError:
                print(f"  seq={seq_len:4d} docs={n_docs:2d} | OOM")
                results.append({"seq_len": seq_len, "n_docs": n_docs, "oom": True})
                reset_memory_stats()
            except Exception as e:
                print(f"  seq={seq_len:4d} docs={n_docs:2d} | Error: {e}")
                results.append({"seq_len": seq_len, "n_docs": n_docs, "error": str(e)})
                reset_memory_stats()

    return results

# ============================================================
# Experiment 4: FlexAttention vs SDPA vs Vanilla Attention
# ============================================================

def experiment4_backend_comparison():
    """Exp4: Compare FlexAttention vs torch SDPA vs vanilla (eager) attention."""
    print("\n" + "="*60)
    print("Experiment 4: FlexAttention vs SDPA vs Vanilla (Forward+Backward)")
    print("="*60)

    seq_lengths = [512, 1024, 2048, 4096]
    batch_size = 1
    num_heads = 32
    head_dim = 64

    results = []

    for seq_len in seq_lengths:
        print(f"\n--- seq_len={seq_len} ---")

        q_base = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        k_base = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        v_base = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)

        # Causal block mask for FlexAttention
        try:
            block_mask = create_block_mask(causal_mask, batch_size, num_heads, seq_len, seq_len,
                                           device='cuda', _compile=True)
        except Exception as e:
            print(f"  BlockMask creation failed: {e}")
            continue

        for backend_name, run_fn in [
            ("FlexAttention", lambda q, k, v: flex_attention(q, k, v, block_mask=block_mask)),
            ("SDPA_causal", lambda q, k, v: F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)),
            ("Vanilla_eager", lambda q, k, v: _vanilla_causal_attention(q, k, v)),
        ]:
            try:
                q = q_base.clone().requires_grad_(True)
                k = k_base.clone().requires_grad_(True)
                v = v_base.clone().requires_grad_(True)

                # Warmup
                out = run_fn(q, k, v)
                out.sum().backward()
                q.grad = None; k.grad = None; v.grad = None

                # Measure forward
                reset_memory_stats()
                fwd_times = []
                for _ in range(5):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    out = run_fn(q, k, v)
                    torch.cuda.synchronize()
                    fwd_times.append((time.perf_counter() - t0) * 1000)

                fwd_peak = get_peak_memory()

                # Measure backward
                reset_memory_stats()
                bwd_times = []
                for _ in range(5):
                    out = run_fn(q, k, v)
                    grad_out = torch.ones_like(out)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    out.backward(grad_out)
                    torch.cuda.synchronize()
                    bwd_times.append((time.perf_counter() - t0) * 1000)
                    q.grad = None; k.grad = None; v.grad = None

                bwd_peak = get_peak_memory()

                result = {
                    "backend": backend_name,
                    "seq_len": seq_len,
                    "forward_ms": round(sum(fwd_times)/len(fwd_times), 2),
                    "backward_ms": round(sum(bwd_times)/len(bwd_times), 2),
                    "total_ms": round(sum(fwd_times)/len(fwd_times) + sum(bwd_times)/len(bwd_times), 2),
                    "fwd_peak_mb": round(fwd_peak, 1),
                    "bwd_peak_mb": round(bwd_peak, 1),
                }
                results.append(result)
                print(f"  {backend_name:20s} | fwd={result['forward_ms']:7.2f}ms | bwd={result['backward_ms']:7.2f}ms | total={result['total_ms']:7.2f}ms")

                del q, k, v, out
                reset_memory_stats()

            except torch.cuda.OutOfMemoryError:
                print(f"  {backend_name:20s} | OOM")
                results.append({"backend": backend_name, "seq_len": seq_len, "oom": True})
                reset_memory_stats()
            except Exception as e:
                print(f"  {backend_name:20s} | Error: {e}")
                results.append({"backend": backend_name, "seq_len": seq_len, "error": str(e)})

        del q_base, k_base, v_base, block_mask
        reset_memory_stats()

    return results

def _vanilla_causal_attention(q, k, v):
    """Vanilla eager attention with causal mask."""
    # q,k,v: [B, S, H, D] -> [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scale = q.shape[-1] ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    seq_len = q.shape[-2]
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device='cuda', dtype=torch.bool), diagonal=1)
    attn = attn.masked_fill(causal_mask, float('-inf'))
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)
    return out.transpose(1, 2)

# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("FlexAttention Backward Benchmark - Full Suite")
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print("=" * 60)

    all_results = {}

    # Run experiments
    all_results["experiment1_latency"] = experiment1_latency_by_mask()
    all_results["experiment2_compilation"] = experiment2_compilation_overhead()
    all_results["experiment3_doc_packing"] = experiment3_doc_packing()
    all_results["experiment4_backend_comparison"] = experiment4_backend_comparison()

    # Save results
    output_path = os.path.join(RESULTS_DIR, "backward_benchmark_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    exp1 = all_results.get("experiment1_latency", [])
    if exp1:
        print("\nExp1: Forward+Backward by Mask (best configs):")
        for r in exp1:
            if r.get("forward_ms") and r["seq_len"] == 2048:
                print(f"  {r['mask']:25s} seq={r['seq_len']} | fwd={r['forward_ms']}ms bwd={r['backward_ms']}ms peak={r.get('total_peak_mb','N/A')}MB")

    exp2 = all_results.get("experiment2_compilation", [])
    if exp2:
        print("\nExp2: Compilation Overhead:")
        for r in exp2:
            print(f"  {r['mask']:25s} | compile={r['compile_overhead_ms']}ms ({r['overhead_ratio']}x overhead)")

    exp4 = all_results.get("experiment4_backend_comparison", [])
    if exp4:
        print("\nExp4: Backend Comparison (seq=2048):")
        for r in exp4:
            if r.get("forward_ms") and r["seq_len"] == 2048:
                print(f"  {r['backend']:20s} | fwd={r['forward_ms']}ms bwd={r['backward_ms']}ms total={r['total_ms']}ms")

    print("\nDone!")
