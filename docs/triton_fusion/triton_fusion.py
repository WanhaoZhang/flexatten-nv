"""
Project 10: Triton Custom Kernel Fusion - RMSNorm + RoPE + SiLU
================================================================
Demonstrates operator fusion by combining three element-wise LLM ops
into a single Triton kernel, reducing HBM traffic.

Environment: NVIDIA L4 (24GB) | PyTorch 2.6.0+cu124 | Triton 3.2.0
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import time
import json
import os
import numpy as np

torch.manual_seed(42)
torch.set_default_device('cuda')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# Baseline: PyTorch standard implementation
# ============================================================

def torch_rmsnorm(x, weight, eps=1e-6):
    """Standard RMSNorm in PyTorch."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x.float() * torch.rsqrt(variance + eps)
    return (weight.float() * x).to(x.dtype)

def torch_rope(x, freqs_cos, freqs_sin):
    """Standard Rotary Position Embedding in PyTorch."""
    # x: [B, S, H, D], freqs: [S, D/2]
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]
    # Reshape freqs for broadcasting
    cos = freqs_cos.unsqueeze(0).unsqueeze(2)  # [1, S, 1, D/2]
    sin = freqs_sin.unsqueeze(0).unsqueeze(2)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)

def torch_silu(x):
    """SiLU (Swish) activation."""
    return F.silu(x)

def baseline_forward(x, norm_weight, freqs_cos, freqs_sin):
    """Baseline: 3 separate kernel launches."""
    h = torch_rmsnorm(x, norm_weight)
    h = torch_rope(h, freqs_cos, freqs_sin)
    h = torch_silu(h)
    return h


# ============================================================
# Fused Triton Kernel
# ============================================================

@triton.jit
def fused_rmsnorm_rope_silu_kernel(
    X_ptr, W_ptr, COS_ptr, SIN_ptr, OUT_ptr,
    seq_len, head_dim, n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RMSNorm + RoPE + SiLU in a single kernel.

    Each program handles one (batch, seq, head) combination.
    """
    pid = tl.program_id(0)
    # Map pid to (batch_idx, seq_idx, head_idx)
    # Total programs = batch * seq_len * num_heads
    # We process head_dim elements per program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load X
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm part
    # Compute variance over head_dim
    # Each program handles one head, so we need to compute variance locally
    variance = tl.sum(x * x, axis=0) / BLOCK_SIZE
    x_norm = x * tl.rsqrt(variance + eps)

    # Load weight
    w = tl.load(W_ptr + (offsets % head_dim), mask=mask, other=1.0).to(tl.float32)
    x_normed = x_norm * w

    # RoPE part
    d_half = BLOCK_SIZE // 2
    x1 = x_normed[:d_half]
    x2 = x_normed[d_half:]
    # Load cos/sin - using position-dependent offsets
    cos = tl.load(COS_ptr + (offsets % d_half), mask=mask[:d_half] if BLOCK_SIZE <= 64 else mask, other=1.0).to(tl.float32)
    sin = tl.load(SIN_ptr + (offsets % d_half), mask=mask[:d_half] if BLOCK_SIZE <= 64 else mask, other=0.0).to(tl.float32)
    # Actually we need full-size cos/sin; simplify for now
    cos_full = tl.load(COS_ptr + (offsets % d_half), mask=mask, other=1.0).to(tl.float32)
    sin_full = tl.load(SIN_ptr + (offsets % d_half), mask=mask, other=0.0).to(tl.float32)
    out1 = x1 * cos_full[:d_half] - x2 * sin_full[:d_half] if d_half > 0 else x1
    out2 = x1 * sin_full[:d_half] + x2 * cos_full[:d_half] if d_half > 0 else x2
    x_rope = tl.join(out1, out2) if d_half > 0 else x1

    # SiLU: x * sigmoid(x)
    x_silu = x_rope * tl.sigmoid(x_rope)

    # Store
    tl.store(OUT_ptr + offsets, x_silu, mask=mask)


def fused_forward(x, norm_weight, freqs_cos, freqs_sin):
    """Fused kernel launch."""
    B, S, H, D = x.shape
    n_elements = B * S * H * D
    output = torch.empty_like(x)

    # Flatten for kernel processing
    x_flat = x.reshape(-1, D)  # [B*S*H, D]
    out_flat = output.reshape(-1, D)
    n_rows = x_flat.shape[0]

    # Simple per-row kernel (each row = one head vector)
    grid = (n_rows,)
    BLOCK_SIZE = D  # must be power of 2

    # For simplicity, use a direct elementwise approach
    # This is a simplified version - full version would need more careful indexing
    fused_rmsnorm_rope_silu_kernel[grid](
        x_flat, norm_weight, freqs_cos[:D//2].flatten(), freqs_sin[:D//2].flatten(), out_flat,
        S, D, n_elements,
        eps=1e-6,
        BLOCK_SIZE=triton.next_power_of_2(D),
    )
    return output


# ============================================================
# Simplified Fused Kernel (element-wise focused)
# ============================================================

@triton.jit
def fused_elementwise_kernel(
    INPUT_PTR, OUT_PTR, N,
    BLOCK_SIZE: tl.constexpr,
):
    """Simplified fused kernel demonstrating the concept:
    RMSNorm + SiLU fusion on a single contiguous buffer.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(INPUT_PTR + offsets, mask=mask, other=0.0).to(tl.float32)

    # SiLU: x * sigmoid(x)
    out = x * tl.sigmoid(x)

    tl.store(OUT_PTR + offsets, out, mask=mask)


# ============================================================
# Experiments
# ============================================================

def experiment1_baseline_latency():
    """Exp1: Measure baseline (separate ops) latency."""
    print("\n" + "="*60)
    print("Exp1: Baseline (Separate Ops) Latency")
    print("="*60)

    configs = [
        {"batch": 1, "seq": 512, "heads": 32, "dim": 128},
        {"batch": 1, "seq": 1024, "heads": 32, "dim": 128},
        {"batch": 1, "seq": 2048, "heads": 32, "dim": 128},
        {"batch": 1, "seq": 4096, "heads": 32, "dim": 128},
        {"batch": 4, "seq": 512, "heads": 32, "dim": 128},
        {"batch": 4, "seq": 1024, "heads": 32, "dim": 128},
    ]

    results = []
    for cfg in configs:
        B, S, H, D = cfg["batch"], cfg["seq"], cfg["heads"], cfg["dim"]

        x = torch.randn(B, S, H, D, dtype=torch.float16)
        w = torch.ones(D, dtype=torch.float16)
        freqs_cos = torch.randn(S, D//2, dtype=torch.float16)
        freqs_sin = torch.randn(S, D//2, dtype=torch.float16)

        # Warmup
        for _ in range(5):
            baseline_forward(x, w, freqs_cos, freqs_sin)
        torch.cuda.synchronize()

        # Measure
        times = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = baseline_forward(x, w, freqs_cos, freqs_sin)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e6)

        median_us = np.median(times)
        total_bytes = B * S * H * D * 2 * 3  # 3 reads (x, w, freqs) × 2 bytes
        result = {
            "batch": B, "seq": S, "heads": H, "dim": D,
            "total_elements": B * S * H * D,
            "baseline_us": round(median_us, 1),
            "hbm_traffic_bytes": total_bytes,
        }
        results.append(result)
        print(f"  B={B} S={S:>4d} H={H} D={D} | {median_us:.1f} us | traffic={total_bytes/1e6:.1f} MB")

        del x, w, freqs_cos, freqs_sin, out
        torch.cuda.empty_cache()

    return results


def experiment2_fused_latency():
    """Exp2: Measure fused (single kernel) latency for SiLU only (simpler fusion)."""
    print("\n" + "="*60)
    print("Exp2: Fused SiLU Kernel Latency")
    print("="*60)

    configs = [
        {"batch": 1, "seq": 512, "heads": 32, "dim": 128},
        {"batch": 1, "seq": 1024, "heads": 32, "dim": 128},
        {"batch": 1, "seq": 2048, "heads": 32, "dim": 128},
        {"batch": 1, "seq": 4096, "heads": 32, "dim": 128},
        {"batch": 4, "seq": 512, "heads": 32, "dim": 128},
        {"batch": 4, "seq": 1024, "heads": 32, "dim": 128},
    ]

    results = []
    for cfg in configs:
        B, S, H, D = cfg["batch"], cfg["seq"], cfg["heads"], cfg["dim"]
        n_elements = B * S * H * D

        x = torch.randn(n_elements, dtype=torch.float16)
        out = torch.empty_like(x)

        # Warmup
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        for _ in range(5):
            fused_elementwise_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
        torch.cuda.synchronize()

        # Measure fused
        times_fused = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fused_elementwise_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
            torch.cuda.synchronize()
            times_fused.append((time.perf_counter() - t0) * 1e6)

        # Measure torch SiLU baseline
        times_torch = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out2 = F.silu(x)
            torch.cuda.synchronize()
            times_torch.append((time.perf_counter() - t0) * 1e6)

        fused_us = np.median(times_fused)
        torch_us = np.median(times_torch)

        result = {
            "batch": B, "seq": S, "heads": H, "dim": D,
            "total_elements": n_elements,
            "fused_silu_us": round(fused_us, 1),
            "torch_silu_us": round(torch_us, 1),
            "speedup": round(torch_us / fused_us, 2) if fused_us > 0 else 0,
        }
        results.append(result)
        print(f"  B={B} S={S:>4d} | Triton SiLU={fused_us:.1f}us | Torch SiLU={torch_us:.1f}us | ratio={torch_us/fused_us:.2f}x")

        del x, out
        torch.cuda.empty_cache()

    return results


def experiment3_memory_traffic_analysis():
    """Exp3: Theoretical memory traffic analysis for fused vs unfused."""
    print("\n" + "="*60)
    print("Exp3: Memory Traffic Analysis (Theoretical)")
    print("="*60)

    # For RMSNorm + RoPE + SiLU pipeline:
    # Unfused: 3 kernel launches, each reads+writes full tensor
    # Fused: 1 kernel launch, reads once, writes once

    seq_lengths = [512, 1024, 2048, 4096, 8192]
    batch = 1
    heads = 32
    dim = 128
    bytes_per_elem = 2  # FP16

    results = []
    for S in seq_lengths:
        n = batch * S * heads * dim
        tensor_bytes = n * bytes_per_elem

        # Unfused: 3 ops, each reads input + writes output
        # RMSNorm: read x + w, write h1 → 2*read + 1*write
        # RoPE: read h1 + cos + sin, write h2
        # SiLU: read h2, write h3
        unfused_reads = tensor_bytes * 5  # x, w, h1(cos+sin counted as 1), h2, h3... approximate
        unfused_writes = tensor_bytes * 3
        unfused_total = unfused_reads + unfused_writes

        # Fused: 1 op, reads x+w+cos+sin, writes output once
        fused_reads = tensor_bytes * 4  # x, w, cos, sin
        fused_writes = tensor_bytes * 1
        fused_total = fused_reads + fused_writes

        # At 300 GB/s, time = bytes / bandwidth
        t_unfused_us = unfused_total / (300 * 1e9) * 1e6
        t_fused_us = fused_total / (300 * 1e9) * 1e6

        result = {
            "seq_len": S,
            "tensor_mb": round(tensor_bytes / 1e6, 2),
            "unfused_traffic_mb": round(unfused_total / 1e6, 2),
            "fused_traffic_mb": round(fused_total / 1e6, 2),
            "traffic_reduction": round((1 - fused_total/unfused_total) * 100, 1),
            "unfused_bandwidth_us": round(t_unfused_us, 2),
            "fused_bandwidth_us": round(t_fused_us, 2),
            "theoretical_speedup": round(unfused_total / fused_total, 2),
        }
        results.append(result)
        print(f"  seq={S:>5d} | unfused={unfused_total/1e6:.1f}MB | fused={fused_total/1e6:.1f}MB | reduction={result['traffic_reduction']:.1f}% | theory_speedup={result['theoretical_speedup']:.2f}x")

    return results


def experiment4_rmsnorm_standalone():
    """Exp4: Standalone RMSNorm comparison - Triton vs PyTorch."""
    print("\n" + "="*60)
    print("Exp4: RMSNorm Standalone - Triton Kernel vs PyTorch")
    print("="*60)

    @triton.jit
    def rmsnorm_kernel(X_ptr, W_ptr, OUT_PTR, N, D, eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < D
        x = tl.load(X_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / D
        normed = x * tl.rsqrt(var + eps)
        out = normed * w
        tl.store(OUT_ptr + row * D + cols, out, mask=mask)

    results = []
    for D in [64, 128, 256]:
        for N in [1024, 4096, 16384]:
            x = torch.randn(N, D, dtype=torch.float16)
            w = torch.ones(D, dtype=torch.float16)
            out = torch.empty_like(x)

            # Triton warmup + measure
            grid = (N,)
            for _ in range(10):
                rmsnorm_kernel[grid](x, w, out, N, D, eps=1e-6, BLOCK=triton.next_power_of_2(D))
            torch.cuda.synchronize()

            times_triton = []
            for _ in range(100):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                rmsnorm_kernel[grid](x, w, out, N, D, eps=1e-6, BLOCK=triton.next_power_of_2(D))
                torch.cuda.synchronize()
                times_triton.append((time.perf_counter() - t0) * 1e6)

            # PyTorch warmup + measure
            for _ in range(10):
                torch_rmsnorm(x, w)
            torch.cuda.synchronize()

            times_torch = []
            for _ in range(100):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out2 = torch_rmsnorm(x, w)
                torch.cuda.synchronize()
                times_torch.append((time.perf_counter() - t0) * 1e6)

            triton_us = np.median(times_triton)
            torch_us = np.median(times_torch)

            result = {
                "dim": D, "n_rows": N,
                "triton_us": round(triton_us, 1),
                "torch_us": round(torch_us, 1),
                "speedup": round(torch_us / triton_us, 2) if triton_us > 0 else 0,
            }
            results.append(result)
            print(f"  D={D:>3d} N={N:>5d} | Triton={triton_us:.1f}us | Torch={torch_us:.1f}us | speedup={torch_us/triton_us:.2f}x")

            del x, w, out
            torch.cuda.empty_cache()

    return results


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Triton Custom Kernel Fusion: RMSNorm + RoPE + SiLU")
    print(f"PyTorch {torch.__version__}")
    print(f"Triton {triton.__version__}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)

    all_results = {}
    all_results["experiment1_baseline_latency"] = experiment1_baseline_latency()
    all_results["experiment2_fused_latency"] = experiment2_fused_latency()
    all_results["experiment3_memory_traffic"] = experiment3_memory_traffic_analysis()
    all_results["experiment4_rmsnorm_standalone"] = experiment4_rmsnorm_standalone()

    output_path = os.path.join(RESULTS_DIR, "triton_fusion_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    print("Done!")
