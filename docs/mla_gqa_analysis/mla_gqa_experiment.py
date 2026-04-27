"""
Project 5: GQA → MLA End-to-End Inference — GPU Measured Analysis
=================================================================
Simulate MHA / GQA / MLA attention on real GPU, measure:
- KV Cache memory footprint
- Decode latency (bandwidth-bound)
- Prefill latency (compute-bound)
- MLA latent projection overhead

Environment: NVIDIA L4 (24GB) | PyTorch 2.6.0+cu124
"""

import torch
import torch.nn.functional as F
import time
import json
import os
import numpy as np
import gc

torch.manual_seed(42)
torch.set_default_device('cuda')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def experiment1_kv_cache_memory():
    """Exp1: Measure actual KV Cache memory for MHA / GQA / MLA."""
    print("\n" + "="*60)
    print("Exp1: KV Cache Memory (GPU Measured)")
    print("="*60)

    configs = [
        {"name": "MHA-32", "kv_heads": 32, "head_dim": 128},
        {"name": "GQA-8", "kv_heads": 8, "head_dim": 128},
        {"name": "GQA-4", "kv_heads": 4, "head_dim": 128},
        {"name": "MLA-512", "kv_heads": 1, "head_dim": 512, "is_mla": True},
    ]

    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]
    batch = 1

    results = []
    for cfg in configs:
        for seq_len in seq_lengths:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            if cfg.get("is_mla"):
                # MLA: store latent [B, S, latent_dim]
                kv = torch.randn(batch, 2, seq_len, 512, dtype=torch.float16)
            else:
                kv = torch.randn(batch, 2 * cfg["kv_heads"], seq_len, cfg["head_dim"], dtype=torch.float16)

            actual_mb = kv.numel() * 2 / (1024**2)
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

            result = {
                "config": cfg["name"],
                "seq_len": seq_len,
                "kv_elements": kv.numel(),
                "kv_actual_mb": round(actual_mb, 2),
                "peak_mb": round(peak_mb, 2),
            }
            results.append(result)

            if seq_len in [1024, 4096, 8192]:
                print(f"  {cfg['name']:10s} seq={seq_len:>5d} | KV={actual_mb:>8.2f}MB | peak={peak_mb:.0f}MB")

            del kv
            torch.cuda.empty_cache()

    return results


def experiment2_decode_latency():
    """Exp2: Measure decode (seq=1 query vs full KV) latency for each architecture."""
    print("\n" + "="*60)
    print("Exp2: Decode Latency (Bandwidth-Bound)")
    print("="*60)

    configs = [
        {"name": "MHA-32", "q_heads": 32, "kv_heads": 32, "head_dim": 128},
        {"name": "GQA-8", "q_heads": 32, "kv_heads": 8, "head_dim": 128},
        {"name": "GQA-4", "q_heads": 32, "kv_heads": 4, "head_dim": 128},
    ]

    seq_lengths = [512, 1024, 2048, 4096, 8192]
    batch = 1
    results = []

    for cfg in configs:
        for seq_len in seq_lengths:
            q = torch.randn(batch, cfg["q_heads"], 1, cfg["head_dim"], dtype=torch.float16)
            k = torch.randn(batch, cfg["kv_heads"], seq_len, cfg["head_dim"], dtype=torch.float16)
            v = torch.randn(batch, cfg["kv_heads"], seq_len, cfg["head_dim"], dtype=torch.float16)

            # For GQA, expand K/V to match Q heads
            if cfg["q_heads"] != cfg["kv_heads"]:
                n_rep = cfg["q_heads"] // cfg["kv_heads"]
                k_exp = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(batch, cfg["q_heads"], seq_len, cfg["head_dim"])
                v_exp = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(batch, cfg["q_heads"], seq_len, cfg["head_dim"])
            else:
                k_exp, v_exp = k, v

            # Warmup
            for _ in range(5):
                F.scaled_dot_product_attention(q, k_exp, v_exp)
            torch.cuda.synchronize()

            # Measure
            times = []
            for _ in range(100):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = F.scaled_dot_product_attention(q, k_exp, v_exp)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e6)

            latency_us = np.median(times)
            kv_bytes = 2 * batch * cfg["kv_heads"] * seq_len * cfg["head_dim"] * 2

            result = {
                "config": cfg["name"],
                "seq_len": seq_len,
                "latency_us": round(float(latency_us), 1),
                "kv_bytes": kv_bytes,
                "kv_mb": round(kv_bytes / (1024**2), 2),
            }
            results.append(result)
            print(f"  {cfg['name']:10s} seq={seq_len:>5d} | latency={latency_us:>7.1f}us | KV={kv_bytes/(1024**2):.1f}MB")

            del q, k, v, k_exp, v_exp, out
            torch.cuda.empty_cache()

    # MLA decode: latent → K/V projection → attention
    print("\n  --- MLA decode with latent projection ---")
    latent_dim = 512
    head_dim = 128
    q_heads = 32
    kv_heads = 32  # after projection

    w_k = torch.randn(latent_dim, kv_heads * head_dim, dtype=torch.float16)
    w_v = torch.randn(latent_dim, kv_heads * head_dim, dtype=torch.float16)

    for seq_len in seq_lengths:
        q = torch.randn(batch, q_heads, 1, head_dim, dtype=torch.float16)
        latent = torch.randn(batch, seq_len, latent_dim, dtype=torch.float16)

        # Warmup
        for _ in range(5):
            k_mla = F.linear(latent, w_k.T).reshape(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
            v_mla = F.linear(latent, w_v.T).reshape(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
            F.scaled_dot_product_attention(q, k_mla, v_mla)
        torch.cuda.synchronize()

        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            k_mla = F.linear(latent, w_k.T).reshape(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
            v_mla = F.linear(latent, w_v.T).reshape(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k_mla, v_mla)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e6)

        latency_us = np.median(times)
        latent_bytes = 2 * batch * seq_len * latent_dim * 2

        result = {
            "config": "MLA-512",
            "seq_len": seq_len,
            "latency_us": round(float(latency_us), 1),
            "kv_bytes": latent_bytes,
            "kv_mb": round(latent_bytes / (1024**2), 2),
        }
        results.append(result)
        print(f"  {'MLA-512':10s} seq={seq_len:>5d} | latency={latency_us:>7.1f}us | KV={latent_bytes/(1024**2):.1f}MB")

        del q, latent, k_mla, v_mla, out
        torch.cuda.empty_cache()

    return results


def experiment3_prefill_latency():
    """Exp3: Measure prefill (full S tokens) latency for each architecture."""
    print("\n" + "="*60)
    print("Exp3: Prefill Latency (Compute-Bound)")
    print("="*60)

    configs = [
        {"name": "MHA-32", "q_heads": 32, "kv_heads": 32, "head_dim": 128},
        {"name": "GQA-8", "q_heads": 32, "kv_heads": 8, "head_dim": 128},
        {"name": "GQA-4", "q_heads": 32, "kv_heads": 4, "head_dim": 128},
    ]

    seq_lengths = [256, 512, 1024, 2048, 4096]
    batch = 1
    results = []

    for cfg in configs:
        for seq_len in seq_lengths:
            q = torch.randn(batch, cfg["q_heads"], seq_len, cfg["head_dim"], dtype=torch.float16)
            k = torch.randn(batch, cfg["kv_heads"], seq_len, cfg["head_dim"], dtype=torch.float16)
            v = torch.randn(batch, cfg["kv_heads"], seq_len, cfg["head_dim"], dtype=torch.float16)

            if cfg["q_heads"] != cfg["kv_heads"]:
                n_rep = cfg["q_heads"] // cfg["kv_heads"]
                k_exp = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(batch, cfg["q_heads"], seq_len, cfg["head_dim"])
                v_exp = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(batch, cfg["q_heads"], seq_len, cfg["head_dim"])
            else:
                k_exp, v_exp = k, v

            # Warmup
            for _ in range(5):
                F.scaled_dot_product_attention(q, k_exp, v_exp)
            torch.cuda.synchronize()

            times = []
            for _ in range(50):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = F.scaled_dot_product_attention(q, k_exp, v_exp)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e6)

            latency_us = np.median(times)
            result = {
                "config": cfg["name"],
                "seq_len": seq_len,
                "prefill_us": round(float(latency_us), 1),
            }
            results.append(result)
            print(f"  {cfg['name']:10s} seq={seq_len:>5d} | prefill={latency_us:>8.1f}us")

            del q, k, v, k_exp, v_exp, out
            torch.cuda.empty_cache()

    # MLA prefill: projection overhead
    print("\n  --- MLA prefill with latent projection ---")
    latent_dim = 512
    head_dim = 128
    q_heads = 32
    kv_heads = 32
    w_k = torch.randn(latent_dim, kv_heads * head_dim, dtype=torch.float16)
    w_v = torch.randn(latent_dim, kv_heads * head_dim, dtype=torch.float16)

    for seq_len in seq_lengths:
        q = torch.randn(batch, q_heads, seq_len, head_dim, dtype=torch.float16)
        latent = torch.randn(batch, seq_len, latent_dim, dtype=torch.float16)

        for _ in range(5):
            k_mla = F.linear(latent, w_k.T).reshape(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
            v_mla = F.linear(latent, w_v.T).reshape(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
            F.scaled_dot_product_attention(q, k_mla, v_mla)
        torch.cuda.synchronize()

        times = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            k_mla = F.linear(latent, w_k.T).reshape(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
            v_mla = F.linear(latent, w_v.T).reshape(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k_mla, v_mla)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e6)

        latency_us = np.median(times)
        result = {
            "config": "MLA-512",
            "seq_len": seq_len,
            "prefill_us": round(float(latency_us), 1),
        }
        results.append(result)
        print(f"  {'MLA-512':10s} seq={seq_len:>5d} | prefill={latency_us:>8.1f}us")

        del q, latent, k_mla, v_mla, out
        torch.cuda.empty_cache()

    del w_k, w_v
    torch.cuda.empty_cache()

    return results


def experiment4_mla_projection_cost():
    """Exp4: Isolate MLA latent→KV projection overhead."""
    print("\n" + "="*60)
    print("Exp4: MLA Latent→KV Projection Overhead")
    print("="*60)

    latent_dim = 512
    kv_dim = 32 * 128  # = 4096
    seq_lengths = [512, 1024, 2048, 4096]
    batch = 1

    results = []
    for seq_len in seq_lengths:
        latent = torch.randn(batch, seq_len, latent_dim, dtype=torch.float16)
        w_proj = torch.randn(latent_dim, kv_dim, dtype=torch.float16)

        # Warmup
        for _ in range(10):
            F.linear(latent, w_proj.T)
        torch.cuda.synchronize()

        # Measure projection
        times_proj = []
        for _ in range(100):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = F.linear(latent, w_proj.T)
            torch.cuda.synchronize()
            times_proj.append((time.perf_counter() - t0) * 1e6)
        proj_us = np.median(times_proj)

        # Measure without projection (direct KV access)
        kv = torch.randn(batch, seq_len, kv_dim, dtype=torch.float16)
        times_direct = []
        for _ in range(100):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = kv  # just access
            torch.cuda.synchronize()
            times_direct.append((time.perf_counter() - t0) * 1e6)
        direct_us = np.median(times_direct)

        result = {
            "seq_len": seq_len,
            "projection_us": round(float(proj_us), 1),
            "direct_access_us": round(float(direct_us), 1),
            "overhead_us": round(float(proj_us - direct_us), 1),
            "overhead_pct": round(float((proj_us / direct_us - 1) * 100), 1) if direct_us > 0 else 0,
        }
        results.append(result)
        print(f"  seq={seq_len:>5d} | projection={proj_us:.1f}us | direct={direct_us:.1f}us | overhead={proj_us - direct_us:.1f}us")

        del latent, w_proj, kv
        torch.cuda.empty_cache()

    return results


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("GQA → MLA End-to-End Inference Analysis")
    print(f"PyTorch {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)

    all_results = {}
    all_results["experiment1_kv_memory"] = experiment1_kv_cache_memory()
    all_results["experiment2_decode_latency"] = experiment2_decode_latency()
    all_results["experiment3_prefill_latency"] = experiment3_prefill_latency()
    all_results["experiment4_mla_projection"] = experiment4_mla_projection_cost()

    output_path = os.path.join(RESULTS_DIR, "mla_gqa_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    print("Done!")
