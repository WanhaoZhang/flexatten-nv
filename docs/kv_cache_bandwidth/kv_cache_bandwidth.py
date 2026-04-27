"""
Project 3: Beyond FP8 KV Cache — INT4 Compression & Bandwidth Wall Analysis
=============================================================================
Mathematical modeling + experimental validation of KV cache quantization
impact on decode throughput.

Environment: NVIDIA L4 (24GB) | PyTorch 2.6.0+cu124
"""

import torch
import torch.nn.functional as F
import time
import json
import os
import gc
import numpy as np

torch.set_default_device('cuda')
torch.manual_seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# L4 specs
L4_MEMORY_BANDWIDTH = 300  # GB/s (theoretical)
L4_FP16_TFLOPS = 121  # TFLOPS
L4_VRAM_GB = 24


def experiment1_memory_modeling():
    """Exp1: Theoretical KV cache memory for different formats."""
    print("\n" + "="*60)
    print("Exp1: KV Cache Memory Modeling")
    print("="*60)

    configs = [
        {"name": "MHA (Llama-2 style)", "num_heads": 32, "kv_heads": 32, "head_dim": 128},
        {"name": "GQA-8 (Llama-3 style)", "num_heads": 32, "kv_heads": 8, "head_dim": 128},
        {"name": "GQA-4 (Qwen-2 style)", "num_heads": 32, "kv_heads": 4, "head_dim": 128},
        {"name": "MLA-DeepSeek (latent=512)", "num_heads": 128, "kv_heads": 1, "latent_dim": 512, "head_dim": 128},
    ]

    seq_lengths = [1024, 4096, 8192, 16384, 32768, 65536, 131072]
    precisions = {"FP16": 2, "FP8": 1, "INT4": 0.5}
    batch_size = 1

    results = []

    for cfg in configs:
        for prec_name, bytes_per_elem in precisions.items():
            for seq_len in seq_lengths:
                if cfg.get("latent_dim"):
                    # MLA: store compressed latent, not full KV
                    kv_bytes = 2 * cfg["latent_dim"] * bytes_per_elem * seq_len * batch_size
                else:
                    # Standard: 2 * seq_len * kv_heads * head_dim * bytes_per_elem
                    kv_bytes = 2 * seq_len * cfg["kv_heads"] * cfg["head_dim"] * bytes_per_elem * batch_size

                kv_gb = kv_bytes / (1024**3)
                fits = kv_gb < L4_VRAM_GB

                result = {
                    "config": cfg["name"],
                    "precision": prec_name,
                    "seq_len": seq_len,
                    "kv_cache_gb": round(kv_gb, 4),
                    "fits_in_vram": fits,
                    "batch_size": batch_size,
                }
                results.append(result)

                if seq_len in [4096, 32768, 131072] and prec_name == "FP16":
                    print(f"  {cfg['name']:30s} {prec_name} seq={seq_len:>6d} | KV={kv_gb:.3f} GB | fits={fits}")

    return results


def experiment2_bandwidth_bound_decode():
    """Exp2: Mathematical model of decode throughput under bandwidth constraint.

    In decode (autoregressive) phase:
    - Each step processes 1 new token
    - Must read entire KV cache from memory
    - Compute is negligible (O(1) per KV position)
    - Bottleneck = KV cache read bandwidth

    T_decode ≈ KV_bytes / Memory_Bandwidth
    Throughput_tokens/s ≈ 1 / T_decode
    """
    print("\n" + "="*60)
    print("Exp2: Bandwidth-Bound Decode Throughput Model")
    print("="*60)

    configs = [
        {"name": "GQA-8 FP16", "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2},
        {"name": "GQA-8 FP8", "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 1},
        {"name": "GQA-8 INT4", "kv_heads": 8, "head_dim": 128, "bytes_per_elem": 0.5},
        {"name": "GQA-4 FP16", "kv_heads": 4, "head_dim": 128, "bytes_per_elem": 2},
        {"name": "GQA-4 FP8", "kv_heads": 4, "head_dim": 128, "bytes_per_elem": 1},
        {"name": "GQA-4 INT4", "kv_heads": 4, "head_dim": 128, "bytes_per_elem": 0.5},
        {"name": "MLA-512 FP16", "latent_dim": 512, "bytes_per_elem": 2},
        {"name": "MLA-512 FP8", "latent_dim": 512, "bytes_per_elem": 1},
    ]

    seq_lengths = [1024, 4096, 8192, 16384, 32768, 65536]
    results = []

    for cfg in configs:
        for seq_len in seq_lengths:
            if cfg.get("latent_dim"):
                kv_bytes = 2 * cfg["latent_dim"] * cfg["bytes_per_elem"] * seq_len
            else:
                kv_bytes = 2 * seq_len * cfg["kv_heads"] * cfg["head_dim"] * cfg["bytes_per_elem"]

            # Time to read KV cache
            kv_gb = kv_bytes / (1024**3)
            t_decode_s = kv_gb / L4_MEMORY_BANDWIDTH
            tokens_per_s = 1.0 / t_decode_s if t_decode_s > 0 else float('inf')

            result = {
                "config": cfg["name"],
                "seq_len": seq_len,
                "kv_bytes": kv_bytes,
                "kv_gb": round(kv_gb, 4),
                "decode_time_us": round(t_decode_s * 1e6, 1),
                "throughput_tokens_per_s": round(tokens_per_s, 1),
            }
            results.append(result)

            if seq_len in [4096, 32768, 65536]:
                print(f"  {cfg['name']:15s} seq={seq_len:>6d} | KV={kv_gb:.3f}GB | t_decode={t_decode_s*1e6:.0f}us | {tokens_per_s:.0f} tok/s")

    return results


def experiment3_dequant_overhead():
    """Exp3: Measure actual dequantization overhead for simulated INT4 KV."""
    print("\n" + "="*60)
    print("Exp3: Dequantization Overhead Simulation")
    print("="*60)

    seq_len = 4096
    num_kv_heads = 8
    head_dim = 128
    num_q_heads = 32
    batch = 1

    results = []

    # FP16 baseline (no dequant)
    k_fp16 = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float16)
    v_fp16 = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float16)
    q = torch.randn(batch, num_q_heads, 1, head_dim, dtype=torch.float16)  # decode: 1 token

    # Warmup SDPA
    for _ in range(5):
        F.scaled_dot_product_attention(q, k_fp16, v_fp16)
    torch.cuda.synchronize()

    # Measure FP16 decode
    times_fp16 = []
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = F.scaled_dot_product_attention(q, k_fp16, v_fp16)
        torch.cuda.synchronize()
        times_fp16.append((time.perf_counter() - t0) * 1e6)
    fp16_us = np.median(times_fp16)
    print(f"  FP16 decode (no dequant): {fp16_us:.1f} us")

    # Simulated INT4: store as int8 (closest PyTorch dtype), dequant to FP16
    # In reality INT4 would use packed storage, here we simulate the dequant compute
    k_int8 = (k_fp16 * 10).to(torch.int8)  # simulate quantized
    v_int8 = (v_fp16 * 10).to(torch.int8)

    # Measure dequant + decode
    times_int4 = []
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        k_deq = k_int8.float().half() / 10.0
        v_deq = v_int8.float().half() / 10.0
        out = F.scaled_dot_product_attention(q, k_deq, v_deq)
        torch.cuda.synchronize()
        times_int4.append((time.perf_counter() - t0) * 1e6)
    int4_us = np.median(times_int4)
    dequant_overhead_us = int4_us - fp16_us
    overhead_pct = (dequant_overhead_us / fp16_us) * 100

    print(f"  INT4 dequant+decode: {int4_us:.1f} us (overhead: {dequant_overhead_us:.1f} us, {overhead_pct:.1f}%)")

    result = {
        "seq_len": seq_len,
        "kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "fp16_decode_us": round(fp16_us, 1),
        "int4_dequant_decode_us": round(int4_us, 1),
        "dequant_overhead_us": round(dequant_overhead_us, 1),
        "overhead_pct": round(overhead_pct, 1),
    }
    results.append(result)

    # Sweep over seq lengths
    for sl in [1024, 2048, 4096, 8192, 16384]:
        try:
            k_fp16 = torch.randn(batch, num_kv_heads, sl, head_dim, dtype=torch.float16)
            v_fp16 = torch.randn(batch, num_kv_heads, sl, head_dim, dtype=torch.float16)
            k_int8 = (k_fp16 * 10).to(torch.int8)
            v_int8 = (v_fp16 * 10).to(torch.int8)

            # FP16
            times_f = []
            for _ in range(50):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                F.scaled_dot_product_attention(q, k_fp16, v_fp16)
                torch.cuda.synchronize()
                times_f.append((time.perf_counter() - t0) * 1e6)
            fp16 = np.median(times_f)

            # INT4 sim
            times_i = []
            for _ in range(50):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                k_d = k_int8.float().half() / 10.0
                v_d = v_int8.float().half() / 10.0
                F.scaled_dot_product_attention(q, k_d, v_d)
                torch.cuda.synchronize()
                times_i.append((time.perf_counter() - t0) * 1e6)
            int4 = np.median(times_i)

            r = {
                "seq_len": sl,
                "fp16_decode_us": round(fp16, 1),
                "int4_dequant_decode_us": round(int4, 1),
                "dequant_overhead_us": round(int4 - fp16, 1),
                "overhead_pct": round((int4 - fp16) / fp16 * 100, 1),
            }
            results.append(r)
            print(f"  seq={sl:>5d} | FP16={fp16:.1f}us | INT4={int4:.1f}us | overhead={int4-fp16:.1f}us ({(int4-fp16)/fp16*100:.1f}%)")

            del k_fp16, v_fp16, k_int8, v_int8
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  seq={sl:>5d} | OOM")
            results.append({"seq_len": sl, "oom": True})
            torch.cuda.empty_cache()

    return results


def experiment4_max_context_analysis():
    """Exp4: What's the maximum context length for different KV formats on L4?"""
    print("\n" + "="*60)
    print("Exp4: Maximum Context Length Analysis (L4 24GB)")
    print("="*60)

    # Assume: 7B model weights ~14GB in FP16, leaving ~8GB for KV cache
    model_weights_gb = 14
    available_kv_gb = L4_VRAM_GB - model_weights_gb
    print(f"  Model weights: {model_weights_gb} GB (7B FP16)")
    print(f"  Available for KV: {available_kv_gb} GB")

    configs = [
        {"name": "GQA-8 FP16", "kv_heads": 8, "head_dim": 128, "bytes": 2},
        {"name": "GQA-8 FP8", "kv_heads": 8, "head_dim": 128, "bytes": 1},
        {"name": "GQA-8 INT4", "kv_heads": 8, "head_dim": 128, "bytes": 0.5},
        {"name": "GQA-4 FP16", "kv_heads": 4, "head_dim": 128, "bytes": 2},
        {"name": "GQA-4 FP8", "kv_heads": 4, "head_dim": 128, "bytes": 1},
        {"name": "GQA-4 INT4", "kv_heads": 4, "head_dim": 128, "bytes": 0.5},
        {"name": "MLA-512 FP16", "latent_dim": 512, "bytes": 2},
        {"name": "MLA-512 FP8", "latent_dim": 512, "bytes": 1},
    ]

    results = []
    for cfg in configs:
        if cfg.get("latent_dim"):
            bytes_per_token = 2 * cfg["latent_dim"] * cfg["bytes"]
        else:
            bytes_per_token = 2 * cfg["kv_heads"] * cfg["head_dim"] * cfg["bytes"]

        max_tokens = int(available_kv_gb * 1024**3 / bytes_per_token)
        gb_per_1k = bytes_per_token * 1024 / (1024**3)

        result = {
            "config": cfg["name"],
            "bytes_per_token": bytes_per_token,
            "gb_per_1k_tokens": round(gb_per_1k, 4),
            "max_context_tokens": max_tokens,
        }
        results.append(result)
        print(f"  {cfg['name']:15s} | {bytes_per_token:>4d} bytes/token | {gb_per_1k:.4f} GB/1K | max={max_tokens:>10d} tokens")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("KV Cache Bandwidth Wall Analysis")
    print(f"PyTorch {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)

    all_results = {}
    all_results["experiment1_memory_modeling"] = experiment1_memory_modeling()
    all_results["experiment2_bandwidth_bound"] = experiment2_bandwidth_bound_decode()
    all_results["experiment3_dequant_overhead"] = experiment3_dequant_overhead()
    all_results["experiment4_max_context"] = experiment4_max_context_analysis()

    output_path = os.path.join(RESULTS_DIR, "kv_cache_bandwidth_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    print("Done!")
