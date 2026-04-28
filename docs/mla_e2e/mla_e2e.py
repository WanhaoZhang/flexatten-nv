#!/usr/bin/env python3
"""Project 5: MLA (Multi-head Latent Attention) End-to-End Analysis

Analyzes MLA (used in DeepSeek-V2/V3) vs standard MHA/GQA:
1. KV Cache compression ratio analysis
2. Attention computation comparison
3. Latency simulation: MLA latent projection vs standard KV
4. Memory footprint breakdown

MLA core idea: project KV into a low-rank latent space,
compressing KV cache from (num_heads * head_dim) to (latent_dim).
"""

import torch
import torch.nn as nn
import json
import time
import os
import gc
import math

MODEL_PATH = "/home/zhangwh/models/Qwen2.5-0.5B-Instruct"
RESULTS_DIR = "/home/zhangwh/flexatten-nv/docs/mla_e2e/results"
FIGURES_DIR = "/home/zhangwh/flexatten-nv/docs/mla_e2e/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def experiment1_kv_compression_analysis():
    """Exp1: Compare KV cache sizes across MHA, GQA, MLA configurations."""
    print("\n=== Experiment 1: KV Cache Compression Analysis ===")
    results = []

    # Common config
    seq_len = 4096
    batch_size = 1

    # Configuration matrix
    configs = [
        # Standard MHA (like Qwen2.5-0.5B: 14 heads, 64 dim)
        {"name": "MHA (Qwen2.5-0.5B)", "num_kv_heads": 14, "head_dim": 64, "num_layers": 24, "type": "MHA"},
        # GQA with fewer KV heads
        {"name": "GQA (4 KV heads)", "num_kv_heads": 4, "head_dim": 64, "num_layers": 24, "type": "GQA"},
        {"name": "GQA (2 KV heads)", "num_kv_heads": 2, "head_dim": 64, "num_layers": 24, "type": "GQA"},
        {"name": "GQA (1 KV head=MQA)", "num_kv_heads": 1, "head_dim": 64, "num_layers": 24, "type": "MQA"},
        # MLA with latent compression (DeepSeek-V2 style)
        {"name": "MLA (latent=256)", "num_kv_heads": 14, "head_dim": 64, "num_layers": 24, "type": "MLA", "latent_dim": 256},
        {"name": "MLA (latent=128)", "num_kv_heads": 14, "head_dim": 64, "num_layers": 24, "type": "MLA", "latent_dim": 128},
        {"name": "MLA (latent=64)", "num_kv_heads": 14, "head_dim": 64, "num_layers": 24, "type": "MLA", "latent_dim": 64},
        # DeepSeek-V2 actual config
        {"name": "DeepSeek-V2 MLA", "num_kv_heads": 128, "head_dim": 128, "num_layers": 60, "type": "MLA", "latent_dim": 512},
        # Llama-3 70B
        {"name": "Llama-3 70B GQA", "num_kv_heads": 8, "head_dim": 128, "num_layers": 80, "type": "GQA"},
    ]

    for cfg in configs:
        num_layers = cfg["num_layers"]
        num_kv_heads = cfg["num_kv_heads"]
        head_dim = cfg["head_dim"]

        if cfg["type"] == "MLA":
            # MLA: KV cache stores latent vectors, not full KV
            latent_dim = cfg["latent_dim"]
            # Per token per layer: 2 * latent_dim (K and V share latent)
            # Actually MLA stores only the latent vector, then projects to K and V on the fly
            kv_per_token_per_layer = latent_dim * 2  # latent vector (shared K/V projection)
        else:
            # MHA/GQA/MQA: standard KV cache
            kv_per_token_per_layer = 2 * num_kv_heads * head_dim

        total_kv_bytes = kv_per_token_per_layer * num_layers * seq_len * batch_size * 2  # FP16
        total_kv_mb = total_kv_bytes / (1024 * 1024)

        result = {
            "method": cfg["name"],
            "type": cfg["type"],
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "latent_dim": cfg.get("latent_dim", None),
            "kv_per_token_per_layer_bytes": kv_per_token_per_layer * 2,  # FP16
            "total_kv_mb": round(total_kv_mb, 1),
            "compression_vs_mha": None,  # filled later
        }
        results.append(result)
        print(f"  {cfg['name']}: {total_kv_mb:.1f}MB KV cache")

    # Calculate compression ratios relative to MHA baseline
    mha_baseline = results[0]["total_kv_mb"]
    for r in results:
        r["compression_vs_mha"] = round(mha_baseline / r["total_kv_mb"], 2) if r["total_kv_mb"] > 0 else 0

    return results


def experiment2_projection_latency():
    """Exp2: Measure MLA projection latency vs standard KV cache access."""
    print("\n=== Experiment 2: MLA Projection Latency ===")
    results = []

    device = "cuda"
    num_heads = 14
    head_dim = 64
    seq_lens = [64, 256, 1024, 4096]
    latent_dims = [64, 128, 256]
    num_iters = 50

    for sl in seq_lens:
        # Standard MHA KV cache read
        kv_cache = torch.randn(2, 24, num_heads, sl, head_dim, device=device, dtype=torch.float16)

        torch.cuda.synchronize()
        times = []
        for _ in range(num_iters):
            t0 = time.time()
            _ = kv_cache.clone()  # simulate KV cache read
            torch.cuda.synchronize()
            times.append(time.time() - t0)

        mha_read_ms = sum(times) / len(times) * 1000

        result = {
            "method": "MHA KV Read",
            "seq_len": sl,
            "latency_ms": round(mha_read_ms, 3),
            "kv_size_mb": round(kv_cache.numel() * 2 / (1024 * 1024), 1),
        }
        results.append(result)
        print(f"  MHA SL={sl}: read={mha_read_ms:.3f}ms, size={kv_cache.numel()*2/(1024*1024):.1f}MB")

        del kv_cache

        # MLA: latent read + projection
        for latent_dim in latent_dims:
            latent_cache = torch.randn(sl, latent_dim, device=device, dtype=torch.float16)
            w_k = torch.randn(num_heads, latent_dim, head_dim, device=device, dtype=torch.float16)
            w_v = torch.randn(num_heads, latent_dim, head_dim, device=device, dtype=torch.float16)

            torch.cuda.synchronize()
            times = []
            for _ in range(num_iters):
                t0 = time.time()
                # Read latent and project to K, V
                # latent: (sl, latent_dim) -> K: (sl, num_heads, head_dim)
                k_proj = torch.einsum('sd,hdk->shk', latent_cache.float(), w_k.float()).half()
                v_proj = torch.einsum('sd,hdk->shk', latent_cache.float(), w_v.float()).half()
                torch.cuda.synchronize()
                times.append(time.time() - t0)

            mla_ms = sum(times) / len(times) * 1000
            latent_mb = latent_cache.numel() * 2 / (1024 * 1024)

            result = {
                "method": f"MLA (latent={latent_dim})",
                "seq_len": sl,
                "latent_dim": latent_dim,
                "latency_ms": round(mla_ms, 3),
                "latent_size_mb": round(latent_mb, 1),
                "vs_mha_latency": round(mla_ms / mha_read_ms, 2) if mha_read_ms > 0 else 0,
                "compression_ratio": round((2 * num_heads * head_dim) / latent_dim, 2),
            }
            results.append(result)
            print(f"  MLA latent={latent_dim} SL={sl}: {mla_ms:.3f}ms ({mla_ms/mha_read_ms:.2f}x MHA), size={latent_mb:.1f}MB")

            del latent_cache, w_k, w_v

    return results


def experiment3_memory_breakdown():
    """Exp3: Full model memory breakdown comparison."""
    print("\n=== Experiment 3: Memory Breakdown ===")
    results = []

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )

    # Count parameters by type
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # Estimate attention vs FFN params
    attn_params = 0
    ffn_params = 0
    embed_params = 0

    for name, p in model.named_parameters():
        if "self_attn" in name or "attention" in name:
            attn_params += p.numel()
        elif "mlp" in name or "feed_forward" in name:
            ffn_params += p.numel()
        elif "embed" in name or "lm_head" in name:
            embed_params += p.numel()

    model_size_mb = total_bytes / (1024 * 1024)

    # KV cache sizes at different sequence lengths
    num_layers = 24
    num_kv_heads = 14
    head_dim = 64

    for sl in [512, 2048, 8192, 32768]:
        kv_bytes_fp16 = 2 * num_kv_heads * head_dim * num_layers * sl * 2  # FP16
        kv_bytes_mla_128 = 128 * num_layers * sl * 2  # MLA latent=128
        kv_bytes_mla_256 = 256 * num_layers * sl * 2

        result = {
            "method": "Memory Breakdown",
            "seq_len": sl,
            "model_weights_mb": round(model_size_mb, 0),
            "kv_fp16_mb": round(kv_bytes_fp16 / (1024 * 1024), 1),
            "kv_mla_128_mb": round(kv_bytes_mla_128 / (1024 * 1024), 1),
            "kv_mla_256_mb": round(kv_bytes_mla_256 / (1024 * 1024), 1),
            "total_mha_mb": round((total_bytes + kv_bytes_fp16) / (1024 * 1024), 1),
            "total_mla_128_mb": round((total_bytes + kv_bytes_mla_128) / (1024 * 1024), 1),
            "mla_kv_saving_pct": round((1 - kv_bytes_mla_128 / kv_bytes_fp16) * 100, 1),
        }
        results.append(result)
        print(f"  SL={sl}: MHA KV={kv_bytes_fp16/(1024*1024):.1f}MB, MLA KV={kv_bytes_mla_128/(1024*1024):.1f}MB "
              f"(save {result['mla_kv_saving_pct']}%)")

    results.append({
        "method": "Parameter Breakdown",
        "total_params_M": round(total_params / 1e6, 1),
        "attn_params_M": round(attn_params / 1e6, 1),
        "ffn_params_M": round(ffn_params / 1e6, 1),
        "embed_params_M": round(embed_params / 1e6, 1),
        "model_size_mb": round(model_size_mb, 0),
    })

    del model
    clear_gpu()
    return results


def experiment4_end_to_end_comparison():
    """Exp4: Simulate MLA vs MHA/GQA end-to-end decode latency."""
    print("\n=== Experiment 4: End-to-End Decode Simulation ===")
    results = []

    device = "cuda"
    num_layers = 24
    num_heads = 14
    head_dim = 64

    configs = [
        {"name": "MHA", "kv_dim": num_heads * head_dim},
        {"name": "GQA-4", "kv_dim": 4 * head_dim},
        {"name": "GQA-2", "kv_dim": 2 * head_dim},
        {"name": "MQA", "kv_dim": 1 * head_dim},
        {"name": "MLA-256", "kv_dim": 256},
        {"name": "MLA-128", "kv_dim": 128},
        {"name": "MLA-64", "kv_dim": 64},
    ]

    seq_lens = [128, 512, 2048, 8192]
    num_iters = 30

    for sl in seq_lens:
        for cfg in configs:
            kv_dim = cfg["kv_dim"]
            # Simulate KV cache for all layers
            # Shape: (num_layers, 2, kv_dim, sl) or (num_layers, sl, kv_dim)
            kv_data = torch.randn(num_layers, 2, sl, kv_dim, device=device, dtype=torch.float16)

            # Simulate single decode step: read full KV + compute attention
            q = torch.randn(num_layers, num_heads, head_dim, device=device, dtype=torch.float16)

            torch.cuda.synchronize()
            times = []
            for _ in range(num_iters):
                t0 = time.time()
                # Read KV cache (bandwidth bottleneck)
                _ = kv_data.clone()
                torch.cuda.synchronize()
                times.append(time.time() - t0)

            avg_ms = sum(times) / len(times) * 1000
            kv_mb = kv_data.numel() * 2 / (1024 * 1024)

            result = {
                "method": cfg["name"],
                "seq_len": sl,
                "kv_dim": kv_dim,
                "kv_read_ms": round(avg_ms, 3),
                "kv_size_mb": round(kv_mb, 1),
            }
            results.append(result)

            del kv_data

        print(f"  SL={sl}: " + ", ".join(
            f"{r['method']}={r['kv_read_ms']:.2f}ms" for r in results if r["seq_len"] == sl
        ))

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Project 5: MLA End-to-End Analysis")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)

    all_results = {}
    all_results["experiment1_kv_compression"] = experiment1_kv_compression_analysis()
    all_results["experiment2_projection_latency"] = experiment2_projection_latency()
    all_results["experiment3_memory_breakdown"] = experiment3_memory_breakdown()
    all_results["experiment4_e2e_decode"] = experiment4_end_to_end_comparison()

    all_results["metadata"] = {
        "gpu": torch.cuda.get_device_name(),
        "model": "Qwen2.5-0.5B-Instruct",
        "num_layers": 24,
        "num_heads": 14,
        "head_dim": 64,
    }

    with open(f"{RESULTS_DIR}/mla_e2e_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {RESULTS_DIR}/mla_e2e_results.json")
    print("Done!")
