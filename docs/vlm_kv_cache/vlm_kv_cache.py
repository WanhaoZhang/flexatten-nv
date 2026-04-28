#!/usr/bin/env python3
"""Project 13: VLM KV Cache Disaster — Visual Token Analysis

Analyzes the memory impact of visual tokens in Vision-Language Models.
Uses mathematical modeling + GPU tensor simulation (no VLM model needed).

Experiments:
1. Visual token count vs image resolution
2. KV Cache memory: text-only vs multimodal
3. Multimodal TTFT simulation
4. VLM-specific cache strategy analysis
"""

import torch
import json
import time
import os
import math

RESULTS_DIR = "/home/zhangwh/flexatten-nv/docs/vlm_kv_cache/results"
FIGURES_DIR = "/home/zhangwh/flexatten-nv/docs/vlm_kv_cache/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def experiment1_visual_tokens():
    """Exp1: Visual token count at different image resolutions and patch sizes."""
    print("\n=== Experiment 1: Visual Token Count ===")
    results = []

    resolutions = [
        (224, 224, "224x224"),
        (512, 512, "512x512"),
        (1080, 720, "1080p"),
        (1920, 1080, "1080p_FHD"),
        (2560, 1440, "1440p_QHD"),
        (3840, 2160, "4K_UHD"),
    ]
    patch_sizes = [14, 16, 32]  # Common patch sizes (CLIP=14, ViT=16, SigLIP=32)

    for pw, ph, name in resolutions:
        for ps in patch_sizes:
            num_patches_h = math.ceil(ph / ps)
            num_patches_w = math.ceil(pw / ps)
            num_tokens = num_patches_h * num_patches_w

            # KV Cache for visual tokens only (FP16, 32 heads, 128 dim)
            kv_bytes = 2 * 32 * 128 * num_tokens * 2  # K + V, FP16
            kv_mb = kv_bytes / (1024 * 1024)

            results.append({
                "resolution": name,
                "pixels": (pw, ph),
                "patch_size": ps,
                "num_visual_tokens": num_tokens,
                "kv_cache_mb": round(kv_mb, 2),
            })

    # Print summary table
    for name in [r[2] for r in resolutions]:
        print(f"\n  {name}:")
        for r in results:
            if r["resolution"] == name:
                print(f"    PS={r['patch_size']}: {r['num_visual_tokens']} tokens, "
                      f"KV={r['kv_cache_mb']:.1f} MB")

    return results


def experiment2_kv_cache_comparison():
    """Exp2: Total KV Cache for text-only vs multimodal inputs."""
    print("\n=== Experiment 2: KV Cache: Text vs Multimodal ===")
    results = []

    # Typical VLM config: 32 layers, 32 heads, 128 dim, FP16
    num_layers = 32
    num_heads = 32
    head_dim = 128
    bytes_per_elem = 2  # FP16
    bytes_per_token = 2 * num_layers * num_heads * head_dim * bytes_per_elem  # K + V

    scenarios = [
        {"name": "Short text (128 tok)", "text_tokens": 128, "num_images": 0, "visual_per_image": 0},
        {"name": "Long text (4096 tok)", "text_tokens": 4096, "num_images": 0, "visual_per_image": 0},
        {"name": "Text + 1 image (576 vis tok)", "text_tokens": 256, "num_images": 1, "visual_per_image": 576},
        {"name": "Text + 1 1080p image", "text_tokens": 256, "num_images": 1, "visual_per_image": 3528},
        {"name": "Text + 1 4K image", "text_tokens": 256, "num_images": 1, "visual_per_image": 18900},
        {"name": "Text + 5 images (1080p)", "text_tokens": 256, "num_images": 5, "visual_per_image": 3528},
        {"name": "Text + 10 images (1080p)", "text_tokens": 256, "num_images": 10, "visual_per_image": 3528},
        {"name": "Video (100 frames)", "text_tokens": 128, "num_images": 100, "visual_per_image": 576},
    ]

    for s in scenarios:
        total_tokens = s["text_tokens"] + s["num_images"] * s["visual_per_image"]
        total_kv_bytes = total_tokens * bytes_per_token
        total_kv_mb = total_kv_bytes / (1024 * 1024)
        total_kv_gb = total_kv_mb / 1024

        # % of L4 24GB
        l4_pct = total_kv_gb / 24 * 100

        result = {
            "name": s["name"],
            "text_tokens": s["text_tokens"],
            "num_images": s["num_images"],
            "visual_per_image": s["visual_per_image"],
            "total_tokens": total_tokens,
            "visual_token_pct": round(s["num_images"] * s["visual_per_image"] / total_tokens * 100, 1) if total_tokens > 0 else 0,
            "kv_cache_mb": round(total_kv_mb, 1),
            "kv_cache_gb": round(total_kv_gb, 3),
            "l4_usage_pct": round(l4_pct, 1),
        }
        results.append(result)
        print(f"  {s['name']}: {total_tokens} tok, KV={total_kv_mb:.0f} MB ({total_kv_gb:.2f} GB), "
              f"L4={l4_pct:.1f}%, visual={result['visual_token_pct']:.0f}%")

    return results


def experiment3_prefill_simulation():
    """Exp3: Simulate prefill latency for multimodal vs text-only."""
    print("\n=== Experiment 3: Prefill Latency Simulation ===")
    results = []

    # L4 specs
    memory_bw = 300e9  # 300 GB/s
    tflops_fp16 = 60e12  # ~60 TFLOPS for FP16 on L4

    num_layers = 32
    hidden_dim = 4096

    configs = [
        {"name": "128 text tokens", "tokens": 128},
        {"name": "1024 text tokens", "tokens": 1024},
        {"name": "1 image (576 tok)", "tokens": 576},
        {"name": "1 1080p image (3528 tok)", "tokens": 3528},
        {"name": "1 4K image (18900 tok)", "tokens": 18900},
        {"name": "Text + 1 img (3784 tok)", "tokens": 3784},
        {"name": "Text + 5 imgs (17920 tok)", "tokens": 17920},
    ]

    for c in configs:
        S = c["tokens"]
        # Attention compute: O(S^2 * d * num_layers)
        attn_flops = 2 * S * S * hidden_dim * num_layers
        # FFN compute: O(S * d^2 * num_layers)
        ffn_flops = 2 * S * hidden_dim * hidden_dim * 4 * num_layers
        total_flops = attn_flops + ffn_flops

        # Memory read: weights + activations
        weight_bytes = 2 * num_layers * (hidden_dim * hidden_dim * 3 + hidden_dim * hidden_dim * 4) * 2
        activation_bytes = S * hidden_dim * 2

        compute_time = total_flops / tflops_fp16
        memory_time = (weight_bytes + activation_bytes) / memory_bw
        estimated_time = max(compute_time, memory_time)

        result = {
            "name": c["name"],
            "tokens": S,
            "attn_flops_T": round(attn_flops / 1e12, 2),
            "total_flops_T": round(total_flops / 1e12, 2),
            "compute_limited_ms": round(compute_time * 1000, 1),
            "memory_limited_ms": round(memory_time * 1000, 1),
            "estimated_ms": round(estimated_time * 1000, 1),
            "bottleneck": "compute" if compute_time > memory_time else "memory",
        }
        results.append(result)
        print(f"  {c['name']}: {S} tok, compute={compute_time*1000:.0f}ms, "
              f"mem={memory_time*1000:.0f}ms, est={estimated_time*1000:.0f}ms ({result['bottleneck']})")

    return results


def experiment4_cache_strategies():
    """Exp4: VLM-specific cache strategies comparison."""
    print("\n=== Experiment 4: Cache Strategy Analysis ===")
    results = []

    # Scenario: 10 requests, each with 1 image + short question
    num_requests = 10
    text_per_req = 64
    visual_per_req = 576  # 1 image
    total_per_req = text_per_req + visual_per_req

    bytes_per_token = 2 * 32 * 32 * 128 * 2  # K+V, 32 layers, 32 heads, 128 dim, FP16

    strategies = [
        {
            "name": "No caching",
            "description": "Each request recomputes all tokens",
            "cache_hit_tokens": 0,
            "cache_miss_tokens": total_per_req,
        },
        {
            "name": "Prefix cache (system prompt)",
            "description": "Cache shared system prompt only",
            "cache_hit_tokens": 32,  # ~32 token system prompt
            "cache_miss_tokens": total_per_req - 32,
        },
        {
            "name": "Image cache",
            "description": "Cache visual tokens (images often reused)",
            "cache_hit_tokens": visual_per_req,
            "cache_miss_tokens": text_per_req,
        },
        {
            "name": "Full multimodal prefix",
            "description": "Cache system prompt + image tokens",
            "cache_hit_tokens": 32 + visual_per_req,
            "cache_miss_tokens": text_per_req - 32,
        },
    ]

    for strat in strategies:
        # Per-request KV Cache
        cached_bytes = strat["cache_hit_tokens"] * bytes_per_token
        compute_bytes = strat["cache_miss_tokens"] * bytes_per_token
        total_stored = (strat["cache_hit_tokens"] + strat["cache_miss_tokens"]) * bytes_per_token

        # Over 10 requests
        total_compute_10 = 10 * compute_bytes
        total_stored_10 = total_stored + 9 * strat["cache_hit_tokens"] * bytes_per_token  # reuse cached

        compute_saving_pct = (1 - compute_bytes / (total_per_req * bytes_per_token)) * 100

        result = {
            "strategy": strat["name"],
            "description": strat["description"],
            "cache_hit_tokens": strat["cache_hit_tokens"],
            "cache_miss_tokens": strat["cache_miss_tokens"],
            "compute_saving_pct": round(compute_saving_pct, 1),
            "per_request_cache_mb": round(cached_bytes / (1024*1024), 1),
            "total_10_requests_gb": round(total_stored_10 / (1024*1024*1024), 2),
        }
        results.append(result)
        print(f"  {strat['name']}: compute saving={compute_saving_pct:.0f}%, "
              f"cache_per_req={cached_bytes/(1024*1024):.1f} MB")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Project 13: VLM KV Cache Analysis")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)

    all_results = {}
    all_results["experiment1_visual_tokens"] = experiment1_visual_tokens()
    all_results["experiment2_kv_comparison"] = experiment2_kv_cache_comparison()
    all_results["experiment3_prefill_simulation"] = experiment3_prefill_simulation()
    all_results["experiment4_cache_strategies"] = experiment4_cache_strategies()

    all_results["metadata"] = {
        "gpu": torch.cuda.get_device_name(),
        "analysis_type": "mathematical_modeling + tensor_simulation",
    }

    with open(f"{RESULTS_DIR}/vlm_kv_cache_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {RESULTS_DIR}/vlm_kv_cache_results.json")
    print("Done!")
