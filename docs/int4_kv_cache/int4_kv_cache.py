#!/usr/bin/env python3
"""Project 3: INT4 KV Cache Quantization Analysis

Compares KV cache with different precision:
1. FP16 KV Cache (baseline)
2. FP8 KV Cache (vLLM native)
3. Theoretical INT4 KV Cache analysis

Since vLLM 0.19.1 supports --kv-cache-dtype fp8_e4m3fn but not INT4 KV cache directly,
we compare FP16 vs FP8 KV cache and analyze theoretical INT4 benefits.
"""

import torch
import json
import time
import os
import gc
import math

MODEL_PATH = "/home/zhangwh/models/Qwen2.5-0.5B-Instruct"
RESULTS_DIR = "/home/zhangwh/flexatten-nv/docs/int4_kv_cache/results"
FIGURES_DIR = "/home/zhangwh/flexatten-nv/docs/int4_kv_cache/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

from vllm import LLM, SamplingParams


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def experiment1_fp16_kv_cache():
    """Exp1: FP16 KV cache baseline - memory and throughput."""
    print("\n=== Experiment 1: FP16 KV Cache Baseline ===")
    results = []

    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.8,
        dtype="float16",
        trust_remote_code=True,
        kv_cache_dtype="auto",  # FP16
    )

    # Warmup
    llm.generate(["Hello"], SamplingParams(max_tokens=4, temperature=0.0))

    # Measure throughput at different batch sizes and sequence lengths
    batch_sizes = [1, 4, 8, 16, 32]
    sampling = SamplingParams(max_tokens=64, temperature=0.0)

    for bs in batch_sizes:
        prompts = [f"Tell me about topic {i}." for i in range(bs)]
        t0 = time.time()
        outputs = llm.generate(prompts, sampling)
        elapsed = time.time() - t0
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

        result = {
            "method": "FP16 KV Cache",
            "batch_size": bs,
            "total_time_ms": round(elapsed * 1000, 1),
            "tokens_per_s": round(total_tokens / elapsed, 0),
            "total_tokens": total_tokens,
        }
        results.append(result)
        print(f"  BS={bs}: {elapsed*1000:.0f}ms, {total_tokens/elapsed:.0f} tok/s")

    # Long sequence test
    long_prompts = ["Write a detailed essay about the history of artificial intelligence. " * 20]
    sampling_long = SamplingParams(max_tokens=128, temperature=0.0)
    t0 = time.time()
    outputs_long = llm.generate(long_prompts, sampling_long)
    elapsed_long = time.time() - t0
    total_prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs_long)
    total_gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs_long)

    results.append({
        "method": "FP16 KV Cache",
        "test": "long_sequence",
        "prompt_tokens": total_prompt_tokens,
        "generated_tokens": total_gen_tokens,
        "total_time_ms": round(elapsed_long * 1000, 1),
        "prefill_tps": round(total_prompt_tokens / elapsed_long, 0),
    })
    print(f"  Long seq: prompt={total_prompt_tokens}, gen={total_gen_tokens}, {elapsed_long*1000:.0f}ms")

    del llm
    clear_gpu()
    return results


def experiment2_fp8_kv_cache():
    """Exp2: FP8 KV cache - memory savings and throughput."""
    print("\n=== Experiment 2: FP8 KV Cache ===")
    results = []

    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.8,
        dtype="float16",
        trust_remote_code=True,
        kv_cache_dtype="fp8_e4m3fn",  # FP8 KV cache
    )

    # Warmup
    llm.generate(["Hello"], SamplingParams(max_tokens=4, temperature=0.0))

    # Same throughput tests
    batch_sizes = [1, 4, 8, 16, 32]
    sampling = SamplingParams(max_tokens=64, temperature=0.0)

    for bs in batch_sizes:
        prompts = [f"Tell me about topic {i}." for i in range(bs)]
        t0 = time.time()
        outputs = llm.generate(prompts, sampling)
        elapsed = time.time() - t0
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

        result = {
            "method": "FP8 KV Cache",
            "batch_size": bs,
            "total_time_ms": round(elapsed * 1000, 1),
            "tokens_per_s": round(total_tokens / elapsed, 0),
            "total_tokens": total_tokens,
        }
        results.append(result)
        print(f"  BS={bs}: {elapsed*1000:.0f}ms, {total_tokens/elapsed:.0f} tok/s")

    # Long sequence test
    long_prompts = ["Write a detailed essay about the history of artificial intelligence. " * 20]
    sampling_long = SamplingParams(max_tokens=128, temperature=0.0)
    t0 = time.time()
    outputs_long = llm.generate(long_prompts, sampling_long)
    elapsed_long = time.time() - t0
    total_prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs_long)
    total_gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs_long)

    results.append({
        "method": "FP8 KV Cache",
        "test": "long_sequence",
        "prompt_tokens": total_prompt_tokens,
        "generated_tokens": total_gen_tokens,
        "total_time_ms": round(elapsed_long * 1000, 1),
        "prefill_tps": round(total_prompt_tokens / elapsed_long, 0),
    })
    print(f"  Long seq: prompt={total_prompt_tokens}, gen={total_gen_tokens}, {elapsed_long*1000:.0f}ms")

    del llm
    clear_gpu()
    return results


def experiment3_kv_cache_capacity_analysis():
    """Exp3: Theoretical analysis of KV cache capacity under different precisions."""
    print("\n=== Experiment 3: KV Cache Capacity Analysis ===")
    results = []

    # Qwen2.5-0.5B config
    num_layers = 24
    num_heads = 14
    head_dim = 64
    bytes_per_element = {"FP16": 2, "FP8": 1, "INT4": 0.5}

    # L4 24GB: ~16GB available for KV cache (after model weights)
    total_kv_cache_bytes = 16 * 1024 * 1024 * 1024  # 16 GB

    seq_lens = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    batch_sizes = [1, 4, 8, 16, 32, 64, 128]

    # Per-token KV cache cost per layer
    # KV cache per token = 2 (K+V) * num_heads * head_dim * bytes_per_element
    per_token_per_layer = {
        dtype: 2 * num_heads * head_dim * bpe
        for dtype, bpe in bytes_per_element.items()
    }

    # Max seq len for single request
    for dtype in ["FP16", "FP8", "INT4"]:
        cost_per_token = per_token_per_layer[dtype] * num_layers
        max_tokens = total_kv_cache_bytes / cost_per_token
        result = {
            "method": f"KV Cache ({dtype})",
            "bytes_per_element": bytes_per_element[dtype],
            "cost_per_token_bytes": cost_per_token,
            "max_single_request_tokens": int(max_tokens),
            "max_single_request_tokens_k": round(max_tokens / 1024, 0),
        }
        results.append(result)
        print(f"  {dtype}: {cost_per_token/1024:.1f}KB/token, max single req = {max_tokens/1024:.0f}K tokens")

    # Max concurrent requests for different seq lengths
    for dtype in ["FP16", "FP8", "INT4"]:
        cost_per_token = per_token_per_layer[dtype] * num_layers
        for sl in seq_lens:
            kv_per_request = cost_per_token * sl
            max_concurrent = total_kv_cache_bytes / kv_per_request
            results.append({
                "method": f"KV Capacity ({dtype})",
                "seq_len": sl,
                "max_concurrent_requests": int(max_concurrent),
                "kv_per_request_mb": round(kv_per_request / (1024 * 1024), 1),
            })

    # KV cache memory breakdown at different batch sizes and seq lengths
    for dtype in ["FP16", "FP8", "INT4"]:
        cost_per_token = per_token_per_layer[dtype] * num_layers
        for bs in [1, 8, 32]:
            for sl in [2048, 8192, 32768]:
                total_kv_mb = cost_per_token * bs * sl / (1024 * 1024)
                utilization = total_kv_mb / (16 * 1024) * 100
                results.append({
                    "method": f"KV Usage ({dtype})",
                    "batch_size": bs,
                    "seq_len": sl,
                    "total_kv_mb": round(total_kv_mb, 1),
                    "kv_cache_utilization_pct": round(utilization, 1),
                })
                if bs == 1 and sl == 32768:
                    print(f"  {dtype} BS={bs} SL={sl}: {total_kv_mb:.0f}MB ({utilization:.1f}%)")

    return results


def experiment4_quality_simulation():
    """Exp4: Simulate KV cache quantization error accumulation."""
    print("\n=== Experiment 4: KV Cache Quantization Error Simulation ===")
    results = []

    # Simulate KV cache values and measure quantization error
    torch.manual_seed(42)
    num_layers = 24
    num_heads = 14
    head_dim = 64
    seq_len = 1024

    for dtype_name in ["FP16", "FP8_E4M3", "INT4_sim"]:
        errors = []
        kv_sizes = []

        for layer_idx in range(num_layers):
            # Simulate KV values (normally distributed, scaled)
            kv = torch.randn(2, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

            if dtype_name == "FP16":
                kv_quant = kv
                bytes_per_elem = 2
            elif dtype_name == "FP8_E4M3":
                # FP8 E4M3: simulate quantization
                kv_quant = kv.float()
                scale = kv_quant.abs().max() / 448.0  # FP8 E4M3 max = 448
                kv_quant = torch.clamp(torch.round(kv_quant / scale), -448, 448) * scale
                kv_quant = kv_quant.half()
                bytes_per_elem = 1
            elif dtype_name == "INT4_sim":
                # INT4 group quantization (group=128)
                group_size = 128
                kv_flat = kv.reshape(-1, group_size)
                w_min = kv_flat.min(dim=1, keepdim=True).values
                w_max = kv_flat.max(dim=1, keepdim=True).values
                scale = (w_max - w_min) / 15
                zero_point = w_min
                kv_q = torch.clamp(torch.round((kv_flat - zero_point) / scale), 0, 15)
                kv_dequant = kv_q * scale + zero_point
                kv_quant = kv_dequant.reshape(kv.shape).half()
                bytes_per_elem = 0.5

            # Measure error
            mse = (kv.float() - kv_quant.float()).pow(2).mean().item()
            max_err = (kv.float() - kv_quant.float()).abs().max().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                kv.float().reshape(1, -1), kv_quant.float().reshape(1, -1)
            ).item()

            errors.append({"mse": mse, "max_err": max_err, "cos_sim": cos_sim})
            kv_sizes.append(kv.numel() * bytes_per_elem)

        avg_mse = sum(e["mse"] for e in errors) / len(errors)
        avg_cos_sim = sum(e["cos_sim"] for e in errors) / len(errors)
        total_kv_bytes = sum(kv_sizes)
        fp16_kv_bytes = num_layers * 2 * num_heads * seq_len * head_dim * 2

        result = {
            "method": dtype_name,
            "avg_mse": round(avg_mse, 6),
            "avg_cosine_similarity": round(avg_cos_sim, 6),
            "total_kv_mb": round(total_kv_bytes / (1024 * 1024), 1),
            "vs_fp16_ratio": round(total_kv_bytes / fp16_kv_bytes, 2),
        }
        results.append(result)
        print(f"  {dtype_name}: MSE={avg_mse:.6f}, cos_sim={avg_cos_sim:.6f}, size={total_kv_bytes/(1024*1024):.1f}MB")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Project 3: INT4 KV Cache Quantization Analysis")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"vLLM: {__import__('vllm').__version__}")
    print("=" * 60)

    all_results = {}
    all_results["experiment1_fp16_kv"] = experiment1_fp16_kv_cache()
    all_results["experiment2_fp8_kv"] = experiment2_fp8_kv_cache()
    all_results["experiment3_capacity"] = experiment3_kv_cache_capacity_analysis()
    all_results["experiment4_quality"] = experiment4_quality_simulation()

    all_results["metadata"] = {
        "gpu": torch.cuda.get_device_name(),
        "model": "Qwen2.5-0.5B-Instruct",
        "vllm_version": "0.19.1",
        "num_layers": 24,
        "num_heads": 14,
        "head_dim": 64,
    }

    with open(f"{RESULTS_DIR}/int4_kv_cache_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {RESULTS_DIR}/int4_kv_cache_results.json")
    print("Done!")
