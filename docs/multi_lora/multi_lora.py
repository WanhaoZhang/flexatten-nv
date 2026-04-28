#!/usr/bin/env python3
"""Project 7: Multi-LoRA High-Concurrency Serving Simulation

Simulates multi-tenant LoRA serving on a single GPU (NVIDIA L4, 24GB).
Measures throughput degradation as the number of concurrent LoRA adapters increases.

Key experiments:
1. Single LoRA baseline vs Multi-LoRA throughput
2. Memory overhead per LoRA adapter
3. Batch scheduling: same-LoRA vs mixed-LoRA
4. LoRA rank impact on throughput
5. Dynamic LoRA swap overhead simulation
"""

import torch
import torch.nn as nn
import json
import time
import os
import sys
from collections import defaultdict

# Add model path
MODEL_PATH = "/home/zhangwh/models/Qwen2.5-0.5B-Instruct"
RESULTS_DIR = "/home/zhangwh/flexatten-nv/docs/multi_lora/results"
FIGURES_DIR = "/home/zhangwh/flexatten-nv/docs/multi_lora/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType


def load_base_model():
    """Load base model in FP16."""
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Base model loaded. Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, tokenizer


def create_lora_adapter(model, rank=8, adapter_name="default"):
    """Add a LoRA adapter to the model."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )
    model.add_adapter(adapter_name, lora_config)
    return model


def experiment1_single_vs_multi_lora(model, tokenizer):
    """Exp1: Single LoRA baseline vs increasing number of concurrent LoRA adapters."""
    print("\n=== Experiment 1: Single vs Multi-LoRA Throughput ===")
    results = []

    base_mem = torch.cuda.memory_allocated() / 1e9

    # Generate random LoRA weights for N adapters
    num_adapters_list = [1, 2, 4, 8, 16]
    rank = 8
    seq_len = 64
    batch_size = 4
    num_iters = 20

    dummy_input = torch.randint(1, 1000, (batch_size, seq_len), device="cuda")

    # First, create all adapters
    for n in num_adapters_list:
        # Reset to base
        try:
            for name in list(model.peft_config.keys()):
                model.delete_adapter(name)
        except:
            pass

        # Create n adapters with random weights
        for i in range(n):
            adapter_name = f"adapter_{i}"
            if adapter_name not in model.peft_config:
                create_lora_adapter(model, rank=rank, adapter_name=adapter_name)

        mem_with_adapters = torch.cuda.memory_allocated() / 1e9

        # Measure throughput: each batch item uses a different adapter (worst case)
        torch.cuda.synchronize()
        times = []
        for _ in range(num_iters):
            t0 = time.time()
            for b in range(batch_size):
                adapter_idx = b % n
                model.set_adapter(f"adapter_{adapter_idx}")
                with torch.no_grad():
                    out = model(dummy_input[b:b+1])
            torch.cuda.synchronize()
            times.append(time.time() - t0)

        avg_time = sum(times) / len(times)
        tokens_per_s = (batch_size * seq_len) / avg_time

        result = {
            "num_adapters": n,
            "rank": rank,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "avg_latency_ms": avg_time * 1000,
            "tokens_per_s": round(tokens_per_s, 1),
            "gpu_memory_gb": round(mem_with_adapters, 3),
            "memory_per_adapter_mb": round((mem_with_adapters - base_mem) / n * 1024, 1),
            "throughput_vs_single": None,
        }
        results.append(result)
        print(f"  Adapters={n}: latency={avg_time*1000:.1f}ms, "
              f"throughput={tokens_per_s:.0f} tok/s, "
              f"mem={mem_with_adapters:.2f} GB, "
              f"mem/adapter={(mem_with_adapters-base_mem)/n*1024:.1f} MB")

    # Normalize throughput
    base_tps = results[0]["tokens_per_s"]
    for r in results:
        r["throughput_vs_single"] = round(r["tokens_per_s"] / base_tps, 3)

    return results


def experiment2_memory_per_adapter(model, tokenizer):
    """Exp2: Memory overhead per LoRA adapter at different ranks."""
    print("\n=== Experiment 2: Memory Overhead per LoRA Adapter ===")
    results = []

    ranks = [1, 2, 4, 8, 16, 32, 64]
    num_adapters = 8

    for rank in ranks:
        try:
            for name in list(model.peft_config.keys()):
                model.delete_adapter(name)
        except:
            pass

        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1e9

        for i in range(num_adapters):
            create_lora_adapter(model, rank=rank, adapter_name=f"adapter_{i}")

        mem_after = torch.cuda.memory_allocated() / 1e9
        total_lora_mem = (mem_after - mem_before) * 1024  # MB
        per_adapter = total_lora_mem / num_adapters

        result = {
            "rank": rank,
            "num_adapters": num_adapters,
            "total_lora_mem_mb": round(total_lora_mem, 1),
            "per_adapter_mb": round(per_adapter, 1),
            "gpu_total_gb": round(mem_after, 3),
        }
        results.append(result)
        print(f"  Rank={rank}: total={total_lora_mem:.1f} MB, "
              f"per_adapter={per_adapter:.1f} MB, "
              f"GPU={mem_after:.2f} GB")

    return results


def experiment3_same_vs_mixed_lora(model, tokenizer):
    """Exp3: Same-LoRA batch vs Mixed-LoRA batch throughput."""
    print("\n=== Experiment 3: Same-LoRA vs Mixed-LoRA Batch ===")
    results = []

    try:
        for name in list(model.peft_config.keys()):
            model.delete_adapter(name)
    except:
        pass

    num_adapters = 8
    rank = 8
    for i in range(num_adapters):
        create_lora_adapter(model, rank=rank, adapter_name=f"adapter_{i}")

    seq_len = 64
    batch_sizes = [1, 2, 4, 8]
    num_iters = 15

    for bs in batch_sizes:
        dummy_input = torch.randint(1, 1000, (bs, seq_len), device="cuda")

        # Same adapter (best case - batched)
        model.set_adapter("adapter_0")
        torch.cuda.synchronize()
        times_same = []
        for _ in range(num_iters):
            t0 = time.time()
            with torch.no_grad():
                out = model(dummy_input)
            torch.cuda.synchronize()
            times_same.append(time.time() - t0)

        # Mixed adapters (worst case - sequential)
        torch.cuda.synchronize()
        times_mixed = []
        for _ in range(num_iters):
            t0 = time.time()
            for b in range(bs):
                model.set_adapter(f"adapter_{b % num_adapters}")
                with torch.no_grad():
                    out = model(dummy_input[b:b+1])
            torch.cuda.synchronize()
            times_mixed.append(time.time() - t0)

        same_latency = sum(times_same) / len(times_same) * 1000
        mixed_latency = sum(times_mixed) / len(times_mixed) * 1000
        same_tps = bs * seq_len / (same_latency / 1000)
        mixed_tps = bs * seq_len / (mixed_latency / 1000)

        result = {
            "batch_size": bs,
            "same_lora_ms": round(same_latency, 1),
            "mixed_lora_ms": round(mixed_latency, 1),
            "slowdown": round(mixed_latency / same_latency, 2),
            "same_tps": round(same_tps, 0),
            "mixed_tps": round(mixed_tps, 0),
        }
        results.append(result)
        print(f"  BS={bs}: same={same_latency:.1f}ms, mixed={mixed_latency:.1f}ms, "
              f"slowdown={mixed_latency/same_latency:.2f}x")

    return results


def experiment4_rank_impact(model, tokenizer):
    """Exp4: LoRA rank impact on single-batch inference latency."""
    print("\n=== Experiment 4: LoRA Rank Impact on Latency ===")
    results = []

    ranks = [1, 2, 4, 8, 16, 32, 64]
    seq_len = 128
    num_iters = 20
    dummy_input = torch.randint(1, 1000, (1, seq_len), device="cuda")

    for rank in ranks:
        try:
            for name in list(model.peft_config.keys()):
                model.delete_adapter(name)
        except:
            pass

        create_lora_adapter(model, rank=rank, adapter_name="test")
        model.set_adapter("test")

        torch.cuda.synchronize()
        times = []
        for _ in range(num_iters):
            t0 = time.time()
            with torch.no_grad():
                out = model(dummy_input)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

        avg_ms = sum(times) / len(times) * 1000
        result = {
            "rank": rank,
            "latency_ms": round(avg_ms, 2),
            "tokens_per_s": round(seq_len / (avg_ms / 1000), 1),
            "params_per_layer": rank * 2 * 2 * 1024,  # lora_A + lora_B for q,v in 1024 dim
        }
        results.append(result)
        print(f"  Rank={rank}: latency={avg_ms:.2f}ms, tps={seq_len/(avg_ms/1000):.0f}")

    # Add baseline (no LoRA)
    try:
        for name in list(model.peft_config.keys()):
            model.delete_adapter(name)
    except:
        pass

    torch.cuda.synchronize()
    times = []
    for _ in range(num_iters):
        t0 = time.time()
        with torch.no_grad():
            out = model(dummy_input)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    base_ms = sum(times) / len(times) * 1000
    results.append({
        "rank": 0,
        "latency_ms": round(base_ms, 2),
        "tokens_per_s": round(seq_len / (base_ms / 1000), 1),
        "params_per_layer": 0,
        "note": "No LoRA baseline"
    })
    print(f"  Baseline: latency={base_ms:.2f}ms, tps={seq_len/(base_ms/1000):.0f}")

    return results


def experiment5_max_adapters_on_l4(model, tokenizer):
    """Exp5: How many LoRA adapters can fit on L4 (24GB)?"""
    print("\n=== Experiment 5: Maximum LoRA Adapters on L4 ===")
    results = []
    rank = 8
    max_adapters = 200

    try:
        for name in list(model.peft_config.keys()):
            model.delete_adapter(name)
    except:
        pass

    torch.cuda.empty_cache()
    mem_start = torch.cuda.memory_allocated() / 1e9

    for i in range(max_adapters):
        try:
            create_lora_adapter(model, rank=rank, adapter_name=f"adapter_{i}")
        except Exception as e:
            print(f"  Failed at adapter {i}: {e}")
            break

        if (i + 1) % 10 == 0:
            mem = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            results.append({
                "num_adapters": i + 1,
                "rank": rank,
                "memory_gb": round(mem, 3),
                "peak_memory_gb": round(peak, 3),
                "per_adapter_mb": round((mem - mem_start) / (i + 1) * 1024, 2),
            })
            print(f"  Adapters={i+1}: mem={mem:.2f} GB, peak={peak:.2f} GB")

            if peak > 22:  # Leave 2GB buffer
                print(f"  Approaching L4 limit at {i+1} adapters!")
                break

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Project 7: Multi-LoRA High-Concurrency Serving Simulation")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print("=" * 60)

    model, tokenizer = load_base_model()

    all_results = {}

    all_results["experiment1_single_vs_multi"] = experiment1_single_vs_multi_lora(model, tokenizer)
    all_results["experiment2_memory_overhead"] = experiment2_memory_per_adapter(model, tokenizer)
    all_results["experiment3_same_vs_mixed"] = experiment3_same_vs_mixed_lora(model, tokenizer)
    all_results["experiment4_rank_impact"] = experiment4_rank_impact(model, tokenizer)
    all_results["experiment5_max_adapters"] = experiment5_max_adapters_on_l4(model, tokenizer)

    all_results["metadata"] = {
        "gpu": torch.cuda.get_device_name(),
        "model": "Qwen2.5-0.5B-Instruct",
        "peft_version": "0.19.1",
        "torch_version": torch.__version__,
    }

    with open(f"{RESULTS_DIR}/multi_lora_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {RESULTS_DIR}/multi_lora_results.json")
    print("Done!")
