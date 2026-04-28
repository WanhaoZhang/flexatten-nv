#!/usr/bin/env python3
"""Project 7: Multi-LoRA High-Concurrency Serving Simulation

Simulates multi-tenant LoRA serving on a single GPU (NVIDIA L4, 24GB).
Measures throughput degradation as the number of concurrent LoRA adapters increases.
"""

import torch
import json
import time
import os
import gc

MODEL_PATH = "/home/zhangwh/models/Qwen2.5-0.5B-Instruct"
RESULTS_DIR = "/home/zhangwh/flexatten-nv/docs/multi_lora/results"
FIGURES_DIR = "/home/zhangwh/flexatten-nv/docs/multi_lora/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def load_peft_model_with_adapters(num_adapters=1, rank=8):
    """Load model and create multiple LoRA adapters."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )

    # Create initial LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank, lora_alpha=rank, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)

    # Add additional adapters
    for i in range(1, num_adapters):
        adapter_name = f"adapter_{i}"
        model.add_adapter(adapter_name, lora_config)

    model.eval()
    return model


def experiment1_multi_lora_throughput():
    """Exp1: Throughput vs number of concurrent LoRA adapters."""
    print("\n=== Experiment 1: Multi-LoRA Throughput ===")
    results = []

    num_adapters_list = [1, 2, 4, 8, 16]
    rank = 8
    seq_len = 64
    batch_size = 4
    num_iters = 15

    for n in num_adapters_list:
        clear_gpu()
        model = load_peft_model_with_adapters(num_adapters=n, rank=rank)
        mem = torch.cuda.memory_allocated() / 1e9

        dummy_input = torch.randint(1, 1000, (1, seq_len), device="cuda")

        # Measure: cycle through adapters for each batch item
        torch.cuda.synchronize()
        times = []
        for _ in range(num_iters):
            t0 = time.time()
            for b in range(batch_size):
                model.set_adapter(f"adapter_{b % n}")
                with torch.no_grad():
                    out = model(dummy_input)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

        avg_time = sum(times) / len(times)
        tokens_per_s = (batch_size * seq_len) / avg_time

        result = {
            "num_adapters": n,
            "rank": rank,
            "avg_latency_ms": round(avg_time * 1000, 1),
            "tokens_per_s": round(tokens_per_s, 1),
            "gpu_memory_gb": round(mem, 3),
        }
        results.append(result)
        print(f"  Adapters={n}: latency={avg_time*1000:.1f}ms, tps={tokens_per_s:.0f}, mem={mem:.2f}GB")
        del model
        clear_gpu()

    # Normalize
    base_tps = results[0]["tokens_per_s"]
    for r in results:
        r["throughput_vs_single"] = round(r["tokens_per_s"] / base_tps, 3)

    return results


def experiment2_memory_overhead():
    """Exp2: Memory overhead per LoRA adapter at different ranks."""
    print("\n=== Experiment 2: Memory Overhead per Adapter ===")
    results = []

    ranks = [1, 2, 4, 8, 16, 32, 64]
    num_adapters = 8

    for rank in ranks:
        clear_gpu()

        # Measure base model first
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
        )
        base_mem = torch.cuda.memory_allocated() / 1e9
        del base_model
        clear_gpu()

        model = load_peft_model_with_adapters(num_adapters=num_adapters, rank=rank)
        mem_with_adapters = torch.cuda.memory_allocated() / 1e9
        total_lora_mem = (mem_with_adapters - base_mem) * 1024  # MB

        result = {
            "rank": rank,
            "num_adapters": num_adapters,
            "total_lora_mem_mb": round(total_lora_mem, 1),
            "per_adapter_mb": round(total_lora_mem / num_adapters, 1),
            "gpu_total_gb": round(mem_with_adapters, 3),
        }
        results.append(result)
        print(f"  Rank={rank}: total={total_lora_mem:.1f} MB, per_adapter={total_lora_mem/num_adapters:.1f} MB")
        del model
        clear_gpu()

    return results


def experiment3_same_vs_mixed():
    """Exp3: Same-LoRA batch vs Mixed-LoRA batch."""
    print("\n=== Experiment 3: Same vs Mixed LoRA Batch ===")
    results = []

    num_adapters = 8
    rank = 8
    seq_len = 64
    num_iters = 10

    for bs in [1, 2, 4, 8]:
        clear_gpu()
        model = load_peft_model_with_adapters(num_adapters=num_adapters, rank=rank)

        dummy_input = torch.randint(1, 1000, (bs, seq_len), device="cuda")

        # Same adapter (batched)
        model.set_adapter("adapter_0")
        torch.cuda.synchronize()
        times_same = []
        for _ in range(num_iters):
            t0 = time.time()
            with torch.no_grad():
                out = model(dummy_input)
            torch.cuda.synchronize()
            times_same.append(time.time() - t0)

        # Mixed adapters (sequential)
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

        same_ms = sum(times_same) / len(times_same) * 1000
        mixed_ms = sum(times_mixed) / len(times_mixed) * 1000

        result = {
            "batch_size": bs,
            "same_lora_ms": round(same_ms, 1),
            "mixed_lora_ms": round(mixed_ms, 1),
            "slowdown": round(mixed_ms / same_ms, 2),
            "same_tps": round(bs * seq_len / (same_ms / 1000), 0),
            "mixed_tps": round(bs * seq_len / (mixed_ms / 1000), 0),
        }
        results.append(result)
        print(f"  BS={bs}: same={same_ms:.1f}ms, mixed={mixed_ms:.1f}ms, slowdown={mixed_ms/same_ms:.2f}x")
        del model
        clear_gpu()

    return results


def experiment4_rank_impact():
    """Exp4: LoRA rank impact on inference latency."""
    print("\n=== Experiment 4: LoRA Rank Impact ===")
    results = []

    seq_len = 128
    num_iters = 20
    dummy_input = torch.randint(1, 1000, (1, seq_len), device="cuda")

    # Baseline (no LoRA)
    clear_gpu()
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    base_model.eval()
    torch.cuda.synchronize()
    times = []
    for _ in range(num_iters):
        t0 = time.time()
        with torch.no_grad():
            out = base_model(dummy_input)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    base_ms = sum(times) / len(times) * 1000
    results.append({"rank": 0, "latency_ms": round(base_ms, 2), "note": "No LoRA baseline"})
    print(f"  Baseline: {base_ms:.2f}ms")
    del base_model
    clear_gpu()

    for rank in [1, 2, 4, 8, 16, 32, 64]:
        model = load_peft_model_with_adapters(num_adapters=1, rank=rank)
        model.set_adapter("adapter_0")

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
            "overhead_pct": round((avg_ms - base_ms) / base_ms * 100, 1),
        }
        results.append(result)
        print(f"  Rank={rank}: {avg_ms:.2f}ms (+{(avg_ms-base_ms)/base_ms*100:.1f}%)")
        del model
        clear_gpu()

    return results


def experiment5_max_adapters():
    """Exp5: How many LoRA adapters fit on L4?"""
    print("\n=== Experiment 5: Maximum Adapters on L4 ===")
    results = []
    rank = 8

    # Start with base
    clear_gpu()
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    base_mem = torch.cuda.memory_allocated() / 1e9

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank, lora_alpha=rank, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base_model, lora_config)

    max_adapters = 300
    count = 1  # Already have default adapter

    for i in range(1, max_adapters):
        try:
            model.add_adapter(f"adapter_{i}", lora_config)
            count = i + 1
        except Exception as e:
            print(f"  Failed at adapter {i}: {e}")
            break

        if count % 20 == 0:
            mem = torch.cuda.memory_allocated() / 1e9
            per_adapter = (mem - base_mem) / count * 1024
            results.append({
                "num_adapters": count,
                "rank": rank,
                "memory_gb": round(mem, 3),
                "per_adapter_mb": round(per_adapter, 2),
            })
            print(f"  Adapters={count}: mem={mem:.2f} GB, per_adapter={per_adapter:.2f} MB")

            if mem > 22:  # 2GB buffer on L4
                print(f"  Approaching L4 limit!")
                break

    # Final count
    mem = torch.cuda.memory_allocated() / 1e9
    results.append({
        "num_adapters": count,
        "rank": rank,
        "memory_gb": round(mem, 3),
        "per_adapter_mb": round((mem - base_mem) / count * 1024, 2),
        "note": "max adapters before limit",
    })
    print(f"  Max adapters: {count} at {mem:.2f} GB")

    del model
    clear_gpu()
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Project 7: Multi-LoRA High-Concurrency Serving")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    all_results = {}
    all_results["experiment1_multi_throughput"] = experiment1_multi_lora_throughput()
    all_results["experiment2_memory_overhead"] = experiment2_memory_overhead()
    all_results["experiment3_same_vs_mixed"] = experiment3_same_vs_mixed()
    all_results["experiment4_rank_impact"] = experiment4_rank_impact()
    all_results["experiment5_max_adapters"] = experiment5_max_adapters()

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
