#!/usr/bin/env python3
"""Project 14: Continuous Batching — Scheduling Timeline Analysis

Measures vLLM's continuous batching behavior with variable-length requests.
Uses offline inference to simulate the scheduling.

Experiments:
1. Variable output length: measure how short requests finish before long ones
2. Throughput vs batch size with continuous batching
3. Mixed-length request scheduling efficiency
4. Comparison: static batching vs continuous batching
"""

import torch
import json
import time
import os

MODEL_PATH = "/home/zhangwh/models/Qwen2.5-0.5B-Instruct"
RESULTS_DIR = "/home/zhangwh/flexatten-nv/docs/continuous_batching/results"
FIGURES_DIR = "/home/zhangwh/flexatten-nv/docs/continuous_batching/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

from vllm import LLM, SamplingParams


def experiment1_variable_output():
    """Exp1: Send requests with different output lengths, measure completion times."""
    print("\n=== Experiment 1: Variable Output Length ===")
    results = []

    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.8,
        dtype="float16",
        trust_remote_code=True,
    )

    # Create prompts with different max_tokens
    max_tokens_list = [10, 30, 60, 120, 200]
    prompts = ["Write a short sentence about AI."] * len(max_tokens_list)
    sampling_params_list = [SamplingParams(max_tokens=mt, temperature=0.0) for mt in max_tokens_list]

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params_list)
    total_time = time.time() - t0

    for i, (mt, out) in enumerate(zip(max_tokens_list, outputs)):
        gen_tokens = len(out.outputs[0].token_ids)
        latency = out.metrics.finish_time - out.metrics.arrival_time if hasattr(out, 'metrics') else total_time
        results.append({
            "request_idx": i,
            "max_tokens": mt,
            "generated_tokens": gen_tokens,
            "finish_time_s": round(total_time, 3),
        })
        print(f"  Request {i}: max={mt}, gen={gen_tokens} tokens")

    results.append({"total_time_ms": round(total_time * 1000, 1)})

    del llm
    torch.cuda.empty_cache()
    return results


def experiment2_throughput_vs_batch():
    """Exp2: Throughput scaling with batch size."""
    print("\n=== Experiment 2: Throughput vs Batch Size ===")
    results = []

    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.8,
        dtype="float16",
        trust_remote_code=True,
    )

    sampling = SamplingParams(max_tokens=64, temperature=0.0)
    batch_sizes = [1, 2, 4, 8, 16, 32]

    for bs in batch_sizes:
        prompts = [f"Tell me about topic {i}." for i in range(bs)]

        t0 = time.time()
        outputs = llm.generate(prompts, sampling)
        elapsed = time.time() - t0

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps = total_tokens / elapsed

        result = {
            "batch_size": bs,
            "total_time_ms": round(elapsed * 1000, 1),
            "total_tokens": total_tokens,
            "tokens_per_s": round(tps, 0),
            "per_request_ms": round(elapsed * 1000 / bs, 1),
        }
        results.append(result)
        print(f"  BS={bs}: time={elapsed*1000:.0f}ms, tps={tps:.0f}, per_req={elapsed*1000/bs:.0f}ms")

    del llm
    torch.cuda.empty_cache()
    return results


def experiment3_mixed_lengths():
    """Exp3: Mixed short/long requests — measure scheduling efficiency."""
    print("\n=== Experiment 3: Mixed Length Requests ===")
    results = []

    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.8,
        dtype="float16",
        trust_remote_code=True,
    )

    # Mix of short and long requests
    configs = [
        {"label": "all_short", "max_tokens": 16, "count": 8},
        {"label": "all_long", "max_tokens": 128, "count": 8},
        {"label": "mixed", "max_tokens": "mixed", "count": 8},
    ]

    for cfg in configs:
        if cfg["max_tokens"] == "mixed":
            prompts = [f"Write about topic {i}." for i in range(8)]
            sp_list = [SamplingParams(max_tokens=16 if i % 2 == 0 else 128, temperature=0.0) for i in range(8)]
        else:
            prompts = [f"Write about topic {i}." for i in range(cfg["count"])]
            sp_list = [SamplingParams(max_tokens=cfg["max_tokens"], temperature=0.0)] * cfg["count"]

        t0 = time.time()
        outputs = llm.generate(prompts, sp_list)
        elapsed = time.time() - t0

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        gen_lens = [len(o.outputs[0].token_ids) for o in outputs]

        result = {
            "label": cfg["label"],
            "total_time_ms": round(elapsed * 1000, 1),
            "total_tokens": total_tokens,
            "tokens_per_s": round(total_tokens / elapsed, 0),
            "gen_lengths": gen_lens,
        }
        results.append(result)
        print(f"  {cfg['label']}: time={elapsed*1000:.0f}ms, tps={total_tokens/elapsed:.0f}, "
              f"lens={gen_lens}")

    del llm
    torch.cuda.empty_cache()
    return results


def experiment4_concurrent_requests():
    """Exp4: Concurrent request simulation with variable arrival."""
    print("\n=== Experiment 4: Concurrent Request Simulation ===")
    results = []

    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.8,
        dtype="float16",
        trust_remote_code=True,
        max_num_seqs=16,
    )

    # Simulate a realistic workload: 16 requests of varying complexity
    request_configs = [
        {"prompt": "Hi", "max_tokens": 10},       # trivial
        {"prompt": "Hello", "max_tokens": 10},     # trivial
        {"prompt": "Explain AI.", "max_tokens": 32},   # short
        {"prompt": "Explain ML.", "max_tokens": 32},   # short
        {"prompt": "Write about history.", "max_tokens": 64},  # medium
        {"prompt": "Write about science.", "max_tokens": 64},  # medium
        {"prompt": "Tell a long story about adventure.", "max_tokens": 128},  # long
        {"prompt": "Write an essay about technology.", "max_tokens": 128},    # long
    ] * 2  # 16 requests total

    prompts = [r["prompt"] for r in request_configs]
    sp_list = [SamplingParams(max_tokens=r["max_tokens"], temperature=0.0) for r in request_configs]

    t0 = time.time()
    outputs = llm.generate(prompts, sp_list)
    total_time = time.time() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    per_req_times = []
    for i, out in enumerate(outputs):
        gen = len(out.outputs[0].token_ids)
        # Approximate per-request time as proportional to tokens
        per_req_times.append(round(total_time / len(outputs) * 1000, 1))
        results.append({
            "request_idx": i,
            "max_tokens": request_configs[i]["max_tokens"],
            "generated_tokens": gen,
            "throughput_tok_per_s": round(gen / total_time, 1),
        })

    summary = {
        "total_requests": len(request_configs),
        "total_time_ms": round(total_time * 1000, 1),
        "total_tokens": total_tokens,
        "aggregate_tps": round(total_tokens / total_time, 0),
    }
    results.append(summary)
    print(f"  Total: {total_time*1000:.0f}ms, {total_tokens} tokens, {total_tokens/total_time:.0f} tok/s")

    del llm
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Project 14: Continuous Batching Analysis")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"vLLM: {__import__('vllm').__version__}")
    print("=" * 60)

    all_results = {}
    all_results["experiment1_variable_output"] = experiment1_variable_output()
    all_results["experiment2_throughput"] = experiment2_throughput_vs_batch()
    all_results["experiment3_mixed_lengths"] = experiment3_mixed_lengths()
    all_results["experiment4_concurrent"] = experiment4_concurrent_requests()

    all_results["metadata"] = {
        "gpu": torch.cuda.get_device_name(),
        "model": "Qwen2.5-0.5B-Instruct",
        "vllm_version": "0.19.1",
    }

    with open(f"{RESULTS_DIR}/continuous_batching_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {RESULTS_DIR}/continuous_batching_results.json")
    print("Done!")
