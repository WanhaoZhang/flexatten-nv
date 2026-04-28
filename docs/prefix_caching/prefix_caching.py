#!/usr/bin/env python3
"""Project 8: Automatic Prefix Caching in RAG — Real Benefit Analysis

Measures cache hit rate and TTFT improvement with vLLM prefix caching.
Uses Qwen2.5-0.5B-Instruct on L4 (24GB).

Experiments:
1. Baseline: no caching, repeated prefix → repeated compute
2. Prefix Caching: same prefix reused → cache hit → faster TTFT
3. Cache eviction under memory pressure
4. Multi-prefix interleaved access
"""

import torch
import json
import time
import os

MODEL_PATH = "/home/zhangwh/models/Qwen2.5-0.5B-Instruct"
RESULTS_DIR = "/home/zhangwh/flexatten-nv/docs/prefix_caching/results"
FIGURES_DIR = "/home/zhangwh/flexatten-nv/docs/prefix_caching/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def experiment1_prefix_caching():
    """Exp1: Compare TTFT with and without prefix caching."""
    print("\n=== Experiment 1: Prefix Caching TTFT ===")
    results = []

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Create a long system prompt (simulating RAG document)
    system_prompt = "You are a helpful assistant. " * 200  # ~1000 tokens
    system_ids = tokenizer.encode(system_prompt)

    # Different user questions (all share the same system prompt)
    questions = [
        "What is the capital of France?",
        "Explain quantum computing briefly.",
        "What are the benefits of exercise?",
        "How does photosynthesis work?",
        "What is the speed of light?",
    ]

    # --- WITHOUT prefix caching ---
    print("  Running without prefix caching...")
    llm_no_cache = LLM(
        model=MODEL_PATH,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.8,
        dtype="float16",
        trust_remote_code=True,
    )

    sampling = SamplingParams(max_tokens=32, temperature=0.0)

    times_no_cache = []
    for i, q in enumerate(questions):
        prompt = system_prompt + "\n\nQuestion: " + q
        t0 = time.time()
        outputs = llm_no_cache.generate([prompt], sampling)
        ttft = time.time() - t0
        times_no_cache.append(ttft)
        print(f"    Q{i+1}: TTFT={ttft*1000:.0f}ms, tokens={len(outputs[0].outputs[0].token_ids)}")

    del llm_no_cache
    torch.cuda.empty_cache()

    # --- WITH prefix caching ---
    print("  Running with prefix caching...")
    llm_cache = LLM(
        model=MODEL_PATH,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.8,
        dtype="float16",
        trust_remote_code=True,
    )

    times_cached = []
    for i, q in enumerate(questions):
        prompt = system_prompt + "\n\nQuestion: " + q
        t0 = time.time()
        outputs = llm_cache.generate([prompt], sampling)
        ttft = time.time() - t0
        times_cached.append(ttft)
        print(f"    Q{i+1}: TTFT={ttft*1000:.0f}ms")

    del llm_cache
    torch.cuda.empty_cache()

    for i in range(len(questions)):
        results.append({
            "question_idx": i,
            "prompt_tokens": len(system_ids),
            "ttft_no_cache_ms": round(times_no_cache[i] * 1000, 1),
            "ttft_cached_ms": round(times_cached[i] * 1000, 1),
            "speedup": round(times_no_cache[i] / times_cached[i], 2),
        })

    return results


def experiment2_prefix_lengths():
    """Exp2: TTFT improvement vs prefix length."""
    print("\n=== Experiment 2: Prefix Length Impact ===")
    results = []

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    llm = LLM(
        model=MODEL_PATH,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.8,
        dtype="float16",
        trust_remote_code=True,
    )
    sampling = SamplingParams(max_tokens=16, temperature=0.0)

    prefix_lengths = [50, 100, 200, 400, 800, 1200]

    for pl in prefix_lengths:
        # Create prefix of target length
        prefix_text = "This is a document about science. " * (pl // 8)
        prefix_tokens = len(tokenizer.encode(prefix_text))

        # First request (no cache)
        q = "Summarize the above."
        prompt = prefix_text + "\n" + q

        t0 = time.time()
        llm.generate([prompt], sampling)
        ttft_cold = time.time() - t0

        # Second request (same prefix, different question)
        q2 = "What is the main topic?"
        prompt2 = prefix_text + "\n" + q2
        t0 = time.time()
        llm.generate([prompt2], sampling)
        ttft_warm = time.time() - t0

        result = {
            "prefix_length_tokens": prefix_tokens,
            "ttft_cold_ms": round(ttft_cold * 1000, 1),
            "ttft_warm_ms": round(ttft_warm * 1000, 1),
            "speedup": round(ttft_cold / ttft_warm, 2),
            "saved_ms": round((ttft_cold - ttft_warm) * 1000, 1),
        }
        results.append(result)
        print(f"  Prefix={prefix_tokens}tok: cold={ttft_cold*1000:.0f}ms, warm={ttft_warm*1000:.0f}ms, saved={result['saved_ms']:.0f}ms")

    del llm
    torch.cuda.empty_cache()
    return results


def experiment3_multi_prefix_interleave():
    """Exp3: Multiple documents interleaved - cache thrashing."""
    print("\n=== Experiment 3: Multi-Prefix Interleave ===")
    results = []

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    llm = LLM(
        model=MODEL_PATH,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.7,  # Lower to force eviction
        dtype="float16",
        trust_remote_code=True,
    )
    sampling = SamplingParams(max_tokens=16, temperature=0.0)

    # Create 3 different long documents
    docs = [
        "Document about physics: " + "The universe is vast. " * 200,
        "Document about biology: " + "Cells divide and multiply. " * 200,
        "Document about chemistry: " + "Atoms bond together. " * 200,
    ]

    # Round 1: Send each doc + question (cold cache)
    cold_times = []
    for i, doc in enumerate(docs):
        prompt = doc + f"\nWhat is document {i+1} about?"
        t0 = time.time()
        llm.generate([prompt], sampling)
        cold_times.append((time.time() - t0) * 1000)

    # Round 2: Repeat same docs + different questions (warm cache)
    warm_times = []
    for i, doc in enumerate(docs):
        prompt = doc + f"\nSummarize document {i+1}."
        t0 = time.time()
        llm.generate([prompt], sampling)
        warm_times.append((time.time() - t0) * 1000)

    # Round 3: Interleave with new long docs to cause eviction
    eviction_times = []
    for i, doc in enumerate(docs):
        # Insert a completely new long doc between accesses
        new_doc = "Novel content: " + "Something completely new. " * 200
        llm.generate([new_doc + "\nWhat is this?"], sampling)

        # Now re-access original doc
        prompt = doc + f"\nDetails about document {i+1}?"
        t0 = time.time()
        llm.generate([prompt], sampling)
        eviction_times.append((time.time() - t0) * 1000)

    for i in range(len(docs)):
        results.append({
            "doc_idx": i,
            "cold_ms": round(cold_times[i], 1),
            "warm_ms": round(warm_times[i], 1),
            "after_eviction_ms": round(eviction_times[i], 1),
            "warm_speedup": round(cold_times[i] / warm_times[i], 2),
            "eviction_penalty": round(eviction_times[i] / warm_times[i], 2),
        })
        print(f"  Doc{i+1}: cold={cold_times[i]:.0f}ms, warm={warm_times[i]:.0f}ms, "
              f"after_eviction={eviction_times[i]:.0f}ms")

    del llm
    torch.cuda.empty_cache()
    return results


def experiment4_batch_prefix():
    """Exp4: Batch requests with shared prefix."""
    print("\n=== Experiment 4: Batch Prefix Sharing ===")
    results = []

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Test with prefix caching
    llm = LLM(
        model=MODEL_PATH,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.8,
        dtype="float16",
        trust_remote_code=True,
    )
    sampling = SamplingParams(max_tokens=16, temperature=0.0)

    prefix = "Context document: " + "Important information. " * 150
    batch_sizes = [1, 2, 4, 8]

    for bs in batch_sizes:
        prompts = [prefix + f"\nQuestion {i}: What is the context?" for i in range(bs)]

        t0 = time.time()
        outputs = llm.generate(prompts, sampling)
        total_time = time.time() - t0

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        result = {
            "batch_size": bs,
            "total_time_ms": round(total_time * 1000, 1),
            "per_request_ms": round(total_time * 1000 / bs, 1),
            "total_tokens": total_tokens,
            "tokens_per_s": round(total_tokens / total_time, 0),
        }
        results.append(result)
        print(f"  BS={bs}: total={total_time*1000:.0f}ms, per_req={total_time*1000/bs:.0f}ms, tps={total_tokens/total_time:.0f}")

    del llm
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Project 8: Automatic Prefix Caching Analysis")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"vLLM: {__import__('vllm').__version__}")
    print("=" * 60)

    all_results = {}
    all_results["experiment1_ttft"] = experiment1_prefix_caching()
    all_results["experiment2_prefix_length"] = experiment2_prefix_lengths()
    all_results["experiment3_eviction"] = experiment3_multi_prefix_interleave()
    all_results["experiment4_batch"] = experiment4_batch_prefix()

    all_results["metadata"] = {
        "gpu": torch.cuda.get_device_name(),
        "model": "Qwen2.5-0.5B-Instruct",
        "vllm_version": "0.19.1",
    }

    with open(f"{RESULTS_DIR}/prefix_caching_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {RESULTS_DIR}/prefix_caching_results.json")
    print("Done!")
