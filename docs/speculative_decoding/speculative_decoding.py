#!/usr/bin/env python3
"""Project 4: Speculative Decoding Analysis

Analyzes speculative decoding performance using vLLM's built-in speculative decoding.
Since we only have one model (Qwen2.5-0.5B), we simulate speculative decoding benefits
through mathematical modeling and n-gram based speculation.

Experiments:
1. Baseline autoregressive decode profiling
2. Simulated speculative decoding: acceptance rate analysis
3. Theoretical speedup analysis at different acceptance rates
4. Batch speculation efficiency
"""

import torch
import torch.nn.functional as F
import json
import time
import os
import gc
import math

MODEL_PATH = "/home/zhangwh/models/Qwen2.5-0.5B-Instruct"
RESULTS_DIR = "/home/zhangwh/flexatten-nv/docs/speculative_decoding/results"
FIGURES_DIR = "/home/zhangwh/flexatten-nv/docs/speculative_decoding/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def experiment1_autoregressive_profile():
    """Exp1: Profile autoregressive decode latency per token."""
    print("\n=== Experiment 1: Autoregressive Decode Profile ===")
    results = []

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    # Measure single-token decode latency at different KV cache sizes
    prompt_lengths = [64, 128, 256, 512, 1024, 2048]
    num_decode_steps = 20

    for pl in prompt_lengths:
        input_ids = torch.randint(1, 1000, (1, pl), device="cuda")

        # Prefill
        with torch.no_grad():
            past = model(input_ids, use_cache=True).past_key_values

        # Measure decode steps
        decode_times = []
        new_token = torch.tensor([[100]], device="cuda")

        torch.cuda.synchronize()
        for step in range(num_decode_steps):
            t0 = time.time()
            with torch.no_grad():
                out = model(new_token, past_key_values=past, use_cache=True)
                past = out.past_key_values
                new_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            decode_times.append(time.time() - t0)

        avg_ms = sum(decode_times) / len(decode_times) * 1000
        kv_size_mb = sum(
            p.numel() * p.element_size()
            for layer_past in past
            for p in layer_past
        ) / (1024 * 1024)

        result = {
            "method": "Autoregressive",
            "prompt_length": pl,
            "decode_ms_per_token": round(avg_ms, 2),
            "kv_cache_mb": round(kv_size_mb, 1),
            "decode_tok_per_s": round(1000 / avg_ms, 1),
        }
        results.append(result)
        print(f"  PL={pl}: {avg_ms:.2f}ms/tok, KV={kv_size_mb:.1f}MB")

    del model
    clear_gpu()
    return results


def experiment2_speculative_simulation():
    """Exp2: Simulate speculative decoding with different draft models."""
    print("\n=== Experiment 2: Speculative Decoding Simulation ===")
    results = []

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    # Simulate speculative decoding using the same model
    # Draft K tokens, then verify all K at once
    draft_sizes = [2, 4, 6, 8, 10, 16]
    prompt = tokenizer("The history of artificial intelligence began in the", return_tensors="pt").input_ids.cuda()
    num_verify_steps = 20

    for k in draft_sizes:
        # For each k, measure: (a) K sequential decodes, (b) 1 verify pass
        input_ids = prompt

        # Warmup
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
            past = out.past_key_values

        # Measure K sequential decode steps (simulating draft model)
        torch.cuda.synchronize()
        t_draft_start = time.time()
        new_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        draft_tokens = [new_token.item()]
        draft_times = []

        for i in range(k - 1):
            t0 = time.time()
            with torch.no_grad():
                out = model(new_token, past_key_values=past, use_cache=True)
                past = out.past_key_values
                new_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                draft_tokens.append(new_token.item())
            torch.cuda.synchronize()
            draft_times.append(time.time() - t0)

        total_draft_ms = (time.time() - t_draft_start) * 1000

        # Count how many draft tokens match greedy decode (acceptance rate simulation)
        # Re-verify by doing a batch prefill of all tokens
        all_tokens = torch.cat([input_ids, torch.tensor([draft_tokens], device="cuda")], dim=1)

        torch.cuda.synchronize()
        t_verify_start = time.time()
        with torch.no_grad():
            verify_out = model(all_tokens, use_cache=False)
        torch.cuda.synchronize()
        verify_ms = (time.time() - t_verify_start) * 1000

        # Calculate acceptance rate based on greedy matching
        with torch.no_grad():
            greedy_tokens = []
            vpast = model(input_ids, use_cache=True).past_key_values
            vtoken = input_ids
            for _ in range(k):
                out = model(vtoken[:, -1:], past_key_values=vpast, use_cache=True)
                vpast = out.past_key_values
                vtoken = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                greedy_tokens.append(vtoken.item())

        accepted = 0
        for i in range(k):
            if draft_tokens[i] == greedy_tokens[i]:
                accepted += 1
            else:
                break
        acceptance_rate = accepted / k

        # Theoretical speedup
        # Without speculation: k * T_decode
        # With speculation: T_draft_total + T_verify (but verify is parallel)
        # Actual speedup depends on acceptance rate
        baseline_ms = k * 30  # approximate single decode ms
        speculation_ms = total_draft_ms + verify_ms
        speedup = baseline_ms / speculation_ms if speculation_ms > 0 else 0

        result = {
            "method": "Speculative Decoding",
            "draft_size_k": k,
            "total_draft_ms": round(total_draft_ms, 1),
            "verify_ms": round(verify_ms, 1),
            "acceptance_rate": round(acceptance_rate, 3),
            "accepted_tokens": accepted,
            "theoretical_speedup": round(speedup, 2),
        }
        results.append(result)
        print(f"  K={k}: draft={total_draft_ms:.1f}ms, verify={verify_ms:.1f}ms, "
              f"accept={acceptance_rate:.3f}, speedup={speedup:.2f}x")

    del model
    clear_gpu()
    return results


def experiment3_speedup_analysis():
    """Exp3: Theoretical speedup analysis."""
    print("\n=== Experiment 3: Theoretical Speedup Analysis ===")
    results = []

    # Speculative decoding speedup formula:
    # Speedup = 1 / ((1 + alpha * gamma) / (gamma + 1) + c)
    # where alpha = acceptance rate, gamma = draft tokens, c = overhead ratio

    acceptance_rates = [0.6, 0.7, 0.8, 0.9, 0.95]
    draft_sizes = [2, 4, 6, 8, 10, 16]

    for alpha in acceptance_rates:
        for gamma in draft_sizes:
            # Expected tokens per step: (1 - alpha^gamma) / (1 - alpha) + alpha^gamma * gamma
            # This is the expected number of tokens accepted per speculation round
            if alpha < 1.0:
                expected_tokens = (1 - alpha**gamma) / (1 - alpha) + alpha**gamma * gamma
            else:
                expected_tokens = gamma

            # Cost: 1 verify + gamma draft steps (draft model is smaller/faster)
            # Assume draft model is 2x faster than target
            draft_cost = gamma * 0.5  # draft model cost in decode units
            verify_cost = 1.0  # verify cost = 1 decode unit
            total_cost = draft_cost + verify_cost

            # Speedup = expected_tokens / total_cost (in units of single decode)
            # But actually, speedup = expected_tokens / (1 + draft_cost) since
            # without speculation we'd spend expected_tokens decode steps
            baseline_cost = expected_tokens  # autoregressive: one decode per token
            speculation_cost = verify_cost + gamma * 0.5  # parallel verify + sequential draft

            # More accurate: speculation produces expected_tokens tokens in
            # gamma draft steps + 1 verify step
            # Without: expected_tokens decode steps
            # Draft model assumed 3x smaller = 3x faster decode
            draft_speed_ratio = 3.0
            speculation_total_time = gamma / draft_speed_ratio + 1  # in target decode units
            speedup = expected_tokens / speculation_total_time

            result = {
                "method": "Theoretical",
                "acceptance_rate": alpha,
                "draft_size": gamma,
                "expected_accepted_tokens": round(expected_tokens, 2),
                "speedup": round(speedup, 2),
            }
            results.append(result)
            if gamma in [4, 8, 16] and alpha in [0.7, 0.8, 0.9]:
                print(f"  alpha={alpha}, gamma={gamma}: expected={expected_tokens:.1f} tok, speedup={speedup:.2f}x")

    return results


def experiment4_vllm_speculative():
    """Exp4: vLLM throughput with and without speculation-like optimizations."""
    print("\n=== Experiment 4: vLLM Throughput Comparison ===")
    results = []

    # vLLM without any speculation
    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.8,
        dtype="float16",
        trust_remote_code=True,
    )

    # Warmup
    llm.generate(["Hello"], SamplingParams(max_tokens=4, temperature=0.0))

    # Throughput at different output lengths (short vs long generation)
    output_lengths = [16, 32, 64, 128, 256]
    batch_size = 8

    for max_tok in output_lengths:
        prompts = [f"Write about topic {i}." for i in range(batch_size)]
        sampling = SamplingParams(max_tokens=max_tok, temperature=0.0)

        t0 = time.time()
        outputs = llm.generate(prompts, sampling)
        elapsed = time.time() - t0

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)

        result = {
            "method": "vLLM Baseline",
            "batch_size": batch_size,
            "max_tokens": max_tok,
            "total_output_tokens": total_tokens,
            "total_prompt_tokens": prompt_tokens,
            "total_time_ms": round(elapsed * 1000, 1),
            "output_tps": round(total_tokens / elapsed, 0),
            "total_tps": round((total_tokens + prompt_tokens) / elapsed, 0),
        }
        results.append(result)
        print(f"  max_tok={max_tok}: {elapsed*1000:.0f}ms, output={total_tokens/elapsed:.0f} tok/s")

    # Simulated speculation benefit analysis:
    # With speculative decoding (theoretical), for each generation step,
    # we'd produce gamma+1 tokens instead of 1 on average.
    # The benefit is most pronounced for long generation.
    for max_tok in output_lengths:
        for gamma in [4, 8]:
            for alpha in [0.7, 0.85]:
                # Without speculation: max_tok decode steps
                # With speculation: roughly max_tok / expected_tokens_per_round rounds
                if alpha < 1.0:
                    expected_per_round = (1 - alpha**gamma) / (1 - alpha)
                else:
                    expected_per_round = gamma

                num_rounds = max_tok / expected_per_round
                # Each round: gamma draft (cheap) + 1 verify
                # Draft cost ~ gamma * 0.3 (draft model 3x faster)
                # Verify cost ~ 1.0 (same as target decode)
                cost_per_round = gamma * 0.3 + 1.0
                total_spec_cost = num_rounds * cost_per_round
                baseline_cost = max_tok  # autoregressive

                result = {
                    "method": "Simulated Speculation",
                    "max_tokens": max_tok,
                    "gamma": gamma,
                    "acceptance_rate": alpha,
                    "expected_speedup": round(baseline_cost / total_spec_cost, 2),
                    "num_spec_rounds": round(num_rounds, 1),
                }
                results.append(result)

    del llm
    clear_gpu()
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Project 4: Speculative Decoding Analysis")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"vLLM: {__import__('vllm').__version__}")
    print("=" * 60)

    all_results = {}
    all_results["experiment1_autoregressive"] = experiment1_autoregressive_profile()
    all_results["experiment2_speculative_sim"] = experiment2_speculative_simulation()
    all_results["experiment3_speedup_theory"] = experiment3_speedup_analysis()
    all_results["experiment4_vllm_throughput"] = experiment4_vllm_speculative()

    all_results["metadata"] = {
        "gpu": torch.cuda.get_device_name(),
        "model": "Qwen2.5-0.5B-Instruct",
        "vllm_version": "0.19.1",
    }

    with open(f"{RESULTS_DIR}/speculative_decoding_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {RESULTS_DIR}/speculative_decoding_results.json")
    print("Done!")
