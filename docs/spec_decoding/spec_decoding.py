"""
Project 4: Speculative Decoding ROI Calculator
================================================
Mathematical simulation of speculative decoding profit/loss under different
acceptance rates, draft model speeds, and task types.

Key formula:
  Speedup = (tokens_verified) / (T_draft + T_verify)
  where T_draft = gamma * t_draft_per_token
        T_verify = t_target_per_token (verifies gamma+1 tokens in one pass)

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

# L4 hardware specs
L4_MEMORY_BANDWIDTH = 300  # GB/s
L4_FP16_TFLOPS = 121


def experiment1_acceptance_rate_model():
    """Exp1: Mathematical model of speculative decoding speedup.

    Speculative decoding works as follows:
    1. Draft model generates gamma tokens autoregressively
    2. Target model verifies all gamma+1 tokens in one forward pass
    3. Accepted tokens are kept; first rejected token is replaced

    Expected tokens per iteration:
      E[accepted] = sum_{k=1}^{gamma} p^k = p * (1 - p^gamma) / (1 - p)
      where p = acceptance rate per token

    Speedup = E[accepted] / (gamma * alpha + 1)
      where alpha = T_draft / T_target (draft/target latency ratio)
    """
    print("\n" + "="*60)
    print("Exp1: Acceptance Rate vs Speedup Model")
    print("="*60)

    acceptance_rates = np.arange(0.50, 1.001, 0.05)
    gammas = [1, 2, 4, 5, 8]  # number of draft tokens
    alpha = 0.1  # draft model is 10x faster than target

    results = []
    for gamma in gammas:
        for p in acceptance_rates:
            p = min(p, 0.999)
            # Expected accepted tokens
            if abs(p - 1.0) < 1e-6:
                expected_tokens = gamma + 1
            else:
                expected_tokens = 1 + p * (1 - p**gamma) / (1 - p)

            # Total time ratio: gamma * alpha (draft) + 1 (target verify)
            time_ratio = gamma * alpha + 1.0

            speedup = expected_tokens / time_ratio
            break_even = speedup >= 1.0

            result = {
                "gamma": gamma,
                "acceptance_rate": round(float(p), 3),
                "expected_tokens": round(float(expected_tokens), 3),
                "time_ratio": time_ratio,
                "speedup": round(float(speedup), 3),
                "profitable": break_even,
            }
            results.append(result)

            if gamma in [1, 4, 8] and round(float(p), 2) in [0.60, 0.70, 0.80, 0.90, 0.95, 1.00]:
                print(f"  gamma={gamma} p={p:.2f} | E[tokens]={expected_tokens:.2f} | speedup={speedup:.2f}x | {'PROFIT' if break_even else 'LOSS'}")

    return results


def experiment2_draft_cost_analysis():
    """Exp2: At what alpha (draft/target speed ratio) does speculation lose money?

    The draft model has overhead: it must generate gamma tokens sequentially.
    If alpha is too large (draft model not fast enough), speculation hurts.

    Break-even condition: speedup >= 1.0
      E[accepted] / (gamma * alpha + 1) >= 1.0
      => alpha <= (E[accepted] - 1) / gamma
    """
    print("\n" + "="*60)
    print("Exp2: Draft Model Cost Analysis")
    print("="*60)

    gammas = [1, 2, 4, 5, 8]
    acceptance_rates = [0.70, 0.80, 0.90, 0.95]
    alphas = np.arange(0.02, 0.52, 0.02)

    results = []
    for p in acceptance_rates:
        for gamma in gammas:
            # Expected tokens
            if abs(p - 1.0) < 1e-6:
                expected_tokens = gamma + 1
            else:
                expected_tokens = 1 + p * (1 - p**gamma) / (1 - p)

            # Max alpha for profitability
            max_alpha = (expected_tokens - 1) / gamma

            # Speedup at different alphas
            for alpha in alphas:
                time_ratio = gamma * alpha + 1.0
                speedup = expected_tokens / time_ratio

                result = {
                    "acceptance_rate": p,
                    "gamma": gamma,
                    "alpha": round(float(alpha), 3),
                    "speedup": round(float(speedup), 3),
                    "max_alpha_for_profit": round(float(max_alpha), 3),
                }
                results.append(result)

            print(f"  p={p:.2f} gamma={gamma} | max_alpha={max_alpha:.3f} | E[tokens]={expected_tokens:.2f}")

    return results


def experiment3_task_type_simulation():
    """Exp3: Simulate different task types with realistic acceptance rates.

    Different tasks have different acceptance rates:
    - Chat/Generic: high acceptance (~0.85-0.95) — predictable language
    - Code: medium acceptance (~0.70-0.85) — some predictability
    - Math: low acceptance (~0.50-0.70) — hard to predict exact tokens
    - Translation: very high (~0.90-0.98) — highly predictable structure
    """
    print("\n" + "="*60)
    print("Exp3: Task Type Simulation")
    print("="*60)

    tasks = {
        "Chat (generic)": {"p_min": 0.85, "p_max": 0.95, "p_mean": 0.90},
        "Code generation": {"p_min": 0.70, "p_max": 0.85, "p_mean": 0.78},
        "Math reasoning": {"p_min": 0.50, "p_max": 0.70, "p_mean": 0.60},
        "Translation": {"p_min": 0.90, "p_max": 0.98, "p_mean": 0.94},
        "Creative writing": {"p_min": 0.60, "p_max": 0.80, "p_mean": 0.70},
    }

    gamma = 5  # typical setting
    alpha = 0.08  # draft model 12x faster
    n_simulations = 10000

    results = []
    for task_name, params in tasks.items():
        p = params["p_mean"]

        # Expected tokens
        expected_tokens = 1 + p * (1 - p**gamma) / (1 - p)
        time_ratio = gamma * alpha + 1.0
        expected_speedup = expected_tokens / time_ratio

        # Monte Carlo simulation
        total_tokens_produced = 0
        total_iterations = n_simulations

        for _ in range(n_simulations):
            # Simulate one speculation round
            accepted = 0
            for k in range(gamma):
                if np.random.random() < p:
                    accepted += 1
                else:
                    break

            # If all gamma accepted, target model produces one more token
            if accepted == gamma:
                accepted += 1

            total_tokens_produced += accepted

        actual_avg_tokens = total_tokens_produced / total_iterations
        sim_speedup = actual_avg_tokens / time_ratio

        result = {
            "task": task_name,
            "acceptance_rate_mean": p,
            "gamma": gamma,
            "alpha": alpha,
            "expected_tokens_analytical": round(float(expected_tokens), 3),
            "expected_tokens_simulated": round(float(actual_avg_tokens), 3),
            "speedup_analytical": round(float(expected_speedup), 3),
            "speedup_simulated": round(float(sim_speedup), 3),
            "profitable": expected_speedup >= 1.0,
        }
        results.append(result)
        print(f"  {task_name:20s} | p={p:.2f} | E[tok]={expected_tokens:.2f} | speedup={expected_speedup:.2f}x | {'PROFIT' if expected_speedup >= 1.0 else 'LOSS'}")

    return results


def experiment4_gpu_latency_simulation():
    """Exp4: Real GPU timing for draft vs target model sizes.

    Simulate the latency difference between a small (0.5B) and large (7B) model
    by measuring actual matrix multiply latency for proportional sizes.
    """
    print("\n" + "="*60)
    print("Exp4: GPU Latency Simulation (Small vs Large Model)")
    print("="*60)

    # Simulate: 0.5B model hidden_dim=896, 7B model hidden_dim=4096
    configs = [
        {"name": "Draft-0.5B", "hidden_dim": 896, "ffn_dim": 4864, "n_layers": 24},
        {"name": "Target-7B", "hidden_dim": 4096, "ffn_dim": 11008, "n_layers": 32},
    ]

    seq_len = 1  # decode mode: 1 token at a time
    batch_size = 1

    results = []
    for cfg in configs:
        D = cfg["hidden_dim"]
        FFN = cfg["ffn_dim"]
        n_layers = cfg["n_layers"]

        # Simulate one transformer layer: QKV proj + Attention + FFN
        # For decode (seq=1), attention is mostly KV cache read (bandwidth bound)
        # FFN: two linear layers [D, FFN] + [FFN, D]
        w1 = torch.randn(FFN, D, dtype=torch.float16)
        w2 = torch.randn(D, FFN, dtype=torch.float16)
        x = torch.randn(batch_size, seq_len, D, dtype=torch.float16)

        # Warmup
        for _ in range(10):
            h = F.linear(x, w1)
            h = F.silu(h)
            out = F.linear(h, w2)
        torch.cuda.synchronize()

        # Measure single-layer FFN latency
        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            h = F.linear(x, w1)
            h = F.silu(h)
            out = F.linear(h, w2)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e6)

        layer_us = np.median(times)
        total_us = layer_us * n_layers

        # Parameter count
        params_ffn = (D * FFN + FFN * D) * n_layers  # FFN params
        params_attn = (4 * D * D) * n_layers  # QKV+O params
        total_params = (params_ffn + params_attn) / 1e9

        result = {
            "name": cfg["name"],
            "hidden_dim": D,
            "ffn_dim": FFN,
            "n_layers": n_layers,
            "total_params_B": round(total_params, 2),
            "single_layer_ffn_us": round(float(layer_us), 1),
            "total_decode_us": round(float(total_us), 1),
        }
        results.append(result)
        print(f"  {cfg['name']:12s} | D={D:>4d} FFN={FFN:>5d} L={n_layers:>2d} | {total_params:.2f}B params | FFN layer={layer_us:.1f}us | total_decode={total_us:.0f}us")

        del w1, w2, x, h, out
        torch.cuda.empty_cache()

    # Compute alpha (draft/target ratio)
    if len(results) == 2:
        draft_time = results[0]["total_decode_us"]
        target_time = results[1]["total_decode_us"]
        alpha = draft_time / target_time
        print(f"\n  alpha (draft/target) = {alpha:.3f}")
        print(f"  Draft is {1/alpha:.1f}x faster than Target")

        # Optimal gamma for this alpha at different acceptance rates
        print(f"\n  Optimal gamma for alpha={alpha:.3f}:")
        for p in [0.70, 0.80, 0.90, 0.95]:
            best_gamma = 1
            best_speedup = 0
            for g in range(1, 16):
                expected = 1 + p * (1 - p**g) / (1 - p)
                time_r = g * alpha + 1.0
                su = expected / time_r
                if su > best_speedup:
                    best_speedup = su
                    best_gamma = g
            print(f"    p={p:.2f} | optimal_gamma={best_gamma} | max_speedup={best_speedup:.2f}x")

    return results


def experiment5_optimal_gamma_sweep():
    """Exp5: Find optimal gamma for each acceptance rate and alpha."""
    print("\n" + "="*60)
    print("Exp5: Optimal Gamma Sweep")
    print("="*60)

    acceptance_rates = np.arange(0.55, 1.001, 0.05)
    alphas = [0.05, 0.08, 0.10, 0.15, 0.20]
    gammas = range(1, 21)

    results = []
    for alpha in alphas:
        for p in acceptance_rates:
            p = min(float(p), 0.999)
            best_gamma = 1
            best_speedup = 0
            for g in gammas:
                expected = 1 + p * (1 - p**g) / (1 - p)
                time_r = g * alpha + 1.0
                su = expected / time_r
                if su > best_speedup:
                    best_speedup = su
                    best_gamma = g

            result = {
                "alpha": alpha,
                "acceptance_rate": round(p, 3),
                "optimal_gamma": best_gamma,
                "max_speedup": round(best_speedup, 3),
            }
            results.append(result)

            if alpha in [0.08, 0.15] and round(p, 2) in [0.60, 0.70, 0.80, 0.90, 1.00]:
                print(f"  alpha={alpha:.2f} p={p:.2f} | optimal_gamma={best_gamma:>2d} | max_speedup={best_speedup:.2f}x")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Speculative Decoding ROI Calculator")
    print(f"PyTorch {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)

    all_results = {}
    all_results["experiment1_acceptance_rate"] = experiment1_acceptance_rate_model()
    all_results["experiment2_draft_cost"] = experiment2_draft_cost_analysis()
    all_results["experiment3_task_simulation"] = experiment3_task_type_simulation()
    all_results["experiment4_gpu_latency"] = experiment4_gpu_latency_simulation()
    all_results["experiment5_optimal_gamma"] = experiment5_optimal_gamma_sweep()

    output_path = os.path.join(RESULTS_DIR, "spec_decoding_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    print("Done!")
