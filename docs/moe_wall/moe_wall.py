"""
Project 9: MoE Inference Wall — Expert Routing & Bandwidth Analysis
====================================================================
Mathematical modeling + GPU simulation of MoE decode-phase bandwidth
bottleneck caused by sparse expert routing.

Key insight: Each token routes to different experts, causing uncoalesced
memory access patterns that destroy GPU memory bandwidth efficiency.

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

L4_MEMORY_BANDWIDTH = 300  # GB/s


def experiment1_expert_weight_footprint():
    """Exp1: Model weight footprint comparison — Dense vs MoE.

    MoE models have N_experts × FFN weights. Only top-K experts are active
    per token, but ALL expert weights must reside in GPU memory.
    """
    print("\n" + "="*60)
    print("Exp1: Expert Weight Footprint Analysis")
    print("="*60)

    configs = [
        {"name": "Dense-7B", "hidden_dim": 4096, "ffn_dim": 11008, "n_layers": 32, "n_experts": 1},
        {"name": "MoE-A2.7B (8 experts)", "hidden_dim": 2048, "ffn_dim": 5632, "n_layers": 28, "n_experts": 8},
        {"name": "MoE-A2.7B (64 experts)", "hidden_dim": 2048, "ffn_dim": 1408, "n_layers": 28, "n_experts": 64},
        {"name": "DeepSeek-V2-Lite (64 exp)", "hidden_dim": 2048, "ffn_dim": 1408, "n_layers": 28, "n_experts": 64},
        {"name": "Mixtral-8x7B", "hidden_dim": 4096, "ffn_dim": 14336, "n_layers": 32, "n_experts": 8},
    ]

    results = []
    for cfg in configs:
        D = cfg["hidden_dim"]
        FFN = cfg["ffn_dim"]
        L = cfg["n_layers"]
        E = cfg["n_experts"]

        # Attention params: 4 * D^2 (QKV + output)
        attn_params = 4 * D * D * L
        # FFN params per expert: 3 * D * FFN (gate+up+down)
        ffn_params_per_expert = 3 * D * FFN
        # Total FFN params = n_experts * per_expert
        total_ffn_params = E * ffn_params_per_expert * L
        total_params = attn_params + total_ffn_params
        total_gb = total_params * 2 / (1024**3)  # FP16

        # Active params per token (top-2 routing)
        active_params = attn_params + 2 * ffn_params_per_expert * L
        active_gb = active_params * 2 / (1024**3)

        # Decode bandwidth: must read active params per token
        decode_time_us = active_gb * 1024 / L4_MEMORY_BANDWIDTH * 1e6  # us
        decode_tok_per_s = 1e6 / decode_time_us if decode_time_us > 0 else 0

        result = {
            "name": cfg["name"],
            "total_params_B": round(total_params / 1e9, 2),
            "total_weights_gb": round(total_gb, 2),
            "active_params_B": round(active_params / 1e9, 2),
            "active_weights_gb": round(active_gb, 2),
            "sparsity_ratio": round(1 - active_params / total_params, 3),
            "decode_time_us": round(decode_time_us, 1),
            "decode_tok_per_s": round(decode_tok_per_s, 1),
        }
        results.append(result)
        print(f"  {cfg['name']:30s} | total={total_gb:.1f}GB | active={active_gb:.1f}GB | sparsity={result['sparsity_ratio']:.1%} | decode={decode_tok_per_s:.0f} tok/s")

    return results


def experiment2_expert_routing_simulation():
    """Exp2: Simulate expert routing patterns and measure GPU access patterns.

    In MoE decode, each token routes to top-K experts. When processing a batch:
    - Dense model: all tokens read same weights (coalesced access)
    - MoE: tokens scatter to different experts (uncoalesced access)

    We simulate this by measuring gather/scatter latency.
    """
    print("\n" + "="*60)
    print("Exp2: Expert Routing Memory Access Simulation")
    print("="*60)

    n_experts = 64
    expert_dim = 1408  # typical FFN hidden dim for small MoE
    model_dim = 2048
    top_k = 2

    batch_sizes = [1, 4, 8, 16, 32]
    results = []

    for B in batch_sizes:
        # Simulate expert weights: [n_experts, expert_dim, model_dim]
        expert_weights = torch.randn(n_experts, expert_dim, model_dim, dtype=torch.float16)
        # Input: [B, model_dim]
        x = torch.randn(B, model_dim, dtype=torch.float16)
        # Routing: random top-K selection
        route_indices = torch.randint(0, n_experts, (B, top_k))

        # Warmup
        for _ in range(5):
            # Gather expert weights for each token
            selected = expert_weights[route_indices]  # [B, top_k, expert_dim, model_dim]
            # Compute: each token does matmul with its selected expert
            out = torch.matmul(selected, x.unsqueeze(-1))  # [B, top_k, expert_dim, 1]
        torch.cuda.synchronize()

        # Measure MoE-style scatter access
        times_moe = []
        for _ in range(100):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            selected = expert_weights[route_indices]
            out = torch.matmul(selected, x.unsqueeze(-1))
            torch.cuda.synchronize()
            times_moe.append((time.perf_counter() - t0) * 1e6)
        moe_us = np.median(times_moe)

        # Baseline: Dense model (single expert, same compute)
        dense_weight = torch.randn(expert_dim, model_dim, dtype=torch.float16)
        for _ in range(5):
            out_dense = F.linear(x, dense_weight)
        torch.cuda.synchronize()

        times_dense = []
        for _ in range(100):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out_dense = F.linear(x, dense_weight)
            torch.cuda.synchronize()
            times_dense.append((time.perf_counter() - t0) * 1e6)
        dense_us = np.median(times_dense)

        # Memory read comparison
        moe_read_bytes = B * top_k * expert_dim * model_dim * 2
        dense_read_bytes = B * expert_dim * model_dim * 2

        result = {
            "batch_size": B,
            "n_experts": n_experts,
            "top_k": top_k,
            "moe_latency_us": round(float(moe_us), 1),
            "dense_latency_us": round(float(dense_us), 1),
            "moe_dense_ratio": round(float(moe_us / dense_us), 2) if dense_us > 0 else 0,
            "moe_read_mb": round(moe_read_bytes / 1e6, 2),
            "dense_read_mb": round(dense_read_bytes / 1e6, 2),
        }
        results.append(result)
        print(f"  B={B:>2d} | MoE={moe_us:.1f}us | Dense={dense_us:.1f}us | ratio={moe_us/dense_us:.2f}x | MoE_read={moe_read_bytes/1e6:.1f}MB Dense_read={dense_read_bytes/1e6:.1f}MB")

        del expert_weights, x, route_indices, selected, out, dense_weight, out_dense
        torch.cuda.empty_cache()

    return results


def experiment3_load_balance_analysis():
    """Exp3: Load balance across experts — skewed routing hurts batching.

    If routing is imbalanced, some experts process many tokens while
    others sit idle. This reduces effective parallelism.
    """
    print("\n" + "="*60)
    print("Exp3: Load Balance Analysis")
    print("="*60)

    n_experts = 64
    batch_size = 64
    top_k = 2

    # Different routing distributions
    distributions = {
        "Uniform": np.ones(n_experts) / n_experts,
        "Mild skew (Zipf-1.1)": None,  # computed below
        "Heavy skew (Zipf-1.5)": None,
        "Extreme skew (80/20)": None,
    }

    # Compute Zipf distributions
    ranks = np.arange(1, n_experts + 1)
    distributions["Mild skew (Zipf-1.1)"] = ranks ** (-1.1)
    distributions["Mild skew (Zipf-1.1)"] /= distributions["Mild skew (Zipf-1.1)"].sum()
    distributions["Heavy skew (Zipf-1.5)"] = ranks ** (-1.5)
    distributions["Heavy skew (Zipf-1.5)"] /= distributions["Heavy skew (Zipf-1.5)"].sum()
    # 80/20: top 20% of experts get 80% of traffic
    extreme = np.ones(n_experts) * 0.2
    extreme[:n_experts//5] = 0.8 * 5 / n_experts * n_experts // (n_experts // 5)
    extreme /= extreme.sum()
    distributions["Extreme skew (80/20)"] = extreme

    results = []
    for dist_name, probs in distributions.items():
        # Simulate routing decisions
        n_tokens = batch_size * top_k
        expert_counts = np.zeros(n_experts)
        for _ in range(1000):  # 1000 decode steps
            tokens = np.random.choice(n_experts, size=n_tokens, p=probs)
            for e in tokens:
                expert_counts[e] += 1

        # Load balance metrics
        max_load = expert_counts.max()
        min_load = expert_counts[expert_counts > 0].min()
        mean_load = expert_counts.mean()
        load_balance_ratio = min_load / max_load  # 1.0 = perfect balance

        # Effective parallelism: if expert i has c_i tokens, effective batch = max(c_i)
        # because all experts must wait for the slowest one
        effective_batch = max_load / mean_load  # > 1 means wasted compute

        # Utilization: fraction of experts that are actually used
        active_experts = (expert_counts > 0).sum()
        utilization = active_experts / n_experts

        result = {
            "distribution": dist_name,
            "max_load": int(max_load),
            "min_load": int(min_load),
            "load_balance_ratio": round(float(load_balance_ratio), 3),
            "effective_batch_ratio": round(float(effective_batch), 3),
            "expert_utilization": round(float(utilization), 3),
            "wasted_compute_pct": round(float((1 - load_balance_ratio) * 100), 1),
        }
        results.append(result)
        print(f"  {dist_name:30s} | balance={load_balance_ratio:.3f} | utilization={utilization:.1%} | wasted={result['wasted_compute_pct']:.1f}%")

    return results


def experiment4_dense_vs_moe_latency():
    """Exp4: Actual GPU timing — Dense FFN vs MoE FFN on L4."""
    print("\n" + "="*60)
    print("Exp4: Dense vs MoE FFN Latency (GPU实测)")
    print("="*60)

    model_dim = 2048
    ffn_dim = 5632
    n_experts = 8

    batch_sizes = [1, 2, 4, 8, 16]
    results = []

    for B in batch_sizes:
        # Dense FFN
        w_gate = torch.randn(ffn_dim, model_dim, dtype=torch.float16)
        w_up = torch.randn(ffn_dim, model_dim, dtype=torch.float16)
        w_down = torch.randn(model_dim, ffn_dim, dtype=torch.float16)
        x = torch.randn(B, model_dim, dtype=torch.float16)

        # Warmup
        for _ in range(10):
            h = F.silu(F.linear(x, w_gate)) * F.linear(x, w_up)
            out = F.linear(h, w_down)
        torch.cuda.synchronize()

        times_dense = []
        for _ in range(100):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            h = F.silu(F.linear(x, w_gate)) * F.linear(x, w_up)
            out = F.linear(h, w_down)
            torch.cuda.synchronize()
            times_dense.append((time.perf_counter() - t0) * 1e6)
        dense_us = np.median(times_dense)

        del w_gate, w_up, w_down, x, h, out
        torch.cuda.empty_cache()

        # MoE FFN (8 experts, top-2 routing)
        expert_gates = torch.randn(n_experts, ffn_dim, model_dim, dtype=torch.float16)
        expert_ups = torch.randn(n_experts, ffn_dim, model_dim, dtype=torch.float16)
        expert_downs = torch.randn(n_experts, model_dim, ffn_dim, dtype=torch.float16)
        x = torch.randn(B, model_dim, dtype=torch.float16)
        router_logits = torch.randn(B, n_experts, dtype=torch.float16)
        top_k_indices = router_logits.topk(2, dim=-1).indices  # [B, 2]

        # Warmup
        for _ in range(5):
            out_moe = torch.zeros(B, model_dim, dtype=torch.float16)
            for b in range(B):
                for k_idx in range(2):
                    e = top_k_indices[b, k_idx].item()
                    h_b = x[b:b+1]
                    h_b = F.silu(F.linear(h_b, expert_gates[e])) * F.linear(h_b, expert_ups[e])
                    out_moe[b] += F.linear(h_b, expert_downs[e]).squeeze(0)
        torch.cuda.synchronize()

        times_moe = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out_moe = torch.zeros(B, model_dim, dtype=torch.float16)
            for b in range(B):
                for k_idx in range(2):
                    e = top_k_indices[b, k_idx].item()
                    h_b = x[b:b+1]
                    h_b = F.silu(F.linear(h_b, expert_gates[e])) * F.linear(h_b, expert_ups[e])
                    out_moe[b] += F.linear(h_b, expert_downs[e]).squeeze(0)
            torch.cuda.synchronize()
            times_moe.append((time.perf_counter() - t0) * 1e6)
        moe_us = np.median(times_moe)

        result = {
            "batch_size": B,
            "dense_us": round(float(dense_us), 1),
            "moe_us": round(float(moe_us), 1),
            "moe_overhead_ratio": round(float(moe_us / dense_us), 2),
        }
        results.append(result)
        print(f"  B={B:>2d} | Dense={dense_us:.1f}us | MoE={moe_us:.1f}us | MoE/Dense={moe_us/dense_us:.1f}x")

        del expert_gates, expert_ups, expert_downs, x, router_logits, top_k_indices, out_moe
        torch.cuda.empty_cache()

    return results


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MoE Inference Wall Analysis")
    print(f"PyTorch {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)

    all_results = {}
    all_results["experiment1_weight_footprint"] = experiment1_expert_weight_footprint()
    all_results["experiment2_routing_simulation"] = experiment2_expert_routing_simulation()
    all_results["experiment3_load_balance"] = experiment3_load_balance_analysis()
    all_results["experiment4_dense_vs_moe"] = experiment4_dense_vs_moe_latency()

    output_path = os.path.join(RESULTS_DIR, "moe_wall_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    print("Done!")
