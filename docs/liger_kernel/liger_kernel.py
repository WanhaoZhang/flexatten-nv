"""
Project 12: Liger Kernel - FusedLinearCrossEntropy Memory Analysis
===================================================================
Demonstrates how fusing Linear + CrossEntropy eliminates the massive
logits tensor [B, S, V], saving enormous memory in LLM training.

Environment: NVIDIA L4 (24GB) | PyTorch 2.6.0+cu124
"""

import torch
import torch.nn as nn
import time
import json
import os
import gc

torch.manual_seed(42)
torch.set_default_device('cuda')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kv_cache_bandwidth", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)


def experiment1_logits_memory_explosion():
    """Exp1: Measure memory spike from logits tensor at different seq_lens."""
    print("\n" + "="*60)
    print("Exp1: Logits Memory Explosion Analysis")
    print("="*60)

    # Typical LLM config
    hidden_dim = 4096   # 7B model
    vocab_size = 151936  # Qwen2.5 tokenizer
    batch_size = 1

    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]

    results = []
    for seq_len in seq_lengths:
        # Hidden states from transformer
        hidden = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, requires_grad=True)
        # Language model head
        lm_head = nn.Linear(hidden_dim, vocab_size, bias=False).half()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Forward: compute logits
        logits = lm_head(hidden)  # [B, S, V] — THIS IS THE PROBLEM
        peak_with_logits = torch.cuda.max_memory_allocated() / 1024**2

        # Compute loss
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss = nn.functional.cross_entropy(
            logits.view(-1, vocab_size).float(),
            targets.view(-1)
        )

        peak_with_loss = torch.cuda.max_memory_allocated() / 1024**2

        # Backward
        loss.backward()
        peak_total = torch.cuda.max_memory_allocated() / 1024**2

        # Theoretical logits size
        logits_theoretical_mb = batch_size * seq_len * vocab_size * 2 / 1024**2  # FP16

        result = {
            "seq_len": seq_len,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "logits_tensor_mb": round(logits_theoretical_mb, 1),
            "peak_before_logits_mb": round(torch.cuda.memory_allocated() / 1024**2 - logits_theoretical_mb, 1),
            "peak_with_logits_mb": round(peak_with_logits, 1),
            "peak_total_mb": round(peak_total, 1),
            "logits_pct_of_total": round(logits_theoretical_mb / peak_total * 100, 1),
        }
        results.append(result)
        print(f"  seq={seq_len:>5d} | logits={logits_theoretical_mb:>8.1f}MB | peak={peak_total:.0f}MB | logits占比={result['logits_pct_of_total']:.1f}%")

        del hidden, lm_head, logits, loss, targets
        gc.collect()
        torch.cuda.empty_cache()

    return results


def experiment2_fusion_simulation():
    """Exp2: Simulate fused Linear+CE by chunked computation to avoid full logits."""
    print("\n" + "="*60)
    print("Exp2: Chunked CE (Fusion Simulation) Memory Savings")
    print("="*60)

    hidden_dim = 4096
    vocab_size = 151936
    batch_size = 1

    seq_lengths = [256, 512, 1024, 2048, 4096]
    chunk_sizes = [1, 4, 16, 64]  # process N tokens at a time

    results = []
    for seq_len in seq_lengths:
        # Standard (unfused) baseline
        hidden = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, requires_grad=True)
        lm_head_weight = torch.randn(vocab_size, hidden_dim, dtype=torch.float16)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Unfused: compute full logits
        torch.cuda.reset_peak_memory_stats()
        logits = torch.nn.functional.linear(hidden, lm_head_weight)  # [B, S, V]
        loss = nn.functional.cross_entropy(
            logits.view(-1, vocab_size).float(), targets.view(-1)
        )
        loss.backward()
        peak_unfused = torch.cuda.max_memory_allocated() / 1024**2

        del hidden, logits, loss
        gc.collect()
        torch.cuda.empty_cache()

        # Chunked (simulates fusion): compute logits chunk by chunk
        for chunk_size in chunk_sizes:
            hidden = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, requires_grad=True)
            torch.cuda.reset_peak_memory_stats()

            total_loss = 0.0
            n_chunks = (seq_len + chunk_size - 1) // chunk_size
            for i in range(n_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, seq_len)
                h_chunk = hidden[:, start:end, :]  # [B, chunk, D]
                logits_chunk = torch.nn.functional.linear(h_chunk, lm_head_weight)  # [B, chunk, V]
                targets_chunk = targets[:, start:end]
                loss_chunk = nn.functional.cross_entropy(
                    logits_chunk.view(-1, vocab_size).float(), targets_chunk.view(-1)
                )
                total_loss = total_loss + loss_chunk * (end - start)

            total_loss = total_loss / seq_len
            total_loss.backward()
            peak_chunked = torch.cuda.max_memory_allocated() / 1024**2

            saving_pct = (1 - peak_chunked / peak_unfused) * 100

            result = {
                "seq_len": seq_len,
                "chunk_size": chunk_size,
                "peak_unfused_mb": round(peak_unfused, 1),
                "peak_chunked_mb": round(peak_chunked, 1),
                "saving_pct": round(saving_pct, 1),
            }
            results.append(result)
            print(f"  seq={seq_len:>5d} chunk={chunk_size:>2d} | unfused={peak_unfused:.0f}MB | chunked={peak_chunked:.0f}MB | saving={saving_pct:.1f}%")

            del hidden, total_loss
            gc.collect()
            torch.cuda.empty_cache()

    return results


def experiment3_max_seq_before_oom():
    """Exp3: Find maximum seq_len before OOM for unfused vs chunked."""
    print("\n" + "="*60)
    print("Exp3: Maximum Seq Length Before OOM")
    print("="*60)

    hidden_dim = 4096
    vocab_size = 151936
    batch_size = 1

    results = []
    for method in ["unfused", "chunked_16", "chunked_1"]:
        max_seq = 0
        for seq_len in [256, 512, 1024, 2048, 4096, 8192, 16384]:
            try:
                gc.collect()
                torch.cuda.empty_cache()

                hidden = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float16, requires_grad=True)
                lm_head_weight = torch.randn(vocab_size, hidden_dim, dtype=torch.float16)
                targets = torch.randint(0, vocab_size, (batch_size, seq_len))

                torch.cuda.reset_peak_memory_stats()

                if method == "unfused":
                    logits = torch.nn.functional.linear(hidden, lm_head_weight)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, vocab_size).float(), targets.view(-1)
                    )
                else:
                    cs = int(method.split("_")[1])
                    total_loss = 0.0
                    n_chunks = (seq_len + cs - 1) // cs
                    for i in range(n_chunks):
                        s = i * cs
                        e = min(s + cs, seq_len)
                        lc = torch.nn.functional.linear(hidden[:, s:e, :], lm_head_weight)
                        tc = targets[:, s:e]
                        total_loss = total_loss + nn.functional.cross_entropy(
                            lc.view(-1, vocab_size).float(), tc.view(-1)
                        ) * (e - s)
                    loss = total_loss / seq_len

                loss.backward()
                peak = torch.cuda.max_memory_allocated() / 1024**2
                max_seq = seq_len

                result_entry = {
                    "method": method, "seq_len": seq_len,
                    "peak_mb": round(peak, 1), "oom": False,
                }
                results.append(result_entry)

                del hidden, lm_head_weight, targets, loss
                gc.collect()
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                result_entry = {"method": method, "seq_len": seq_len, "oom": True}
                results.append(result_entry)
                print(f"  {method:12s} seq={seq_len:>5d} | OOM")
                gc.collect()
                torch.cuda.empty_cache()
                break

        print(f"  {method:12s} max_seq_before_oom = {max_seq}")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Liger Kernel Analysis: FusedLinearCrossEntropy")
    print(f"PyTorch {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)

    all_results = {}
    all_results["experiment1_logits_memory"] = experiment1_logits_memory_explosion()
    all_results["experiment2_fusion_simulation"] = experiment2_fusion_simulation()
    all_results["experiment3_max_seq_oom"] = experiment3_max_seq_before_oom()

    output_path = os.path.join(RESULTS_DIR, "liger_kernel_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    print("Done!")
