#!/usr/bin/env python3
"""Fix: Exp2 Document Packing + Exp4 Sparsity (Document masks)"""

import torch
import torch.nn.functional as F
import json
import time
import gc
from pathlib import Path

device = "cuda"
dtype = torch.float16

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def benchmark(fn, warmup=3, runs=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return {"mean_ms": round(sum(times)/len(times), 3), "min_ms": round(min(times), 3), "max_ms": round(max(times), 3)}

def peak_mem(fn):
    clear_cache()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    mem_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    del out
    return mem_gb

# ================================================================
# Exp2: Document Packing + Causal (FIXED)
# ================================================================
def exp2_document_packing():
    print("\n" + "="*70)
    print("  Experiment 2: Document Packing + Causal (FIXED)")
    print("="*70)
    H, D, B = 8, 64, 1
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    rows = []
    for S in [1024, 2048, 4096, 8192]:
        for num_docs in [2, 4, 8, 16]:
            if S // num_docs < 128:
                continue
            print(f"\n  [S={S}, docs={num_docs}]")
            clear_cache()
            doc_ids = (torch.arange(S, device=device) // (S // num_docs)).clamp(max=num_docs-1)
            q = torch.randn(B, H, S, D, device=device, dtype=dtype)
            k = torch.randn(B, H, S, D, device=device, dtype=dtype)
            v = torch.randn(B, H, S, D, device=device, dtype=dtype)

            # Dense
            dense_ok = True
            try:
                def dense_fn():
                    sc = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
                    cm = torch.ones(S, S, device=device, dtype=torch.bool).tril_()
                    dm = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)
                    sc = sc.masked_fill(~(cm & dm), float('-inf'))
                    return torch.matmul(F.softmax(sc.float(), dim=-1).to(dtype), v)
                mem_d = peak_mem(dense_fn)
                t_d = benchmark(dense_fn, warmup=1, runs=3)
                print(f"    Dense: {t_d['mean_ms']:>8.3f} ms | {mem_d:>6.3f} GB")
            except RuntimeError:
                print(f"    Dense: OOM"); dense_ok = False; mem_d = None; t_d = None
                clear_cache()

            # Flex - use mask_mod (returns bool), NOT score_mod
            flex_ok = True
            sparsity = None
            try:
                # mask_mod: returns True if position should be attended
                def make_packed_mask(doc_ids_tensor):
                    def packed_mask(b, h, q_idx, kv_idx):
                        causal_ok = q_idx >= kv_idx
                        doc_ok = doc_ids_tensor[q_idx] == doc_ids_tensor[kv_idx]
                        return causal_ok & doc_ok
                    return packed_mask

                mask_fn = make_packed_mask(doc_ids)
                bm = create_block_mask(mask_fn, B, 1, S, S, device=device)
                total_blks = bm.kv_num_blocks.sum().item()
                max_blks = B * (S // 128) * (S // 128)
                sparsity = round(1.0 - total_blks / max_blks, 4) if max_blks > 0 else 0
                _ = flex_attention(q, k, v, block_mask=bm)
                torch.cuda.synchronize()
                def flex_fn():
                    return flex_attention(q, k, v, block_mask=bm)
                mem_f = peak_mem(flex_fn)
                t_f = benchmark(flex_fn, warmup=2, runs=5)
                print(f"    Flex:  {t_f['mean_ms']:>8.3f} ms | {mem_f:>6.3f} GB | sparse={sparsity:.1%}")
            except RuntimeError as e:
                print(f"    Flex:  OOM/Error: {e}"); flex_ok = False; mem_f = None; t_f = None
                clear_cache()

            diff = None
            if dense_ok and flex_ok:
                o1 = dense_fn()
                o2 = flex_attention(q, k, v, block_mask=bm)
                diff = round((o1.float() - o2.float()).abs().max().item(), 6)
                print(f"    Max diff: {diff}")

            rows.append({"S": S, "num_docs": num_docs,
                          "dense": {"time": t_d, "mem": mem_d},
                          "flex": {"time": t_f, "mem": mem_f},
                          "sparsity": sparsity, "max_diff": diff})
            del q, k, v; clear_cache()
    return rows

# ================================================================
# Exp4: BlockMask Sparsity (FIXED - Document masks)
# ================================================================
def exp4_sparsity_fixed():
    print("\n" + "="*70)
    print("  Experiment 4: BlockMask Sparsity vs Speed (FIXED)")
    print("="*70)
    H, D, B = 8, 64, 1
    S = 2048
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    max_blks = B * (S // 128) ** 2

    masks = {}

    # Causal
    masks["Causal"] = lambda: create_block_mask(
        lambda b, h, q_idx, kv_idx: q_idx >= kv_idx, B, 1, S, S, device=device)

    # Sliding Window 128
    masks["SlidingWindow(128)"] = lambda: create_block_mask(
        lambda b, h, q_idx, kv_idx: (q_idx >= kv_idx) & ((q_idx - kv_idx) <= 128),
        B, 1, S, S, device=device)

    # Sliding Window 256
    masks["SlidingWindow(256)"] = lambda: create_block_mask(
        lambda b, h, q_idx, kv_idx: (q_idx >= kv_idx) & ((q_idx - kv_idx) <= 256),
        B, 1, S, S, device=device)

    # Sliding Window 512
    masks["SlidingWindow(512)"] = lambda: create_block_mask(
        lambda b, h, q_idx, kv_idx: (q_idx >= kv_idx) & ((q_idx - kv_idx) <= 512),
        B, 1, S, S, device=device)

    # Document masks (FIXED - use mask_mod returning bool)
    for nd in [4, 8, 16]:
        did = (torch.arange(S, device=device) // (S // nd)).clamp(max=nd-1)
        def make_doc_mask(doc_ids_tensor):
            def doc_mask(b, h, q_idx, kv_idx):
                return (q_idx >= kv_idx) & (doc_ids_tensor[q_idx] == doc_ids_tensor[kv_idx])
            return doc_mask
        masks[f"Document({nd})"] = (lambda d=did, f=make_doc_mask: (lambda: create_block_mask(f(d), B, 1, S, S, device=device)))()

    # Prefix LM 25%
    masks["PrefixLM(25%)"] = lambda: create_block_mask(
        lambda b, h, q_idx, kv_idx: (kv_idx < S//4) | (q_idx >= kv_idx),
        B, 1, S, S, device=device)

    # Prefix LM 50%
    masks["PrefixLM(50%)"] = lambda: create_block_mask(
        lambda b, h, q_idx, kv_idx: (kv_idx < S//2) | (q_idx >= kv_idx),
        B, 1, S, S, device=device)

    rows = []
    for name, mk_mask in masks.items():
        print(f"\n  [{name}]")
        try:
            bm = mk_mask()
            sparsity = round(1.0 - bm.kv_num_blocks.sum().item() / max_blks, 4)
            _ = flex_attention(q, k, v, block_mask=bm); torch.cuda.synchronize()
            t = benchmark(lambda: flex_attention(q, k, v, block_mask=bm), warmup=3, runs=10)
            print(f"    Sparsity: {sparsity:.1%} | Time: {t['mean_ms']:.3f} ms")
            rows.append({"mask": name, "sparsity": sparsity, "time_ms": t['mean_ms']})
        except Exception as e:
            print(f"    Error: {e}")

    # SDPA baseline
    t_sdpa = benchmark(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    rows.append({"mask": "SDPA_Causal_Baseline", "sparsity": 0.5, "time_ms": t_sdpa['mean_ms']})
    print(f"\n  [SDPA Causal Baseline] {t_sdpa['mean_ms']:.3f} ms")

    del q, k, v; clear_cache()
    return rows

if __name__ == "__main__":
    all_results = {}

    try:
        all_results["exp2_document_packing"] = exp2_document_packing()
    except Exception as e:
        print(f"exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2_document_packing"] = {"error": str(e)}

    try:
        all_results["exp4_sparsity_analysis"] = exp4_sparsity_fixed()
    except Exception as e:
        print(f"exp4 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp4_sparsity_analysis"] = {"error": str(e)}

    out = Path(__file__).parent / "experiment_results_fix.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out}")
    print("  DONE!")
