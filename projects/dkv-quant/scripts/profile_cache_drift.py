#!/usr/bin/env python3
"""No-GPU experiment #1: quantify KV-cache drift across denoising steps.

This produces the central motivating figure of the paper: how much do the
per-channel KV statistics move from step to step in a *bidirectional* diffusion
LM?  If the drift is large, AR-style write-once calibration is provably stale and
step-aware recalibration is justified.

Runs on the random-weight mock dLLM, so it needs no checkpoint and no GPU.  The
same `profile_denoising` hook attaches to a real LLaDA/Dream forward later; only
the model object changes.

Usage:
    python scripts/profile_cache_drift.py --layers 3 --gen-len 32 --out results/drift.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dkv_quant.mock_dllm import MockConfig, MockDiffusionLM
from dkv_quant.denoise import DenoiseConfig
from dkv_quant.profiler import profile_denoising


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--gen-len", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--steps-per-block", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--outlier-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/drift.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model = MockDiffusionLM(MockConfig(
        vocab_size=args.vocab, d_model=args.d_model, n_heads=args.heads,
        n_layers=args.layers, max_seq=args.prompt_len + args.gen_len + 4,
        seed=args.seed))
    prompt = rng.integers(0, args.vocab, size=args.prompt_len)
    dcfg = DenoiseConfig(gen_len=args.gen_len, block_size=args.block_size,
                         steps_per_block=args.steps_per_block, seed=args.seed)

    per_layer = {}
    for layer in range(args.layers):
        rep = profile_denoising(model, prompt, dcfg, layer=layer,
                                outlier_frac=args.outlier_frac)
        per_layer[f"layer_{layer}"] = rep.summary()

    # Aggregate across layers.
    keys = ["k_tensor_drift_mean", "v_tensor_drift_mean",
            "k_scale_drift_mean", "v_scale_drift_mean", "outlier_churn_mean"]
    agg = {k: float(np.mean([per_layer[l][k] for l in per_layer])) for k in keys}

    result = {"config": vars(args), "per_layer": per_layer, "aggregate": agg}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("=== KV-cache drift across denoising steps (mock dLLM) ===")
    for k in keys:
        print(f"  {k:24s}: {agg[k]:.4f}")
    print(f"\nInterpretation: a non-trivial scale-drift / outlier-churn means an")
    print(f"AR write-once calibration is stale by late steps -> motivates dKV-Quant.")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
