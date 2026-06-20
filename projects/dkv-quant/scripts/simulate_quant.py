#!/usr/bin/env python3
"""No-GPU experiment #2: simulated KV-cache quantization sweep.

Replays the denoising schedule and, at each step, quantizes the freshly computed
KV with every method in the registry, recording the reconstruction error.  This
isolates the algorithmic claim — *step-aware calibration reconstructs the dLLM
cache better than frozen AR calibration at the same bit-width* — with no GPU and
no real weights.

Outputs a per-method error table (relative-L2, SNR) and the effective
compression ratio, which is the skeleton of the paper's main results table.  On
GPU you swap `MockDiffusionLM` for LLaDA/Dream and swap the reconstruction error
for downstream task accuracy; the harness is identical.

Usage:
    python scripts/simulate_quant.py --bits 4 --group-size 128 --out results/quant_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dkv_quant.mock_dllm import MockConfig, MockDiffusionLM, MASK_ID
from dkv_quant.denoise import DenoiseConfig, _confidence
from dkv_quant.baselines import default_methods
from dkv_quant.metrics import relative_l2, snr_db, compression_ratio


def _collect_kv_stream(model, prompt, dcfg, layer):
    """Replay denoising; yield the (k, v) seen at each forward pass for a layer."""
    prompt = np.asarray(prompt, np.int64).reshape(-1)
    T_p = prompt.shape[0]
    seq = np.concatenate([prompt, np.full(dcfg.gen_len, MASK_ID, np.int64)])
    stream = []
    n_blocks = (dcfg.gen_len + dcfg.block_size - 1) // dcfg.block_size
    for b in range(n_blocks):
        lo = T_p + b * dcfg.block_size
        hi = min(lo + dcfg.block_size, T_p + dcfg.gen_len)
        block_idx = np.arange(lo, hi)
        for step in range(dcfg.steps_per_block):
            logits, kv = model.forward(seq[None, :], capture=True)
            stream.append((kv[layer]["k"].copy(), kv[layer]["v"].copy()))
            pred, conf = _confidence(logits[0])
            masked = block_idx[seq[block_idx] == MASK_ID]
            if masked.size == 0:
                break
            keep = max(1, int(np.ceil(masked.size * (step + 1) / dcfg.steps_per_block)))
            order = masked[np.argsort(-conf[masked])]
            seq[order[:keep]] = pred[order[:keep]]
        residual = block_idx[seq[block_idx] == MASK_ID]
        if residual.size:
            logits = model.forward(seq[None, :])
            pred, _ = _confidence(logits[0])
            seq[residual] = pred[residual]
    return stream


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--momentum", type=float, default=0.5)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--gen-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/quant_sweep.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model = MockDiffusionLM(MockConfig(
        vocab_size=256, d_model=args.d_model, n_heads=args.heads,
        n_layers=args.layers, max_seq=args.gen_len + 16, seed=args.seed))
    prompt = rng.integers(0, 256, size=8)
    dcfg = DenoiseConfig(gen_len=args.gen_len, block_size=8, steps_per_block=8,
                         seed=args.seed)
    # gs=0 for per-channel methods is fine; group_size only used by per_group.
    gs = args.group_size if args.group_size <= args.d_model // args.heads else \
        args.d_model // args.heads

    methods = default_methods(bits=args.bits, group_size=gs, momentum=args.momentum)

    # Average error over all layers.
    table = {}
    for meth in methods:
        if meth.name == "fp16":
            table[meth.name] = {"rel_l2": 0.0, "snr_db": float("inf"),
                                "bits": 16.0, "compression": 1.0,
                                "is_ours": False, "desc": meth.description}
            continue
        per_layer_err, per_layer_snr = [], []
        for layer in range(args.layers):
            stream = _collect_kv_stream(model, prompt, dcfg, layer)
            qk = meth.make_quantizer()  # for K
            qv = meth.make_quantizer()  # for V (separate running stats)
            errs, snrs = [], []
            for k, v in stream:
                kq, vq = qk(k), qv(v)
                errs.append(0.5 * (relative_l2(k, kq) + relative_l2(v, vq)))
                snrs.append(0.5 * (snr_db(k, kq) + snr_db(v, vq)))
            per_layer_err.append(np.mean(errs))
            per_layer_snr.append(np.mean(snrs))
        cr = compression_ratio(bits=meth.bits,
                               group_size=gs if "group" else 0)
        table[meth.name] = {
            "rel_l2": float(np.mean(per_layer_err)),
            "snr_db": float(np.mean(per_layer_snr)),
            "bits": meth.bits, "compression": round(cr, 2),
            "is_ours": meth.is_ours, "desc": meth.description,
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": table}, f, indent=2)

    print(f"=== Simulated KV quantization @ {args.bits}-bit "
          f"(mock dLLM, {args.layers} layers) ===")
    print(f"{'method':26s} {'rel_L2 (lower=better)':>22s} {'SNR dB':>9s} {'comp':>6s}")
    for name, r in table.items():
        tag = "  <-- ours" if r["is_ours"] else ""
        snr = "inf" if r["snr_db"] == float("inf") else f"{r['snr_db']:.1f}"
        print(f"{name:26s} {r['rel_l2']:>22.4f} {snr:>9s} {r['compression']:>5.2f}x{tag}")
    print(f"\nSaved -> {args.out}")
    print("Expectation: dkv_step_aware* < ar_static* in rel_L2 at equal bits.")


if __name__ == "__main__":
    main()
