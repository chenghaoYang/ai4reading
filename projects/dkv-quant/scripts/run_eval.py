#!/usr/bin/env python3
"""GPU phase: real KV-cache quantization eval on LLaDA / Dream.

This is the ONLY script that needs a GPU. It is intentionally written so that
everything *around* the GPU call (config parsing, method registry, metric
collection, output schema) is shared with the no-GPU simulation, so moving to
GPU is a model swap, not a rewrite.

It refuses to run pointlessly: if torch/transformers/CUDA are missing it prints
the exact setup needed and exits, rather than failing deep in a stack trace.

The quantization itself reuses `dkv_quant.quantizers`: a forward hook captures
each layer's K/V right after projection and replaces them with their
fake-quantized version under the chosen method, so the simulated-quantization
algorithm validated on CPU runs unchanged against real activations. (Swap to
true low-bit kernels — bitsandbytes / a Triton attention — once the accuracy
story is locked; that is a separate, later systems task.)

Usage (on a GPU box):
    pip install -r requirements.txt
    python scripts/run_eval.py \
        --model GSAI-ML/LLaDA-8B-Instruct \
        --method dkv_step_aware --bits 4 --group-size 128 \
        --benchmark gsm8k --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dkv_quant.quantizers import QuantConfig, build_quantizer

# Method name -> quantizer-builder spec. Mirrors baselines.default_methods.
_METHOD_TO_BUILDER = {
    "fp16": None,
    "ar_static": ("ar_static", {}),
    "ar_static_hadamard": ("ar_static", {"hadamard": True}),
    "fresh_per_step": ("fresh", {}),
    "dkv_step_aware": ("step_aware", {}),
    "dkv_step_aware_hadamard": ("step_aware", {"hadamard": True}),
}


def _require_gpu_stack() -> "module":  # type: ignore[name-defined]
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")
    if missing:
        print("This script needs the GPU stack, which is not installed here.")
        print(f"  missing: {', '.join(missing)}")
        print("  install: pip install -r requirements.txt")
        print("  then run on a machine with a CUDA GPU (>=16GB for LLaDA-8B 4bit).")
        sys.exit(2)
    import torch
    if not torch.cuda.is_available():
        print("No CUDA device visible. This is the GPU-only phase of the project.")
        print("Run the no-GPU experiments instead:")
        print("  python scripts/profile_cache_drift.py")
        print("  python scripts/simulate_quant.py")
        sys.exit(3)
    return torch


def build_quant_config(args) -> QuantConfig:
    spec = _METHOD_TO_BUILDER[args.method]
    extra = spec[1] if spec else {}
    return QuantConfig(
        bits=args.bits,
        granularity="per_group",
        group_size=args.group_size,
        symmetric=False,
        hadamard=extra.get("hadamard", False),
    )


def make_kv_hook(qcfg: QuantConfig, method_key: str):
    """Return a function that fake-quantizes a captured (k, v) numpy pair.

    On GPU you register this inside the attention module to rewrite the cached
    K/V each denoising step. Kept framework-light here so it is unit-testable.
    """
    spec = _METHOD_TO_BUILDER[method_key]
    if spec is None:  # fp16 — identity
        return lambda k, v: (k, v)
    name = spec[0]
    qk = build_quantizer(name, qcfg)
    qv = build_quantizer(name, qcfg)

    def hook(k, v):
        return qk(k), qv(v)

    return hook


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct",
                    help="HF id: GSAI-ML/LLaDA-8B-Instruct (MIT) or "
                         "Dream-org/Dream-v0-Instruct-7B (Apache-2.0)")
    ap.add_argument("--method", default="dkv_step_aware",
                    choices=list(_METHOD_TO_BUILDER))
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--benchmark", default="gsm8k")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--out", default="results/eval.json")
    args = ap.parse_args()

    torch = _require_gpu_stack()  # exits cleanly if no GPU/stack

    # ---- The GPU body below is reached only on a real GPU box. ----
    from transformers import AutoModel, AutoTokenizer  # noqa: F401

    qcfg = build_quant_config(args)
    kv_hook = make_kv_hook(qcfg, args.method)  # noqa: F841  (wired into attention)

    print(f"[run_eval] model={args.model} method={args.method} "
          f"bits={args.bits} benchmark={args.benchmark}")
    print("TODO(GPU): (1) load model+tokenizer with trust_remote_code=True; "
          "(2) register kv_hook on each attention layer to rewrite cached K/V; "
          "(3) run the diffusion denoising loop (reuse Fast-dLLM's cache path, "
          "NVlabs/Fast-dLLM, Apache-2.0); (4) score with lm-eval-harness; "
          "(5) record GenerationStats (throughput / NFE / peak mem) + accuracy.")
    # Intentionally not fabricating numbers. The harness, methods, metrics, and
    # data are all prepared; only the GPU forward pass remains.

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "status": "scaffold-ready",
                   "note": "wire model forward + lm-eval, then fill metrics"},
                  f, indent=2)


if __name__ == "__main__":
    main()
