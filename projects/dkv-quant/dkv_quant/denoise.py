"""Reverse denoising / semi-autoregressive block decoding for the mock dLLM.

Implements the LLaDA-style generation loop so the KV-cache quantizer can be
exercised inside a realistic control flow:

  1. The response window starts fully masked.
  2. Decoding proceeds block-by-block, left to right (semi-autoregressive).
  3. Within a block we run several denoising steps; each step runs a forward
     pass, predicts the masked tokens, commits the highest-confidence ones, and
     re-masks the rest (low-confidence remasking).
  4. Already-committed blocks act as a growing, *bidirectional* prefix whose
     K/V are cached — and, crucially, re-encoded every step, which is what makes
     dLLM cache statistics drift.

Quality is meaningless here (random weights); the value is a faithful schedule
of forward passes and KV writes for profiling and quantization experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .kv_cache import QuantizedKVCache
from .mock_dllm import MASK_ID, MockDiffusionLM


@dataclass
class DenoiseConfig:
    gen_len: int = 32          # response length to generate
    block_size: int = 8        # semi-autoregressive block width
    steps_per_block: int = 4   # denoising steps within each block
    temperature: float = 0.0   # 0 => greedy argmax
    seed: int = 0


@dataclass
class DenoiseTrace:
    """Bookkeeping returned by :func:`generate` for metrics/plots."""

    nfe: int = 0                       # number of model forward passes
    committed_tokens: int = 0
    cache_footprint_bits: float = 0.0
    per_step_kv_relative_drift: list = None  # filled by the profiler hook

    def __post_init__(self):
        if self.per_step_kv_relative_drift is None:
            self.per_step_kv_relative_drift = []

    @property
    def tokens_per_nfe(self) -> float:
        return self.committed_tokens / max(self.nfe, 1)


def _confidence(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Greedy prediction and its softmax confidence per position."""
    x = logits - logits.max(-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(-1, keepdims=True)
    pred = p.argmax(-1)
    conf = p.max(-1)
    return pred.astype(np.int64), conf


def generate(
    model: MockDiffusionLM,
    prompt: np.ndarray,
    cfg: DenoiseConfig,
    cache: Optional[QuantizedKVCache] = None,
) -> tuple[np.ndarray, DenoiseTrace]:
    """Run block denoising. ``prompt`` is (T_p,) int ids. Returns (tokens, trace).

    If ``cache`` is provided, every forward pass writes the prefix K/V through
    the (quantizing) cache, so the run reflects the chosen precision policy.
    """
    rng = np.random.default_rng(cfg.seed)
    prompt = np.asarray(prompt, dtype=np.int64).reshape(-1)
    T_p = prompt.shape[0]
    # Full sequence = prompt + masked response window.
    seq = np.concatenate([prompt, np.full(cfg.gen_len, MASK_ID, dtype=np.int64)])
    trace = DenoiseTrace()

    n_blocks = (cfg.gen_len + cfg.block_size - 1) // cfg.block_size
    for b in range(n_blocks):
        lo = T_p + b * cfg.block_size
        hi = min(lo + cfg.block_size, T_p + cfg.gen_len)
        block_idx = np.arange(lo, hi)
        for step in range(cfg.steps_per_block):
            logits, kv = model.forward(seq[None, :], capture=True)
            trace.nfe += 1
            if cache is not None:
                for li, layer_kv in enumerate(kv):
                    cache.update(li, layer_kv["k"], layer_kv["v"])
            pred, conf = _confidence(logits[0])
            # Only consider still-masked positions in the current block.
            masked = block_idx[seq[block_idx] == MASK_ID]
            if masked.size == 0:
                break
            # Commit a schedule-determined fraction of the most confident ones.
            keep = max(1, int(np.ceil(masked.size *
                                      (step + 1) / cfg.steps_per_block)))
            order = masked[np.argsort(-conf[masked])]
            commit = order[:keep]
            seq[commit] = pred[commit]
            trace.committed_tokens += int(commit.size)
        # Force-fill any residual masks in the block at block end.
        residual = block_idx[seq[block_idx] == MASK_ID]
        if residual.size:
            logits = model.forward(seq[None, :])
            pred, _ = _confidence(logits[0])
            seq[residual] = pred[residual]
            trace.committed_tokens += int(residual.size)

    if cache is not None:
        trace.cache_footprint_bits = cache.footprint_bits()
    return seq[T_p:], trace
