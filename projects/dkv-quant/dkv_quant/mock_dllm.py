"""A tiny masked-diffusion language model in pure numpy.

This is NOT a model you train or benchmark for quality — it is a faithful
*mechanism mock* so the whole pipeline (bidirectional attention, KV extraction,
block / semi-autoregressive denoising, KV-cache reuse, and fake quantization)
can be exercised end-to-end on a CPU with random weights and zero downloads.

It reproduces the structural facts that matter for KV-cache quantization in a
diffusion LLM:

  * attention is **bidirectional** (no causal mask) — every position attends to
    every other position, so cached K/V for a position are *not* immutable;
  * generation is the **reverse denoising** process: start from an all-[MASK]
    response and, over several steps, predict masked tokens and re-mask the
    low-confidence ones;
  * decoding is **semi-autoregressive**: the response is split into blocks
    decoded left-to-right, tokens within a block denoised in parallel.

Because the weights are fixed random projections, the per-channel K/V statistics
genuinely *drift* as the (re-)masked input changes from step to step — which is
exactly the phenomenon `StepAwareKVQuantizer` is designed to track and the
`profiler` is designed to measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

MASK_ID = -1  # sentinel token id for [MASK]


@dataclass
class MockConfig:
    vocab_size: int = 256
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    max_seq: int = 128
    seed: int = 0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _layernorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


class MockDiffusionLM:
    """Fixed-random-weight bidirectional transformer with a denoising head."""

    def __init__(self, cfg: MockConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        d = cfg.d_model
        scale = 1.0 / np.sqrt(d)
        # Embeddings: ordinary tokens + a dedicated [MASK] embedding.
        self.tok_emb = rng.standard_normal((cfg.vocab_size, d)) * scale
        self.mask_emb = rng.standard_normal(d) * scale
        self.pos_emb = rng.standard_normal((cfg.max_seq, d)) * scale
        # Per-layer attention + MLP projections.
        self.Wq, self.Wk, self.Wv, self.Wo = [], [], [], []
        self.W1, self.W2 = [], []
        for _ in range(cfg.n_layers):
            self.Wq.append(rng.standard_normal((d, d)) * scale)
            self.Wk.append(rng.standard_normal((d, d)) * scale)
            self.Wv.append(rng.standard_normal((d, d)) * scale)
            self.Wo.append(rng.standard_normal((d, d)) * scale)
            self.W1.append(rng.standard_normal((d, 4 * d)) * scale)
            self.W2.append(rng.standard_normal((4 * d, d)) * scale)
        self.head = rng.standard_normal((d, cfg.vocab_size)) * scale

    # --------------------------------------------------------------------- #
    def embed(self, tokens: np.ndarray) -> np.ndarray:
        """Embed a (B, T) int array; MASK_ID maps to the mask embedding."""
        B, T = tokens.shape
        d = self.cfg.d_model
        out = np.empty((B, T, d), dtype=np.float64)
        mask = tokens == MASK_ID
        safe = np.where(mask, 0, tokens)
        out[:] = self.tok_emb[safe]
        out[mask] = self.mask_emb
        out = out + self.pos_emb[:T]
        return out

    def forward(self, tokens: np.ndarray, capture: bool = False):
        """Run the bidirectional stack.

        Returns logits (B, T, vocab).  If ``capture`` is True, also returns a
        list (per layer) of dicts with the K and V tensors (B, H, T, head_dim)
        — this is what the profiler and KV-cache quantizer consume.
        """
        cfg = self.cfg
        H, hd = cfg.n_heads, cfg.head_dim
        x = self.embed(tokens)
        B, T, _ = x.shape
        captured = []
        for li in range(cfg.n_layers):
            h = _layernorm(x)
            q = (h @ self.Wq[li]).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            k = (h @ self.Wk[li]).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            v = (h @ self.Wv[li]).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            # Bidirectional attention — NO causal mask.
            att = _softmax(q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd))
            ctx = att @ v  # (B, H, T, hd)
            ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, cfg.d_model)
            x = x + ctx @ self.Wo[li]
            h2 = _layernorm(x)
            x = x + np.maximum(h2 @ self.W1[li], 0) @ self.W2[li]
            if capture:
                captured.append({"k": k.copy(), "v": v.copy()})
        logits = _layernorm(x) @ self.head
        if capture:
            return logits, captured
        return logits

    def forward_kv(self, tokens: np.ndarray):
        """Convenience: return only the captured K/V (list per layer)."""
        _, kv = self.forward(tokens, capture=True)
        return kv
