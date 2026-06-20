"""Cache-drift profiler — the empirical motivation for step-aware quantization.

This module measures, for the mock dLLM (or any model exposing per-step K/V),
*how much the KV statistics move from one denoising step to the next*.  Those
measurements are the quantitative argument of the paper:

  * If per-channel min/max and outlier channels are essentially static across
    steps, an AR-style write-once calibration is fine and step-awareness buys
    nothing.
  * If they drift substantially — which bidirectional re-encoding causes — then
    a frozen scale is stale and step-aware recalibration should recover the gap.

All measurements are CPU-only and run on a random-weight model, so this whole
analysis is part of the no-GPU preparation.  The same hooks attach to a real
LLaDA/Dream forward later without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .denoise import DenoiseConfig, generate
from .metrics import relative_l2
from .mock_dllm import MASK_ID, MockDiffusionLM


@dataclass
class DriftReport:
    """Aggregated drift statistics across denoising steps."""

    n_steps: int = 0
    # Mean relative-L2 change of K and V tensors between consecutive steps.
    k_step_drift: list = field(default_factory=list)
    v_step_drift: list = field(default_factory=list)
    # Mean relative change of the *per-channel max* (the calibration target).
    k_scale_drift: list = field(default_factory=list)
    v_scale_drift: list = field(default_factory=list)
    # How often the set of top-outlier channels changes step to step (Jaccard).
    outlier_churn: list = field(default_factory=list)

    def summary(self) -> dict:
        def m(x):
            return float(np.mean(x)) if x else 0.0
        return {
            "n_steps": self.n_steps,
            "k_tensor_drift_mean": m(self.k_step_drift),
            "v_tensor_drift_mean": m(self.v_step_drift),
            "k_scale_drift_mean": m(self.k_scale_drift),
            "v_scale_drift_mean": m(self.v_scale_drift),
            "outlier_churn_mean": m(self.outlier_churn),
        }


def _per_channel_max(t: np.ndarray) -> np.ndarray:
    """Per-channel (last-axis) absolute max over (B,H,T)."""
    return np.max(np.abs(t), axis=tuple(range(t.ndim - 1)))


def _top_outlier_channels(t: np.ndarray, frac: float = 0.05) -> set:
    cmax = _per_channel_max(t)
    k = max(1, int(np.ceil(cmax.size * frac)))
    return set(np.argsort(-cmax)[:k].tolist())


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def profile_denoising(
    model: MockDiffusionLM,
    prompt: np.ndarray,
    cfg: DenoiseConfig,
    layer: int = 0,
    outlier_frac: float = 0.05,
) -> DriftReport:
    """Re-run the denoising schedule, capturing K/V each step, and measure drift.

    We replay the exact greedy commit schedule of :func:`generate` so the
    captured sequence of forward passes matches a real generation, then compare
    consecutive steps' K/V for the chosen layer.
    """
    from .denoise import _confidence  # local import to avoid cycle at top

    prompt = np.asarray(prompt, dtype=np.int64).reshape(-1)
    T_p = prompt.shape[0]
    seq = np.concatenate([prompt, np.full(cfg.gen_len, MASK_ID, dtype=np.int64)])
    report = DriftReport()

    prev_k = prev_v = None
    prev_kmax = prev_vmax = None
    prev_kset = prev_vset = None

    n_blocks = (cfg.gen_len + cfg.block_size - 1) // cfg.block_size
    for b in range(n_blocks):
        lo = T_p + b * cfg.block_size
        hi = min(lo + cfg.block_size, T_p + cfg.gen_len)
        block_idx = np.arange(lo, hi)
        for step in range(cfg.steps_per_block):
            logits, kv = model.forward(seq[None, :], capture=True)
            k, v = kv[layer]["k"], kv[layer]["v"]
            report.n_steps += 1

            if prev_k is not None:
                report.k_step_drift.append(relative_l2(prev_k, k))
                report.v_step_drift.append(relative_l2(prev_v, v))
                kmax, vmax = _per_channel_max(k), _per_channel_max(v)
                report.k_scale_drift.append(relative_l2(prev_kmax, kmax))
                report.v_scale_drift.append(relative_l2(prev_vmax, vmax))
                kset = _top_outlier_channels(k, outlier_frac)
                vset = _top_outlier_channels(v, outlier_frac)
                report.outlier_churn.append(
                    1.0 - 0.5 * (_jaccard(prev_kset, kset)
                                 + _jaccard(prev_vset, vset))
                )
            prev_k, prev_v = k, v
            prev_kmax, prev_vmax = _per_channel_max(k), _per_channel_max(v)
            prev_kset = _top_outlier_channels(k, outlier_frac)
            prev_vset = _top_outlier_channels(v, outlier_frac)

            pred, conf = _confidence(logits[0])
            masked = block_idx[seq[block_idx] == MASK_ID]
            if masked.size == 0:
                break
            keep = max(1, int(np.ceil(masked.size *
                                      (step + 1) / cfg.steps_per_block)))
            order = masked[np.argsort(-conf[masked])]
            seq[order[:keep]] = pred[order[:keep]]
        residual = block_idx[seq[block_idx] == MASK_ID]
        if residual.size:
            logits = model.forward(seq[None, :])
            pred, _ = _confidence(logits[0])
            seq[residual] = pred[residual]

    return report
