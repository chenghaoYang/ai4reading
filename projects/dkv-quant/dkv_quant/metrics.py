"""Quality, efficiency, and quantization-error metrics.

These are deliberately framework-free (numpy only) so they run in the no-GPU
prototyping phase and are reused unchanged when real LLaDA/Dream outputs arrive
on a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Quantization-error metrics (used by the CPU fake-quant sweep)
# --------------------------------------------------------------------------- #
def mse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.mean((a - b) ** 2))


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """||a-b|| / ||a|| — the natural "how much did quant perturb this" number."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    denom = np.linalg.norm(a.ravel()) + 1e-12
    return float(np.linalg.norm((a - b).ravel()) / denom)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def snr_db(a: np.ndarray, b: np.ndarray) -> float:
    """Signal-to-quantization-noise ratio in dB (higher is better)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    sig = np.mean(a**2)
    noise = np.mean((a - b) ** 2) + 1e-12
    return float(10.0 * np.log10(sig / noise))


# --------------------------------------------------------------------------- #
# Memory accounting
# --------------------------------------------------------------------------- #
def kv_cache_bits(n_tokens: int, n_layers: int, n_heads: int, head_dim: int,
                  bits: float, group_size: int = 0) -> float:
    """Total bits to store a KV cache, including scale/zero-point overhead.

    With group quantization each group of ``group_size`` elements carries an
    fp16 scale + zero-point (32 bits total) of overhead; group_size=0 ignores
    overhead (use for the fp16 baseline).
    """
    n_elems = 2 * n_tokens * n_layers * n_heads * head_dim  # K and V
    payload = n_elems * bits
    if group_size and group_size > 0:
        overhead = (n_elems / group_size) * 32.0
        return payload + overhead
    return payload


def compression_ratio(bits: float, group_size: int = 0,
                      baseline_bits: float = 16.0) -> float:
    """Effective compression vs. an fp16 cache (accounts for group overhead)."""
    per_elem = bits + (32.0 / group_size if group_size and group_size > 0 else 0.0)
    return baseline_bits / per_elem


# --------------------------------------------------------------------------- #
# Efficiency metrics for the (later, GPU) generation runs
# --------------------------------------------------------------------------- #
@dataclass
class GenerationStats:
    """One generation run's raw counters."""

    n_prompts: int
    total_output_tokens: int
    total_nfe: int          # number of function evaluations (denoising forwards)
    wall_seconds: float
    peak_mem_bytes: int = 0

    @property
    def throughput_tok_s(self) -> float:
        return self.total_output_tokens / max(self.wall_seconds, 1e-9)

    @property
    def tokens_per_nfe(self) -> float:
        """Parallelism: how many tokens each model forward commits on average."""
        return self.total_output_tokens / max(self.total_nfe, 1)

    @property
    def peak_mem_gb(self) -> float:
        return self.peak_mem_bytes / 1024**3


def accuracy_under_parallelism(accuracies: Sequence[float],
                               tokens_per_nfe: Sequence[float]) -> float:
    """AUP — area under the accuracy-vs-parallelism curve (d3LLM, 2601.07568).

    A single scalar that rewards methods staying accurate as they decode more
    tokens per forward pass.  ``tokens_per_nfe`` is the x-axis (parallelism),
    ``accuracies`` the y-axis; both sorted by x internally.
    """
    x = np.asarray(tokens_per_nfe, np.float64)
    y = np.asarray(accuracies, np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x.size < 2:
        return float(y.mean()) if y.size else 0.0
    area = np.trapezoid(y, x)
    return float(area / (x[-1] - x[0] + 1e-12))
