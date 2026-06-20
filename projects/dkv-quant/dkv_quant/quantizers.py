"""Fake (simulated) quantization for KV-cache entries.

Everything here is *quantize-dequantize* in fp32 (numpy) so it runs on CPU with
no GPU kernels.  The point of simulated quantization is to isolate the
*algorithmic* question — "how much accuracy do we lose at B bits, and which
calibration policy is correct for a diffusion LLM?" — from the systems question
of building fast low-bit kernels (which is the later, GPU-side part of the
project).

The novel contribution prototyped here is the *step-aware* calibration policy.
Autoregressive (AR) KV-cache quantizers (KIVI, RotateKV, MixKVQ, ...) assume a
write-once, monotonically-growing causal cache: a key/value is computed once and
never changes, so a single per-channel scale calibrated at write time is valid
forever.  Diffusion LLMs violate this: under bidirectional attention every
denoising step re-encodes the *same* token positions, so the per-channel
min/max (and the outlier channels) drift from step to step.  A scale calibrated
at step 0 is stale by step N.  `StepAwareKVQuantizer` re-derives scales from the
"warm" recomputation that the dLLM already performs each step, giving a free,
data-less calibration point (the software analogue of the BAOS hardware trick).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

Granularity = Literal["per_tensor", "per_channel", "per_token", "per_group"]


@dataclass
class QuantConfig:
    """Configuration for a fake-quantizer.

    Attributes:
        bits: bit-width of the integer grid (2, 3, 4, 8 typical).
        granularity: how scales/zero-points are shared across the tensor.
            ``per_channel`` shares one scale per channel (the head_dim axis);
            ``per_token`` shares one per token (the sequence axis);
            ``per_group`` uses blocks of ``group_size`` along the channel axis.
        symmetric: symmetric (zero-point fixed at 0) vs. asymmetric (affine).
        channel_axis: which axis is the "channel" (default last = head_dim).
        token_axis: which axis is the "token"/sequence axis.
        group_size: group width when ``granularity == 'per_group'``.
        hadamard: apply a fast Hadamard rotation before quantizing to spread
            channel outliers (RotateKV / KVLinC style). Rotation is exact and
            invertible, so it is undone on dequant.
    """

    bits: int = 4
    granularity: Granularity = "per_channel"
    symmetric: bool = False
    channel_axis: int = -1
    token_axis: int = -2
    group_size: int = 128
    hadamard: bool = False

    def __post_init__(self) -> None:
        if self.bits < 1 or self.bits > 16:
            raise ValueError(f"bits must be in [1, 16], got {self.bits}")
        if self.granularity == "per_group" and self.group_size <= 0:
            raise ValueError("group_size must be positive for per_group")


# --------------------------------------------------------------------------- #
# Low-level uniform quant-dequant
# --------------------------------------------------------------------------- #
def _qmin_qmax(bits: int, symmetric: bool) -> tuple[int, int]:
    if symmetric:
        q = 2 ** (bits - 1) - 1
        return -q, q
    return 0, 2**bits - 1


def _reduce_axes(shape: int, keep_axis: int) -> tuple[int, ...]:
    """All axes except ``keep_axis`` (normalised to positive)."""
    keep_axis = keep_axis % shape
    return tuple(a for a in range(shape) if a != keep_axis)


def quantize_dequantize(
    x: np.ndarray,
    cfg: QuantConfig,
    scale: Optional[np.ndarray] = None,
    zero_point: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Simulate quantization: x -> int grid -> back to fp32.

    If ``scale``/``zero_point`` are provided they are *reused* (this is how the
    step-aware quantizer injects a calibration computed at a different step).
    Otherwise they are derived from ``x`` itself (standard min/max calibration).
    """
    x = np.asarray(x, dtype=np.float64)
    if cfg.hadamard:
        x = hadamard_transform(x, axis=cfg.channel_axis)

    qmin, qmax = _qmin_qmax(cfg.bits, cfg.symmetric)

    if scale is None or zero_point is None:
        scale, zero_point = compute_scale_zero_point(x, cfg)

    q = np.round(x / scale + zero_point)
    q = np.clip(q, qmin, qmax)
    deq = (q - zero_point) * scale

    if cfg.hadamard:
        deq = inverse_hadamard_transform(deq, axis=cfg.channel_axis)
    return deq.astype(np.float32)


def compute_scale_zero_point(
    x: np.ndarray, cfg: QuantConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Derive (scale, zero_point) with the configured granularity.

    Returns arrays broadcastable against ``x`` so they can be cached and reused
    across denoising steps.
    """
    x = np.asarray(x, dtype=np.float64)
    qmin, qmax = _qmin_qmax(cfg.bits, cfg.symmetric)
    eps = 1e-8

    if cfg.granularity == "per_tensor":
        axes: tuple[int, ...] = tuple(range(x.ndim))
    elif cfg.granularity == "per_channel":
        axes = _reduce_axes(x.ndim, cfg.channel_axis)
    elif cfg.granularity == "per_token":
        axes = _reduce_axes(x.ndim, cfg.token_axis)
    elif cfg.granularity == "per_group":
        return _group_scale_zero_point(x, cfg)
    else:  # pragma: no cover - guarded by QuantConfig
        raise ValueError(cfg.granularity)

    if cfg.symmetric:
        amax = np.max(np.abs(x), axis=axes, keepdims=True)
        scale = np.maximum(amax / qmax, eps)
        zero_point = np.zeros_like(scale)
    else:
        xmin = np.min(x, axis=axes, keepdims=True)
        xmax = np.max(x, axis=axes, keepdims=True)
        scale = np.maximum((xmax - xmin) / (qmax - qmin), eps)
        zero_point = np.round(qmin - xmin / scale)
    return scale, zero_point


def _group_scale_zero_point(
    x: np.ndarray, cfg: QuantConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Per-group scales along the channel axis (groups of ``group_size``)."""
    ax = cfg.channel_axis % x.ndim
    C = x.shape[ax]
    g = cfg.group_size
    n_groups = (C + g - 1) // g
    qmin, qmax = _qmin_qmax(cfg.bits, cfg.symmetric)
    eps = 1e-8

    scale = np.ones_like(x)
    zero_point = np.zeros_like(x)
    xm = np.moveaxis(x, ax, -1)
    sm = np.moveaxis(scale, ax, -1)
    zm = np.moveaxis(zero_point, ax, -1)
    for gi in range(n_groups):
        lo, hi = gi * g, min((gi + 1) * g, C)
        chunk = xm[..., lo:hi]
        if cfg.symmetric:
            amax = np.max(np.abs(chunk), axis=-1, keepdims=True)
            s = np.maximum(amax / qmax, eps)
            z = np.zeros_like(s)
        else:
            cmin = np.min(chunk, axis=-1, keepdims=True)
            cmax = np.max(chunk, axis=-1, keepdims=True)
            s = np.maximum((cmax - cmin) / (qmax - qmin), eps)
            z = np.round(qmin - cmin / s)
        sm[..., lo:hi] = s
        zm[..., lo:hi] = z
    return np.moveaxis(sm, -1, ax), np.moveaxis(zm, -1, ax)


# --------------------------------------------------------------------------- #
# Hadamard rotation (outlier smoothing)
# --------------------------------------------------------------------------- #
def _hadamard_matrix(n: int) -> np.ndarray:
    """Normalised Hadamard matrix for n a power of two."""
    if n & (n - 1) != 0:
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    H = np.ones((1, 1), dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def hadamard_transform(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Apply an orthonormal Hadamard rotation along ``axis``.

    If the axis length is not a power of two we zero-pad to the next one and the
    inverse crops back, so the round-trip is exact for the original entries.
    """
    ax = axis % x.ndim
    n = x.shape[ax]
    n2 = 1 << (n - 1).bit_length()
    H = _hadamard_matrix(n2)
    xm = np.moveaxis(x, ax, -1)
    if n2 != n:
        pad = [(0, 0)] * xm.ndim
        pad[-1] = (0, n2 - n)
        xm = np.pad(xm, pad)
    out = xm @ H
    return np.moveaxis(out, -1, ax)


def inverse_hadamard_transform(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Inverse of :func:`hadamard_transform` (Hadamard is its own inverse)."""
    # Forward already divides by sqrt(n2); applying H again multiplies back.
    return hadamard_transform(x, axis=axis)


# --------------------------------------------------------------------------- #
# Quantizers used by the KV cache
# --------------------------------------------------------------------------- #
class KVQuantizer:
    """Plain (stateless) per-step KV quantizer.

    Recalibrates scales from whatever tensor it is handed.  This is the
    *correct* but expensive reference: it always sees fresh statistics.
    """

    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return quantize_dequantize(x, self.cfg)


class StaticARQuantizer:
    """Calibrate ONCE (at first write) and freeze — the AR assumption.

    This deliberately models how an off-the-shelf autoregressive KV-cache
    quantizer behaves when dropped onto a diffusion LLM: it locks the scales at
    step 0 and reuses them forever, ignoring the step-to-step drift.  We expect
    it to degrade; quantifying that degradation is half the paper's motivation.
    """

    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg
        self._scale: Optional[np.ndarray] = None
        self._zp: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._scale = None
        self._zp = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        xr = hadamard_transform(x, self.cfg.channel_axis) if self.cfg.hadamard else x
        if self._scale is None:
            self._scale, self._zp = compute_scale_zero_point(xr, self.cfg)
        return quantize_dequantize(x, self.cfg, self._scale, self._zp)


@dataclass
class StepAwareKVQuantizer:
    """Step-aware KV-cache quantizer (the proposed method).

    Key idea: a diffusion LLM recomputes the cached positions every denoising
    step anyway (the "warm step").  We treat that recomputation as a free,
    data-less calibration probe and update the per-channel scales with an EMA so
    they track the drifting outlier statistics instead of freezing at step 0.

    ``momentum`` blends the previous scale with the freshly observed one:
    momentum=0 -> fully fresh every step (== KVQuantizer), momentum=1 -> frozen
    (== StaticARQuantizer).  Intermediate values track drift while damping noise.
    """

    cfg: QuantConfig
    momentum: float = 0.5
    _scale: Optional[np.ndarray] = field(default=None, repr=False)
    _zp: Optional[np.ndarray] = field(default=None, repr=False)

    def reset(self) -> None:
        self._scale = None
        self._zp = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        xr = hadamard_transform(x, self.cfg.channel_axis) if self.cfg.hadamard else x
        s_new, z_new = compute_scale_zero_point(xr, self.cfg)
        if self._scale is None:
            self._scale, self._zp = s_new, z_new
        else:
            m = self.momentum
            self._scale = m * self._scale + (1 - m) * s_new
            self._zp = np.round(m * self._zp + (1 - m) * z_new)
        return quantize_dequantize(x, self.cfg, self._scale, self._zp)


def build_quantizer(method: str, cfg: QuantConfig, **kwargs):
    """Factory mapping a method name to a quantizer instance."""
    method = method.lower()
    if method in ("fresh", "ideal", "per_step"):
        return KVQuantizer(cfg)
    if method in ("ar_static", "static", "ar"):
        return StaticARQuantizer(cfg)
    if method in ("step_aware", "dkv", "ours"):
        return StepAwareKVQuantizer(cfg, momentum=kwargs.get("momentum", 0.5))
    raise ValueError(f"unknown quantizer method: {method!r}")
