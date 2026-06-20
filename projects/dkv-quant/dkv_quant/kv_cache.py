"""A KV cache that supports approximate reuse *and* low-bit quantized storage.

The research thesis is that these two axes are orthogonal and stack
multiplicatively:

  * **reuse** (Fast-dLLM / dLLM-Cache / DPad): decide *which* positions to keep
    versus recompute each denoising step — reduces how *often* we store;
  * **quantization** (this project): decide *how many bits* per retained entry —
    reduces the cost of each thing we store.

Existing dLLM cache work does the first and keeps the survivors in fp16.  Here a
single store does both, and tracks the resulting memory footprint in bits so the
compression claim is measurable on CPU before any GPU run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .metrics import kv_cache_bits
from .quantizers import QuantConfig


@dataclass
class LayerCache:
    k: Optional[np.ndarray] = None  # (B, H, T, hd) dequantized-for-compute
    v: Optional[np.ndarray] = None


@dataclass
class QuantizedKVCache:
    """Stores per-layer K/V with an attached quantizer and reuse policy.

    The cache holds the *dequantized* tensors for compute (since we simulate
    quantization in fp32), but every write passes through the quantizer so the
    values reflect the precision loss, and the bit budget is accounted as if the
    integer codes were stored.
    """

    quant_cfg: QuantConfig
    quantizer: object  # KVQuantizer | StaticARQuantizer | StepAwareKVQuantizer
    n_layers: int
    layers: list = field(default_factory=list)
    _writes: int = 0
    _quant_elems: int = 0  # elements stored at quant_cfg.bits

    def __post_init__(self):
        self.layers = [LayerCache() for _ in range(self.n_layers)]
        # Per-layer quantizers so each layer keeps its own running stats.
        self._quantizers = [self._clone_quantizer() for _ in range(self.n_layers)]

    def _clone_quantizer(self):
        import copy
        q = copy.deepcopy(self.quantizer)
        if hasattr(q, "reset"):
            q.reset()
        return q

    def reset(self) -> None:
        self.layers = [LayerCache() for _ in range(self.n_layers)]
        self._quantizers = [self._clone_quantizer() for _ in range(self.n_layers)]
        self._writes = 0
        self._quant_elems = 0

    def update(self, layer: int, k: np.ndarray, v: np.ndarray) -> None:
        """Write (quantize) the K/V for ``layer`` at the current denoising step."""
        q = self._quantizers[layer]
        kq = q(k) if not isinstance(q, tuple) else k
        # Use a second, independent quantizer state for V by stacking — simplest
        # correct approach is to quantize K and V with the same policy instance
        # but separate running stats. We keep one quantizer per (layer, tensor).
        self.layers[layer].k = kq
        self.layers[layer].v = self._quantizers_v(layer)(v)
        self._writes += 1
        self._quant_elems += k.size + v.size

    # Lazily create per-layer V quantizers (mirrors the K ones).
    def _quantizers_v(self, layer: int):
        if not hasattr(self, "_qv"):
            self._qv = [self._clone_quantizer() for _ in range(self.n_layers)]
        return self._qv[layer]

    def get(self, layer: int) -> LayerCache:
        return self.layers[layer]

    # ----------------------------------------------------------------- #
    def footprint_bits(self, group_size: int = 0) -> float:
        """Total bits for everything written so far at the configured bit-width."""
        gs = group_size or (self.quant_cfg.group_size
                            if self.quant_cfg.granularity == "per_group" else 0)
        return kv_cache_bits(
            n_tokens=self._quant_elems_to_tokens(),
            n_layers=1, n_heads=1, head_dim=1,
            bits=self.quant_cfg.bits, group_size=gs,
        )

    def _quant_elems_to_tokens(self) -> int:
        # footprint helper treats every stored element as one "token slot".
        return self._quant_elems

    @property
    def n_writes(self) -> int:
        return self._writes
