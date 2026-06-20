"""Method registry: the comparison points reported in the paper.

Each entry returns a callable that builds the per-run quantizer, plus metadata
(effective bits, whether it is the proposed method) so experiment scripts can
sweep them uniformly.  This keeps the GPU eval script and the CPU simulation in
lockstep on exactly what "baseline X" means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .quantizers import QuantConfig, build_quantizer


@dataclass
class Method:
    name: str
    description: str
    make_quantizer: Callable
    bits: float
    is_ours: bool = False


def default_methods(bits: int = 4, group_size: int = 128,
                     momentum: float = 0.5) -> list[Method]:
    """The standard comparison set for a KV-cache quantization table."""

    def cfg(**kw):
        base = dict(bits=bits, granularity="per_group", group_size=group_size,
                    symmetric=False)
        base.update(kw)
        return QuantConfig(**base)

    return [
        Method(
            name="fp16",
            description="No quantization — upper-bound reference (16-bit).",
            make_quantizer=lambda: None,
            bits=16.0,
        ),
        Method(
            name="ar_static",
            description="AR-style write-once calibration frozen at step 0. "
                        "Models dropping an off-the-shelf KV quantizer onto a "
                        "dLLM; expected to degrade.",
            make_quantizer=lambda: build_quantizer("ar_static", cfg()),
            bits=float(bits),
        ),
        Method(
            name="ar_static_hadamard",
            description="AR-static + Hadamard rotation (RotateKV/KVLinC-style "
                        "outlier smoothing), still frozen calibration.",
            make_quantizer=lambda: build_quantizer("ar_static", cfg(hadamard=True)),
            bits=float(bits),
        ),
        Method(
            name="fresh_per_step",
            description="Recalibrate fully every step — accurate but the most "
                        "expensive; an oracle-ish reference for the gap.",
            make_quantizer=lambda: build_quantizer("fresh", cfg()),
            bits=float(bits),
        ),
        Method(
            name="dkv_step_aware",
            description="PROPOSED: EMA step-aware recalibration using the dLLM's "
                        "warm-step recomputation as a free calibration probe.",
            make_quantizer=lambda: build_quantizer("step_aware", cfg(),
                                                   momentum=momentum),
            bits=float(bits),
            is_ours=True,
        ),
        Method(
            name="dkv_step_aware_hadamard",
            description="PROPOSED + Hadamard rotation: step-aware EMA on rotated "
                        "channels for the strongest outlier handling.",
            make_quantizer=lambda: build_quantizer("step_aware", cfg(hadamard=True),
                                                   momentum=momentum),
            bits=float(bits),
            is_ours=True,
        ),
    ]
