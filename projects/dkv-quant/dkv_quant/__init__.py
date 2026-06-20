"""dKV-Quant: Step-Aware Low-Bit KV-Cache Quantization for Diffusion LLMs.

A research scaffold. The numpy core (quantizers, mock dLLM, denoising loop,
cache-drift profiler, metrics) runs entirely on CPU for the no-GPU preparation
phase; `scripts/run_eval.py` wires the same abstractions to real LLaDA/Dream
checkpoints for the GPU phase.
"""

__version__ = "0.1.0"

from .quantizers import (  # noqa: F401
    QuantConfig,
    KVQuantizer,
    StaticARQuantizer,
    StepAwareKVQuantizer,
    build_quantizer,
    quantize_dequantize,
    hadamard_transform,
)
from .mock_dllm import MockConfig, MockDiffusionLM, MASK_ID  # noqa: F401
from .denoise import DenoiseConfig, generate  # noqa: F401
from .kv_cache import QuantizedKVCache  # noqa: F401
from .profiler import profile_denoising, DriftReport  # noqa: F401
from .baselines import default_methods, Method  # noqa: F401
from . import metrics  # noqa: F401
