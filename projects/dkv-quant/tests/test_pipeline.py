import numpy as np
import pytest

from dkv_quant.mock_dllm import MockConfig, MockDiffusionLM, MASK_ID
from dkv_quant.denoise import DenoiseConfig, generate
from dkv_quant.kv_cache import QuantizedKVCache
from dkv_quant.quantizers import QuantConfig, build_quantizer
from dkv_quant.profiler import profile_denoising
from dkv_quant.baselines import default_methods
from dkv_quant import metrics


def _model():
    return MockDiffusionLM(MockConfig(vocab_size=64, d_model=32, n_heads=4,
                                      n_layers=2, max_seq=64, seed=0))


def test_forward_shapes_and_kv_capture():
    m = _model()
    toks = np.array([[1, 2, 3, MASK_ID, MASK_ID]])
    logits, kv = m.forward(toks, capture=True)
    assert logits.shape == (1, 5, 64)
    assert len(kv) == 2
    assert kv[0]["k"].shape == (1, 4, 5, 8)  # (B, H, T, head_dim)


def test_generation_fills_all_masks():
    m = _model()
    cfg = DenoiseConfig(gen_len=16, block_size=4, steps_per_block=3, seed=1)
    out, trace = generate(m, prompt=np.array([1, 2, 3]), cfg=cfg)
    assert out.shape == (16,)
    assert not np.any(out == MASK_ID)        # everything committed
    assert trace.nfe > 0
    assert trace.committed_tokens >= 16
    assert 0 < trace.tokens_per_nfe


def test_generation_with_quantized_cache_runs():
    m = _model()
    qcfg = QuantConfig(bits=4, granularity="per_group", group_size=8)
    cache = QuantizedKVCache(
        quant_cfg=qcfg,
        quantizer=build_quantizer("step_aware", qcfg),
        n_layers=m.cfg.n_layers,
    )
    cfg = DenoiseConfig(gen_len=12, block_size=4, steps_per_block=2, seed=2)
    out, trace = generate(m, prompt=np.array([5, 6]), cfg=cfg, cache=cache)
    assert out.shape == (12,)
    assert cache.n_writes > 0
    assert trace.cache_footprint_bits > 0


def test_profiler_detects_drift():
    """A bidirectional dLLM must show non-trivial step-to-step KV drift."""
    m = _model()
    cfg = DenoiseConfig(gen_len=24, block_size=6, steps_per_block=4, seed=3)
    report = profile_denoising(m, prompt=np.array([1, 2, 3, 4]), cfg=cfg, layer=0)
    s = report.summary()
    assert s["n_steps"] > 3
    # Drift must be measurable and positive — the core empirical premise.
    assert s["k_tensor_drift_mean"] > 0
    assert s["k_scale_drift_mean"] > 0
    assert 0.0 <= s["outlier_churn_mean"] <= 1.0


def test_default_methods_registry():
    methods = default_methods(bits=4, group_size=64)
    names = {m.name for m in methods}
    assert {"fp16", "ar_static", "dkv_step_aware"} <= names
    ours = [m for m in methods if m.is_ours]
    assert len(ours) >= 1
    # Each non-fp16 method builds a usable quantizer.
    for meth in methods:
        q = meth.make_quantizer()
        if meth.name != "fp16":
            assert q is not None


def test_metrics_compression_and_aup():
    # 4-bit with group overhead compresses < 4x vs fp16.
    cr = metrics.compression_ratio(bits=4, group_size=128)
    assert 3.0 < cr < 4.0
    # AUP rewards staying accurate at higher parallelism.
    flat = metrics.accuracy_under_parallelism([0.8, 0.8, 0.8], [1, 2, 3])
    assert abs(flat - 0.8) < 1e-6
    decaying = metrics.accuracy_under_parallelism([0.9, 0.6, 0.3], [1, 2, 3])
    assert decaying < flat
