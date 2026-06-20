import numpy as np
import pytest

from dkv_quant.quantizers import (
    QuantConfig,
    StaticARQuantizer,
    StepAwareKVQuantizer,
    KVQuantizer,
    build_quantizer,
    quantize_dequantize,
    hadamard_transform,
    inverse_hadamard_transform,
    compute_scale_zero_point,
)
from dkv_quant.metrics import relative_l2


def test_roundtrip_error_shrinks_with_bits():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 4, 16, 32))
    errs = []
    for b in (2, 3, 4, 8):
        cfg = QuantConfig(bits=b, granularity="per_channel")
        errs.append(relative_l2(x, quantize_dequantize(x, cfg)))
    # More bits => strictly less error.
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1)), errs


def test_8bit_is_near_lossless():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 2, 8, 16))
    cfg = QuantConfig(bits=8, granularity="per_group", group_size=8)
    assert relative_l2(x, quantize_dequantize(x, cfg)) < 0.02


@pytest.mark.parametrize("gran", ["per_tensor", "per_channel", "per_token", "per_group"])
def test_granularities_run_and_shapes(gran):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 3, 10, 16))
    cfg = QuantConfig(bits=4, granularity=gran, group_size=4)
    out = quantize_dequantize(x, cfg)
    assert out.shape == x.shape
    assert np.isfinite(out).all()


def test_per_channel_finer_than_per_tensor_with_outliers():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 1, 8, 16))
    x[..., 0] *= 50.0  # one outlier channel
    e_tensor = relative_l2(x, quantize_dequantize(x, QuantConfig(4, "per_tensor")))
    e_chan = relative_l2(x, quantize_dequantize(x, QuantConfig(4, "per_channel")))
    assert e_chan < e_tensor


def test_hadamard_is_invertible():
    rng = np.random.default_rng(4)
    for c in (16, 24, 32):  # includes a non-power-of-two
        x = rng.standard_normal((2, 3, 5, c))
        back = inverse_hadamard_transform(hadamard_transform(x))
        assert np.allclose(back[..., :c], x, atol=1e-9)


def test_static_quantizer_freezes_scale():
    cfg = QuantConfig(bits=4, granularity="per_channel")
    q = StaticARQuantizer(cfg)
    a = np.ones((1, 1, 4, 8))
    q(a)  # calibrate on small-range tensor
    s_before = q._scale.copy()
    big = a * 100.0
    q(big)  # should NOT recalibrate
    assert np.allclose(q._scale, s_before)


def test_step_aware_tracks_drift_better_than_static():
    """On a drifting tensor stream, step-aware beats frozen AR calibration."""
    rng = np.random.default_rng(5)
    cfg = QuantConfig(bits=4, granularity="per_channel")
    static = StaticARQuantizer(cfg)
    aware = StepAwareKVQuantizer(cfg, momentum=0.5)
    base = rng.standard_normal((1, 2, 8, 16))
    static_err, aware_err = [], []
    for step in range(8):
        # The per-channel scale grows over steps (simulated drift).
        x = base * (1.0 + 0.6 * step)
        static_err.append(relative_l2(x, static(x)))
        aware_err.append(relative_l2(x, aware(x)))
    # Averaged over the drift, step-aware should be at least as good and
    # strictly better once drift accumulates.
    assert np.mean(aware_err) < np.mean(static_err)
    assert aware_err[-1] < static_err[-1]


def test_step_aware_momentum_extremes():
    cfg = QuantConfig(bits=4, granularity="per_channel")
    rng = np.random.default_rng(6)
    xs = [rng.standard_normal((1, 1, 4, 8)) * (1 + s) for s in range(5)]
    fresh = KVQuantizer(cfg)
    m0 = StepAwareKVQuantizer(cfg, momentum=0.0)
    # momentum=0 == always-fresh recalibration
    for x in xs:
        assert np.allclose(fresh(x), m0(x))


def test_build_quantizer_dispatch():
    cfg = QuantConfig(bits=4)
    assert isinstance(build_quantizer("ar_static", cfg), StaticARQuantizer)
    assert isinstance(build_quantizer("step_aware", cfg), StepAwareKVQuantizer)
    assert isinstance(build_quantizer("fresh", cfg), KVQuantizer)
    with pytest.raises(ValueError):
        build_quantizer("nope", cfg)


def test_scale_shapes_broadcast():
    x = np.random.default_rng(7).standard_normal((2, 3, 9, 16))
    for gran in ("per_channel", "per_token"):
        s, z = compute_scale_zero_point(x, QuantConfig(4, gran))
        assert (x / s).shape == x.shape  # broadcastable
