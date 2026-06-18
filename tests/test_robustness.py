"""Tests for the input-degradation transforms."""
import numpy as np

from achilles.ml.robustness import degrade


def _signal(C=3, T=101):
    t = np.linspace(0, 2 * np.pi, T)
    return np.stack([np.sin(t), np.cos(t), np.sin(2 * t)]).astype(float)


def test_degrade_preserves_shape():
    x = _signal()
    ch_std = x.std(axis=1)
    rng = np.random.default_rng(0)
    for kind, lvl in (("noise", 0.3), ("downsample", 4), ("quantize", 4)):
        assert degrade(x, kind, lvl, ch_std, rng).shape == x.shape


def test_noise_adds_variance_and_zero_is_identity():
    x = _signal()
    ch_std = x.std(axis=1)
    rng = np.random.default_rng(0)
    np.testing.assert_allclose(degrade(x, "noise", 0.0, ch_std, rng), x)
    noisy = degrade(x, "noise", 0.5, ch_std, rng)
    assert np.mean((noisy - x) ** 2) > 0


def test_quantize_reduces_distinct_levels():
    x = _signal()
    q = degrade(x, "quantize", 2, x.std(axis=1), np.random.default_rng(0))
    # 2-bit quantisation => at most ~4 distinct levels per channel
    assert len(np.unique(np.round(q[0], 6))) <= 5


def test_downsample_loses_information_but_keeps_knots():
    x = _signal()
    T = x.shape[1]
    d = degrade(x, "downsample", 8, x.std(axis=1), np.random.default_rng(0))
    assert np.mean((d - x) ** 2) > 0                      # information is lost
    kept = np.arange(0, T, 8)
    np.testing.assert_allclose(d[:, kept], x[:, kept], atol=1e-9)  # kept samples exact
    assert degrade(x, "downsample", 1, x.std(axis=1), np.random.default_rng(0)).shape == x.shape
