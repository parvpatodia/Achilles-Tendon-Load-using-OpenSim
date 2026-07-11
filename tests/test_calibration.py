"""Tests for the subject-specific calibration sweep."""
import numpy as np

from achilles.data.synthetic import SyntheticGaitSource
from achilles.ml.calibration import fit_affine, subject_calibration_sweep


def _result(method="affine"):
    trials = SyntheticGaitSource(n_subjects=12, seed=0).load_trials()  # 6 trials/subject
    return subject_calibration_sweep(trials, ks=(1, 2, 3), method=method,
                                     n_draws=2, k_fold=4, seed=0)


def test_structure_and_finite():
    r = _result()
    assert r.ks == [1, 2, 3]
    for k in r.ks:
        for d in (r.uncal_loaded_r2, r.cal_loaded_r2, r.uncal_worst_loaded_r2,
                  r.cal_worst_loaded_r2, r.uncal_peak_mape, r.cal_peak_mape):
            assert np.isfinite(d[k])
    assert "uncal->cal" in r.table()


def test_identity_is_a_noop():
    # With the identity correction, calibrated == uncalibrated on every metric,
    # which proves the uncal-vs-cal comparison is matched on identical eval steps.
    r = _result(method="identity")
    for k in r.ks:
        assert abs(r.cal_loaded_r2[k] - r.uncal_loaded_r2[k]) < 1e-9
        assert abs(r.cal_worst_loaded_r2[k] - r.uncal_worst_loaded_r2[k]) < 1e-9
        assert abs(r.cal_peak_mape[k] - r.uncal_peak_mape[k]) < 1e-9


def test_calibration_does_not_hurt_worst_case():
    # Calibration removes a systematic per-person bias; even on synthetic data
    # (little bias to remove) it must not badly hurt the worst athlete. A small
    # tolerance covers the tiny-K fit.
    r = _result(method="affine")
    k = r.ks[-1]
    assert r.cal_worst_loaded_r2[k] >= r.uncal_worst_loaded_r2[k] - 0.10


def test_affine_recovers_a_known_bias():
    # A pure scale+offset applied to a curve is exactly recoverable by the affine
    # fit, so a biased prediction calibrates back onto the truth.
    rng = np.random.default_rng(0)
    true = [np.clip(np.sin(np.linspace(0, np.pi, 101)) * 5 + rng.normal(0, 0.01, 101), 0, None)
            for _ in range(3)]
    pred = [1.3 * c + 0.4 for c in true]           # a known systematic bias
    a, b = fit_affine(true, pred)
    recovered = [a * p + b for p in pred]
    assert np.allclose(np.concatenate(recovered), np.concatenate(true), atol=0.05)
