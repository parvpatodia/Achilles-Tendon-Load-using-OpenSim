"""Tests for the cross-fold conformal calibration logic (no training needed)."""
import numpy as np

from achilles.ml.uncertainty import (cross_fold_conformal_coverage,
                                      cross_fold_halfwidths)


def test_conformal_coverage_matches_nominal_on_iid_residuals():
    """On iid residuals, cross-fold conformal coverage should track the nominal
    level (the calibration guarantee), within sampling tolerance."""
    rng = np.random.default_rng(0)
    fold_abs = [np.abs(rng.normal(0, 1, size=(200, 5))) for _ in range(5)]
    for p in (0.5, 0.8, 0.9):
        cov = cross_fold_conformal_coverage(fold_abs, p)
        assert abs(cov - p) < 0.05


def test_halfwidths_increase_with_coverage():
    rng = np.random.default_rng(1)
    fold_abs = [np.abs(rng.normal(0, 1, size=(100, 4))) for _ in range(4)]
    hw50 = np.mean(cross_fold_halfwidths(fold_abs, 0.5))
    hw95 = np.mean(cross_fold_halfwidths(fold_abs, 0.95))
    assert hw95 > hw50


def test_halfwidths_exclude_own_fold():
    # fold 0 has huge residuals; its half-width (from others) must stay small
    fold_abs = [np.full((50, 3), 100.0), np.abs(np.random.default_rng(2).normal(0, 1, (50, 3))),
                np.abs(np.random.default_rng(3).normal(0, 1, (50, 3)))]
    hw = cross_fold_halfwidths(fold_abs, 0.9)
    assert hw[0].mean() < 5.0   # fold 0's band comes from the small-residual folds
