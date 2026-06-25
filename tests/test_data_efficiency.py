"""Tests for the data-efficiency experiment.

Kept fast: tiny synthetic cohort, 3 epochs, 1 seed. We test the harness
contract (shapes, finiteness, fixed-test-set bookkeeping, fair physics-vs-
data-only ablation), not the scientific conclusion, which needs the real run.
"""
import numpy as np

from achilles.data.synthetic import SyntheticGaitSource
from achilles.ml.data_efficiency import (DATA_ONLY, LINEAR, MODELS, PHYSICS,
                                          data_efficiency_curve)


def _result():
    trials = SyntheticGaitSource(n_subjects=14, seed=0).load_trials()
    return data_efficiency_curve(
        trials, sizes=(3, 6), n_seeds=1, n_test_subjects=3, epochs=3, base_seed=0)


def test_shapes_and_finiteness():
    r = _result()
    assert r.sizes == [3, 6]
    assert r.models == MODELS
    assert r.n_test_subjects == 3
    for m in MODELS:
        assert r.mean_loaded_r2[m].shape == (2,)
        assert r.std_loaded_r2[m].shape == (2,)
        assert np.all(np.isfinite(r.mean_loaded_r2[m]))      # training actually ran
    assert r.gap_physics_minus_dataonly().shape == (2,)
    assert isinstance(r.table(), str) and "train subjects" in r.table()


def test_sizes_clipped_to_pool():
    # pool = 14 - 3 = 11 subjects; a request for 20 must be dropped, not crash.
    trials = SyntheticGaitSource(n_subjects=14, seed=1).load_trials()
    r = data_efficiency_curve(trials, sizes=(5, 20), n_seeds=1,
                              n_test_subjects=3, epochs=2, base_seed=0)
    assert r.sizes == [5]


def test_single_seed_gives_zero_spread():
    # with one seed there is no seed-to-seed variance to report.
    r = _result()
    for m in MODELS:
        np.testing.assert_allclose(r.std_loaded_r2[m], 0.0, atol=1e-12)
