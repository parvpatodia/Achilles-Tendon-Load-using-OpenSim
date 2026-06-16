"""Tests for the data layer: trial invariants, synthetic source, QC."""
import numpy as np
import pytest

from achilles.data.fukuchi import FukuchiDataSource
from achilles.data.synthetic import SyntheticGaitSource
from achilles.data.trial import GaitTrial


def test_trial_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        GaitTrial("S", "R", 3.5, 70, np.zeros(101), np.zeros(50),
                  np.zeros(101), np.zeros(101))


def test_trial_rejects_bad_side():
    with pytest.raises(ValueError):
        GaitTrial("S", "X", 3.5, 70, np.zeros(10), np.zeros(10),
                  np.zeros(10), np.zeros(10))


def test_bw_conversion():
    t = GaitTrial("S", "R", 3.5, 100, np.zeros(3), np.zeros(3),
                  np.zeros(3), np.array([9.81, 19.62, 0.0]))
    np.testing.assert_allclose(t.vgrf_bw, [1.0, 2.0, 0.0], rtol=1e-6)


def test_synthetic_source_shapes_and_signs():
    src = SyntheticGaitSource(n_subjects=3, seed=0)
    trials = src.load_trials()
    assert len(trials) == 3 * 3 * 2  # subjects x speeds x sides
    for t in trials:
        assert len(t.gait_phase) == 101
        assert np.all(t.vgrf_n_per_kg >= 0)
        assert t.vgrf_bw.max() < 4.0  # physiological


def test_synthetic_peak_force_reasonable():
    from achilles.biomech.achilles import AchillesLoadModel
    src = SyntheticGaitSource(n_subjects=5, seed=1)
    model = AchillesLoadModel()
    peaks = [model.compute(t).peak_force_bw for t in src.load_trials()]
    assert 3.0 <= np.mean(peaks) <= 8.0


def test_physiological_filter_rejects_corrupt():
    # peak moment in the thousands (the corrupt reduced-schema case) is rejected
    assert not FukuchiDataSource._is_physiological(
        np.array([5000.0, -1000.0]), np.array([20.0, 10.0]))
    assert FukuchiDataSource._is_physiological(
        np.array([2.5, 0.1]), np.array([23.0, 5.0]))
