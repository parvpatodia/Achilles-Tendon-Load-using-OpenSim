"""Tests for the leave-one-speed-out load-regime generalization harness."""
import numpy as np

from achilles.data.synthetic import SyntheticGaitSource
from achilles.ml.generalization import leave_one_speed_out


def _result():
    trials = SyntheticGaitSource(n_subjects=10, seed=0).load_trials()
    return leave_one_speed_out(trials)


def test_one_entry_per_speed_and_finite():
    r = _result()
    assert r.speeds == [2.5, 3.5, 4.5]            # synthetic source speeds
    for s in r.speeds:
        assert np.isfinite(r.loaded_r2[s])
        assert np.isfinite(r.peak_mape_pct[s]) and r.peak_mape_pct[s] >= 0
        assert r.n_test[s] > 0
    assert "held-out speed" in r.table()


def test_lowest_speed_flagged_toward_walking():
    r = _result()
    assert "toward walking loads" in r.table()
    # the flag sits on the lowest speed only
    assert r.table().count("toward walking loads") == 1
