"""Tests for the walking data source.

Skips automatically if the (large, optional) walking data is not downloaded,
so the suite stays green on a fresh clone. Run with the data present to
validate parsing and the physiological range.
"""
import numpy as np
import pytest

from achilles.biomech.achilles import AchillesLoadModel
from achilles.config import DATA_RAW

pytestmark = pytest.mark.skipif(
    not (DATA_RAW / "wbds").exists(),
    reason="walking data absent (run: python scripts/download_data.py --walking)",
)


def _walking_trials():
    from achilles.data.walking import WalkingDataSource
    return WalkingDataSource().load_trials()


def test_walking_trials_load_without_grf():
    trials = _walking_trials()
    assert len(trials) > 100
    for t in trials[:20]:
        assert t.task == "walk"
        assert not t.has_grf          # walking carries no normalised GRF
        assert len(t.ankle_moment_nm_per_kg) == len(t.gait_phase)


def test_walking_achilles_force_in_literature_range():
    """Walking peak Achilles force must land ~2-3.5 BW, below running's ~5 BW."""
    m = AchillesLoadModel()
    peaks = np.array([m.compute(t).peak_force_bw for t in _walking_trials()])
    assert 2.0 <= peaks.mean() <= 3.5
    assert peaks.max() < 5.0
