"""Tests for Stage 4 product logic and the physics-informed loss terms."""
import numpy as np
import pytest
import torch

from achilles.ml.dataset import subject_wise_split
from achilles.ml.losses import LossWeights, PhysicsInformedLoss
from achilles.product.load_index import (AccumulationTimeline, SessionSimulator)


def test_session_simulator_deterministic():
    a = SessionSimulator(n_sessions=10, seed=3).generate()
    b = SessionSimulator(n_sessions=10, seed=3).generate()
    np.testing.assert_array_equal(a.volume, b.volume)
    assert len(a.sessions) == 10


def test_accumulation_monotonic_cumulative():
    plan = SessionSimulator(n_sessions=12).generate()
    res = AccumulationTimeline().compute(base_impulse=100.0, plan=plan)
    assert np.all(np.diff(res.cumulative) > 0)  # cumulative load only grows
    assert res.acwr.shape == res.sessions.shape


def test_acwr_flags_overload_spike():
    """The deliberate spike must push ACWR into the watch zone (>1.5)."""
    plan = SessionSimulator(n_sessions=14, spike_session=11, spike_mult=2.8).generate()
    res = AccumulationTimeline().compute(base_impulse=100.0, plan=plan)
    assert np.nanmax(res.acwr) > 1.5


def test_subject_wise_split_no_leakage():
    from achilles.data.synthetic import SyntheticGaitSource
    trials = SyntheticGaitSource(n_subjects=8, seed=0).load_trials()
    train, test, test_subj = subject_wise_split(trials, test_frac=0.25, seed=0)
    train_subj = {t.subject_id for t in train}
    assert set(test_subj).isdisjoint(train_subj)  # the honest-generalisation guarantee
    assert len(train) + len(test) == len(trials)


def test_asymmetry_index_sign_and_formula():
    """ASI is +ve when the right limb is loaded more, and equals the formula."""
    from achilles.biomech.achilles import AchillesLoadModel
    from achilles.data.synthetic import SyntheticGaitSource
    from achilles.product.load_index import AsymmetryAnalyzer, SessionSimulator
    trials = SyntheticGaitSource(n_subjects=2, seed=0).load_trials()
    m = AchillesLoadModel()
    L = m.compute(next(t for t in trials if t.side == "L"))
    R = m.compute(next(t for t in trials if t.side == "R"))
    plan = SessionSimulator(n_sessions=8).generate()
    res = AsymmetryAnalyzer(L, R, seed=0).analyze(plan)
    expected = 100.0 * (res.right_load - res.left_load) / (0.5 * (res.right_load + res.left_load))
    np.testing.assert_allclose(res.asi_pct, expected, rtol=1e-9)
    # synthetic right limb is the stronger side, so ASI is positive
    assert res.asi_pct[0] > 0


def test_stress_safety_factor():
    """Safety factor = ultimate stress / peak stress, and is finite/>1 here."""
    import numpy as _np
    from achilles.biomech.achilles import AchillesLoadModel
    from achilles.biomech.moment_arm import ConstantMomentArm
    from achilles.config import TENDON
    from achilles.data.synthetic import SyntheticGaitSource
    t = SyntheticGaitSource(n_subjects=1, seed=0).load_trials()[0]
    res = AchillesLoadModel(moment_arm=ConstantMomentArm(0.05)).compute(t)
    expected = TENDON.ultimate_stress_pa / _np.max(res.stress_pa)
    assert res.stress_safety_factor == pytest.approx(expected, rel=1e-9)
    assert res.stress_safety_factor > 1.0


def test_physics_loss_penalises_negative_force():
    """Non-negativity term must be larger when predictions go negative."""
    T = 101
    batch = {
        "y": torch.ones(2, T),
        "bw": torch.tensor([700.0, 700.0]),
        "moment_arm_m": torch.full((2, T), 0.05),
        "moment_nm": torch.full((2, T), 35.0),
    }
    loss = PhysicsInformedLoss(LossWeights(data=0, non_neg=1, moment=0, smooth=0))
    _, pos = loss(torch.ones(2, T), batch)
    _, neg = loss(-torch.ones(2, T), batch)
    assert neg["non_neg"] > pos["non_neg"]
    assert pos["non_neg"] == 0.0
