"""Tests for Stage 4 product logic and the physics-informed loss terms."""
import numpy as np
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
