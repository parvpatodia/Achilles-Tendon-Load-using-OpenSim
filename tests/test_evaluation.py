"""Tests for the evaluation metrics and baseline models."""
import numpy as np

from achilles.ml.baselines import MeanCurveModel, RidgeSequenceModel
from achilles.ml.evaluation import (bland_altman, cluster_bootstrap_r2,
                                     evaluate_predictions, r2_score, rmse)


def test_r2_and_rmse_perfect_and_mean():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(t, t) == 1.0
    assert rmse(t, t) == 0.0
    # predicting the mean gives R^2 = 0 by definition
    assert abs(r2_score(t, np.full_like(t, t.mean()))) < 1e-12


def test_bland_altman_bias_and_loa():
    true = np.array([5.0, 5.0, 5.0, 5.0])
    pred = true + np.array([0.1, -0.1, 0.2, -0.2])
    a = bland_altman(true, pred)
    assert abs(a.bias) < 1e-9
    assert a.loa_upper > 0 > a.loa_lower


def test_cluster_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    by_subject = {}
    for s in range(8):
        t = rng.normal(3, 1, 50)
        by_subject[f"S{s}"] = (t, t + rng.normal(0, 0.2, 50))
    lo, hi = cluster_bootstrap_r2(by_subject, n_boot=300, seed=0)
    pooled = r2_score(np.concatenate([t for t, _ in by_subject.values()]),
                      np.concatenate([p for _, p in by_subject.values()]))
    assert lo <= pooled <= hi
    assert lo < hi


def test_evaluate_predictions_peak_and_loaded():
    # two subjects, two curves each; pred = true -> perfect
    phase = np.linspace(0, 100, 101)
    curve = np.clip(np.sin(np.pi * phase / 40) * 5, 0, None); curve[phase > 40] = 0
    subj = ["A", "A", "B", "B"]
    true = [curve, curve * 0.9, curve * 1.1, curve]
    m = evaluate_predictions(subj, true, [c.copy() for c in true], n_boot=100)
    assert m.r2 > 0.999 and m.r2_loaded > 0.999
    assert m.peak_mae_bw < 1e-6 and m.peak_mape_pct < 1e-3
    assert set(m.per_subject_r2) == {"A", "B"}


def test_baseline_shapes_and_ridge_beats_mean():
    rng = np.random.default_rng(0)
    N, C, T = 40, 3, 101
    X = rng.normal(size=(N, C, T))
    # target genuinely depends on channel 0 -> ridge should beat the mean curve
    Y = X[:, 0, :] * 2.0 + rng.normal(0, 0.1, size=(N, T))
    mean_pred = MeanCurveModel().fit(X, Y).predict(X)
    ridge_pred = RidgeSequenceModel(channels=(0,)).fit(X, Y).predict(X)
    assert mean_pred.shape == (N, T) and ridge_pred.shape == (N, T)
    assert r2_score(Y, ridge_pred) > r2_score(Y, mean_pred)
