"""Calibrated uncertainty: deep ensemble for the prediction, cross-fold
CONFORMAL calibration for the bands.

A deep ensemble gives a mean and a notion of model disagreement, but its raw
spread is not calibrated for regression (the errors here are heavy-tailed, so a
Gaussian sigma under-covers). Conformal prediction fixes this distribution-free:
the band half-width at coverage p is the empirical p-quantile of held-out
absolute residuals, and by construction a nominal p% band covers ~p% of unseen
points. We do it CROSS-FOLD (each fold's bands use the other folds' residuals),
so there is no calibration-on-test leakage, and we report nominal-vs-empirical
coverage to prove it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from achilles.biomech.achilles import AchillesLoadModel
from achilles.ml.dataset import AchillesSequenceDataset, build_samples
from achilles.ml.trainer import Trainer, TrainConfig


def cross_fold_halfwidths(fold_abs_resid: list[np.ndarray], p: float) -> list[np.ndarray]:
    """Per-fold, per-timestep conformal half-width = the p-quantile of the OTHER
    folds' absolute residuals (so a fold's bands never see its own residuals)."""
    n = len(fold_abs_resid)
    hw = []
    for i in range(n):
        other = np.concatenate([fold_abs_resid[j] for j in range(n) if j != i])
        hw.append(np.quantile(other, p, axis=0))
    return hw


def cross_fold_conformal_coverage(fold_abs_resid: list[np.ndarray], p: float) -> float:
    """Empirical coverage of the cross-fold conformal p-band (should be ~p)."""
    hw = cross_fold_halfwidths(fold_abs_resid, p)
    covered = [fold_abs_resid[i] <= hw[i][None, :] for i in range(len(fold_abs_resid))]
    return float(np.mean(np.concatenate([c.ravel() for c in covered])))


@dataclass
class UQResult:
    true: np.ndarray          # (N, T) held-out truth
    mean: np.ndarray          # (N, T) ensemble mean
    band_lo: np.ndarray       # (N, T) 90% conformal lower band
    band_hi: np.ndarray       # (N, T) 90% conformal upper band
    subject_ids: list[str]
    nominal: np.ndarray
    empirical: np.ndarray     # conformal coverage at each nominal level
    mean_halfwidth_bw: float  # mean 90% band half-width (BW)
    phase: np.ndarray

    def coverage_table(self):
        return list(zip(self.nominal.tolist(), self.empirical.tolist()))


def deep_ensemble_cv(trials, k: int = 5, n_models: int = 5, epochs: int = 120,
                     seed: int = 0) -> UQResult:
    load_model = AchillesLoadModel()
    subjects = sorted({t.subject_id for t in trials})
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    folds = [list(f) for f in np.array_split(subjects, k)]

    # 1) train an ensemble per fold; collect held-out mean predictions + truth
    fold_true, fold_mean, fold_sid = [], [], []
    for test_subj in folds:
        ts = set(test_subj)
        train_t = [t for t in trials if t.subject_id not in ts]
        test_t = [t for t in trials if t.subject_id in ts]
        train_ds = AchillesSequenceDataset(build_samples(train_t, load_model))
        test_ds = AchillesSequenceDataset(build_samples(test_t, load_model),
                                          train_ds.feat_mean, train_ds.feat_std)
        Xte = np.stack([test_ds[i]["x"].numpy() for i in range(len(test_ds))]).astype(np.float32)
        Yte = np.stack([s.y_bw for s in test_ds.samples])
        preds = []
        for m in range(n_models):
            tr = Trainer(train_ds, test_ds, TrainConfig(epochs=epochs, seed=seed + 1000 * m))
            tr.train(verbose=False)
            tr.model.eval()
            with torch.no_grad():
                preds.append(np.clip(tr.model(torch.from_numpy(Xte)).numpy(), 0, None))
        fold_true.append(Yte)
        fold_mean.append(np.stack(preds).mean(0))
        fold_sid.append([s.subject_id for s in test_ds.samples])

    abs_resid = [np.abs(m - t) for m, t in zip(fold_mean, fold_true)]  # per fold (n_i, T)
    nominal = np.array([0.5, 0.8, 0.9, 0.95])

    # 2) cross-fold conformal coverage (distribution-free, no calibration leakage)
    empirical = np.array([cross_fold_conformal_coverage(abs_resid, p) for p in nominal])

    # 3) assemble 90% bands for every held-out curve
    hw90 = cross_fold_halfwidths(abs_resid, 0.9)
    band_lo = np.concatenate([np.clip(fold_mean[i] - hw90[i][None, :], 0, None) for i in range(len(folds))])
    band_hi = np.concatenate([fold_mean[i] + hw90[i][None, :] for i in range(len(folds))])
    true = np.concatenate(fold_true)
    mean = np.concatenate(fold_mean)
    sid = [s for f in fold_sid for s in f]
    mean_hw = float(np.mean(np.concatenate([hw90[i] for i in range(len(folds))])))

    return UQResult(true, mean, band_lo, band_hi, sid, nominal, empirical,
                    mean_hw, np.linspace(0, 100, true.shape[1]))
