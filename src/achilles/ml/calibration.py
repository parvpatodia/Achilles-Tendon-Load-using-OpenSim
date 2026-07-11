"""Subject-specific calibration: remove the systematic per-person bias with a
handful of a new athlete's own labelled steps.

Why this exists. The subject-wise surrogate is scored on people it never saw,
and on the worst athletes its dominant error is a SYSTEMATIC per-person bias (a
near-constant scale/offset), not random scatter: the repo's worst held-out
runner is over-predicted at push-off by ~1.2 BW on every stride (README 6c). A
systematic bias is exactly what a low-parameter per-subject correction removes,
and it is also where a wrong per-person moment arm hides (a bad lever scales
force almost linearly, so an affine fit absorbs its constant part).

Where the calibration labels come from. Fitting the correction needs the true
tendon load for the athlete's calibration steps. In deployment those come once,
at onboarding, from the same lab reference used as the training target (README
9.3: pretrain on public data, fine-tune to the lab estimate). So this is a
realistic one-time step, not test leakage: the base model never trains on the
held-out athlete, and the calibration steps are excluded from scoring.

Method. Take each athlete's out-of-fold base predictions (from the existing
subject-wise k-fold). For that athlete use K of their step-cycles to fit the
correction, apply it to their REMAINING steps, and score there. The calibrated
and uncalibrated numbers are computed on the identical remaining steps, so the
delta is the calibration effect and nothing else. Repeat over a few seeded
draws of which K steps are used, and average, so the result is not a lucky split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from achilles.data.trial import GaitTrial
from achilles.ml.baselines import RidgeSequenceModel, SequenceModel
from achilles.ml.cross_val import compare_models_kfold
from achilles.ml.evaluation import LOADED_THRESHOLD_BW, r2_score

Correction = tuple[float, float]  # (a, b) applied as a * pred + b


# -- per-subject corrections (fit on the loaded phase of the calib steps) -----
def _loaded_points(cal_true: list[np.ndarray], cal_pred: list[np.ndarray],
                   min_points: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Pool the calibration curves and keep the loaded-phase samples (the swing
    zeros carry no bias information). Fall back to all samples if too few."""
    t = np.concatenate([np.asarray(c).ravel() for c in cal_true])
    p = np.concatenate([np.asarray(c).ravel() for c in cal_pred])
    m = t > LOADED_THRESHOLD_BW
    if m.sum() < min_points:
        m = np.ones_like(t, dtype=bool)
    return t[m], p[m]


def fit_identity(cal_true, cal_pred) -> Correction:
    return (1.0, 0.0)


def fit_offset(cal_true, cal_pred) -> Correction:
    """One parameter: the additive shift that matches mean load on the loaded phase."""
    t, p = _loaded_points(cal_true, cal_pred)
    return (1.0, float(np.mean(t - p)))


def fit_affine(cal_true, cal_pred) -> Correction:
    """Two parameters: least-squares scale + offset on the loaded phase. This is
    the minimal fix for a systematic per-person bias, and it also absorbs the
    constant part of a wrong per-athlete moment arm."""
    t, p = _loaded_points(cal_true, cal_pred)
    A = np.vstack([p, np.ones_like(p)]).T
    (a, b), *_ = np.linalg.lstsq(A, t, rcond=None)
    return (float(a), float(b))


_FITTERS: dict[str, Callable[[list, list], Correction]] = {
    "identity": fit_identity,
    "offset": fit_offset,
    "affine": fit_affine,
}


def _apply(pred_curves: list[np.ndarray], ab: Correction) -> list[np.ndarray]:
    a, b = ab
    return [np.clip(a * np.asarray(c) + b, 0.0, None) for c in pred_curves]  # force >= 0


# -- metrics on a pooled set of held-out eval curves -------------------------
def _pooled_loaded_r2(true_curves: list[np.ndarray], pred_curves: list[np.ndarray]) -> float:
    if not true_curves:
        return float("nan")
    t = np.concatenate([np.asarray(c).ravel() for c in true_curves])
    p = np.concatenate([np.asarray(c).ravel() for c in pred_curves])
    m = t > LOADED_THRESHOLD_BW
    return r2_score(t[m], p[m]) if m.any() else float("nan")


def _worst_subject_loaded_r2(subjects, true_curves, pred_curves) -> float:
    by: dict[str, tuple[list, list]] = {}
    for s, t, p in zip(subjects, true_curves, pred_curves):
        ts, ps = by.setdefault(s, ([], []))
        ts.append(t)
        ps.append(p)
    vals = [_pooled_loaded_r2(ts, ps) for ts, ps in by.values()]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.min(vals)) if vals else float("nan")


def _peak_mape(true_curves: list[np.ndarray], pred_curves: list[np.ndarray]) -> float:
    if not true_curves:
        return float("nan")
    pt = np.array([np.asarray(c).max() for c in true_curves])
    pp = np.array([np.asarray(c).max() for c in pred_curves])
    return float(np.mean(np.abs(pp - pt) / pt) * 100)


@dataclass
class CalibrationResult:
    method: str
    base_model: str
    ks: list[int]
    n_draws: int
    k_fold: int
    n_subjects: int
    # matched on identical eval steps, per calibration-set size K
    uncal_loaded_r2: dict[int, float] = field(default_factory=dict)
    cal_loaded_r2: dict[int, float] = field(default_factory=dict)
    uncal_worst_loaded_r2: dict[int, float] = field(default_factory=dict)
    cal_worst_loaded_r2: dict[int, float] = field(default_factory=dict)
    uncal_peak_mape: dict[int, float] = field(default_factory=dict)
    cal_peak_mape: dict[int, float] = field(default_factory=dict)

    def table(self) -> str:
        head = (f"method={self.method}  base={self.base_model}  draws={self.n_draws}  "
                f"k-fold={self.k_fold}  subjects={self.n_subjects}")
        cols = (f"{'K steps':>7s}  {'loaded R2 uncal->cal':>22s}  "
                f"{'worst-athlete uncal->cal':>26s}  {'peak MAPE% uncal->cal':>24s}")
        lines = [head, cols]
        for k in self.ks:
            lines.append(
                f"{k:>7d}  "
                f"{self.uncal_loaded_r2[k]:>9.3f} -> {self.cal_loaded_r2[k]:<9.3f}  "
                f"{self.uncal_worst_loaded_r2[k]:>11.3f} -> {self.cal_worst_loaded_r2[k]:<11.3f}  "
                f"{self.uncal_peak_mape[k]:>9.1f} -> {self.cal_peak_mape[k]:<9.1f}")
        return "\n".join(lines)


def subject_calibration_sweep(
    trials: list[GaitTrial],
    ks: tuple[int, ...] = (1, 2, 3),
    method: str = "affine",
    base_model_factory: Callable[[], SequenceModel] = lambda: RidgeSequenceModel(channels=None),
    k_fold: int = 5,
    n_draws: int = 5,
    seed: int = 0,
) -> CalibrationResult:
    """Sweep the calibration-set size K and report, per K, the uncalibrated vs
    calibrated loaded-phase R2, worst-athlete loaded R2, and peak error, all on
    identical held-out eval steps.
    """
    if method not in _FITTERS:
        raise ValueError(f"unknown method {method!r} (use {sorted(_FITTERS)})")
    fitter = _FITTERS[method]

    # One subject-wise k-fold gives every athlete an out-of-fold base prediction
    # (the base model never trains on the athlete it is scored on).
    base = base_model_factory()
    block = compare_models_kfold(trials, [base], k=k_fold, seed=seed, include_cnn=False)[base.name]

    by_subject: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for s, t, p in zip(block["subject_ids"], block["true_curves"], block["pred_curves"]):
        by_subject.setdefault(s, []).append((np.asarray(t), np.asarray(p)))

    res = CalibrationResult(method=method, base_model=base.name, ks=list(ks),
                            n_draws=n_draws, k_fold=k_fold, n_subjects=len(by_subject))
    rng = np.random.default_rng(seed)

    for k in ks:
        u_r2, c_r2, u_w, c_w, u_mape, c_mape = ([] for _ in range(6))
        for _ in range(n_draws):
            subs: list[str] = []
            u_true, u_pred, c_true, c_pred = [], [], [], []
            for s, curves in by_subject.items():
                if len(curves) <= k:            # need at least one step left to score
                    continue
                order = rng.permutation(len(curves))
                cal, ev = order[:k], order[k:]
                ab = fitter([curves[i][0] for i in cal], [curves[i][1] for i in cal])
                ev_true = [curves[i][0] for i in ev]
                ev_pred = [curves[i][1] for i in ev]
                ev_cal = _apply(ev_pred, ab)
                subs.extend([s] * len(ev))
                u_true.extend(ev_true); u_pred.extend(ev_pred)
                c_true.extend(ev_true); c_pred.extend(ev_cal)
            u_r2.append(_pooled_loaded_r2(u_true, u_pred))
            c_r2.append(_pooled_loaded_r2(c_true, c_pred))
            u_w.append(_worst_subject_loaded_r2(subs, u_true, u_pred))
            c_w.append(_worst_subject_loaded_r2(subs, c_true, c_pred))
            u_mape.append(_peak_mape(u_true, u_pred))
            c_mape.append(_peak_mape(c_true, c_pred))
        res.uncal_loaded_r2[k] = float(np.nanmean(u_r2))
        res.cal_loaded_r2[k] = float(np.nanmean(c_r2))
        res.uncal_worst_loaded_r2[k] = float(np.nanmean(u_w))
        res.cal_worst_loaded_r2[k] = float(np.nanmean(c_w))
        res.uncal_peak_mape[k] = float(np.nanmean(u_mape))
        res.cal_peak_mape[k] = float(np.nanmean(c_mape))
    return res
