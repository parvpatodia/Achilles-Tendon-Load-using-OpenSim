"""Evaluation metrics with statistically honest uncertainty.

Designed to survive scrutiny from a statistics-trained reviewer:

- Point metrics: R^2, RMSE, MAE (in body weights, interpretable).
- Agreement: Bland-Altman bias and 95% limits of agreement (the clinical
  standard for "do two methods agree", not just "do they correlate").
- Uncertainty: a CLUSTER bootstrap that resamples whole subjects, not samples.
  Samples within a subject/trial are correlated, so a naive sample bootstrap
  would understate the confidence interval. Resampling subjects respects that.
- Per-subject distribution: the worst subject matters for a wearable, so we
  report the spread of per-subject R^2, not only the pooled number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LOADED_THRESHOLD_BW = 0.5  # "loaded phase" = tendon force above this (excludes swing)


def r2_score(true: np.ndarray, pred: np.ndarray) -> float:
    true, pred = np.asarray(true).ravel(), np.asarray(pred).ravel()
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    true, pred = np.asarray(true).ravel(), np.asarray(pred).ravel()
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(true: np.ndarray, pred: np.ndarray) -> float:
    true, pred = np.asarray(true).ravel(), np.asarray(pred).ravel()
    return float(np.mean(np.abs(true - pred)))


@dataclass
class Agreement:
    bias: float          # mean(pred - true)
    loa_lower: float     # bias - 1.96 * sd
    loa_upper: float     # bias + 1.96 * sd
    sd_diff: float


def physical_validity(curves: list[np.ndarray]) -> dict:
    """Check predicted force curves against the physics the constraints encode:
    non-negativity (a tendon pulls, never pushes) and bounded loading rate
    (smoothness). Returns the fraction of negative samples and the mean
    roughness (mean |second difference|), so the physics claim is measured."""
    arr = np.vstack([np.asarray(c, dtype=float) for c in curves])
    neg_fraction = float(np.mean(arr < 0))
    roughness = float(np.mean(np.abs(arr[:, 2:] - 2 * arr[:, 1:-1] + arr[:, :-2])))
    return {"neg_fraction": neg_fraction, "roughness": roughness}


def bland_altman(true: np.ndarray, pred: np.ndarray) -> Agreement:
    """Bland-Altman agreement of predicted vs reference (units of the inputs)."""
    true, pred = np.asarray(true).ravel(), np.asarray(pred).ravel()
    diff = pred - true
    bias, sd = float(diff.mean()), float(diff.std(ddof=1))
    return Agreement(bias, bias - 1.96 * sd, bias + 1.96 * sd, sd)


@dataclass
class CVMetrics:
    r2: float
    rmse: float
    mae: float
    r2_ci: tuple[float, float]
    per_subject_r2: dict[str, float]
    agreement: Agreement
    n_samples: int
    n_subjects: int
    # honest extras: the force is ~0 for most of the cycle (swing), which
    # inflates full-curve R^2. r2_loaded is computed only on the loaded phase;
    # peak error is the clinically relevant quantity.
    r2_loaded: float = float("nan")
    peak_mae_bw: float = float("nan")
    peak_mape_pct: float = float("nan")
    peak_agreement: Agreement | None = None
    # loaded-phase R^2 per held-out subject: a wearable is judged on its weakest
    # athlete, so the worst case is reported, not only the cohort average.
    per_subject_r2_loaded: dict[str, float] = field(default_factory=dict)

    @property
    def r2_subject_summary(self) -> tuple[float, float, float]:
        v = np.array(list(self.per_subject_r2.values()))
        return float(np.min(v)), float(np.median(v)), float(np.max(v))

    @property
    def r2_loaded_subject_summary(self) -> tuple[float, float, float]:
        """(worst, median, best) loaded-phase R^2 across held-out subjects."""
        v = np.array([x for x in self.per_subject_r2_loaded.values() if np.isfinite(x)])
        return float(np.min(v)), float(np.median(v)), float(np.max(v))


def cluster_bootstrap_r2(
    by_subject: dict[str, tuple[np.ndarray, np.ndarray]],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """95% CI for pooled R^2 by resampling SUBJECTS with replacement.

    by_subject maps subject_id -> (true_concat, pred_concat) for that subject.
    """
    subjects = list(by_subject)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pick = rng.choice(len(subjects), size=len(subjects), replace=True)
        t = np.concatenate([by_subject[subjects[i]][0] for i in pick])
        p = np.concatenate([by_subject[subjects[i]][1] for i in pick])
        boot.append(r2_score(t, p))
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return lo, hi


def evaluate_predictions(
    subject_ids: list[str],
    true_curves: list[np.ndarray],
    pred_curves: list[np.ndarray],
    n_boot: int = 1000,
    seed: int = 0,
) -> CVMetrics:
    """Build the full metric set from per-trial held-out predictions."""
    by_subject: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for sid, t, p in zip(subject_ids, true_curves, pred_curves):
        by_subject.setdefault(sid, []).append((t, p))

    concat = {s: (np.concatenate([t for t, _ in v]),
                  np.concatenate([p for _, p in v])) for s, v in by_subject.items()}
    per_subject_r2 = {s: r2_score(t, p) for s, (t, p) in concat.items()}

    def _loaded_subject_r2(t, p):
        m = t > LOADED_THRESHOLD_BW
        return r2_score(t[m], p[m]) if m.any() else float("nan")
    per_subject_r2_loaded = {s: _loaded_subject_r2(t, p) for s, (t, p) in concat.items()}

    all_true = np.concatenate([t for t, _ in concat.values()])
    all_pred = np.concatenate([p for _, p in concat.values()])

    # loaded-phase R^2 (exclude the trivial near-zero swing phase)
    loaded = all_true > LOADED_THRESHOLD_BW
    r2_loaded = r2_score(all_true[loaded], all_pred[loaded]) if loaded.any() else float("nan")

    # per-curve peak agreement (the clinically relevant number)
    peak_true = np.array([t.max() for t in true_curves])
    peak_pred = np.array([p.max() for p in pred_curves])
    peak_mae = float(np.mean(np.abs(peak_pred - peak_true)))
    peak_mape = float(np.mean(np.abs(peak_pred - peak_true) / peak_true) * 100)

    return CVMetrics(
        r2=r2_score(all_true, all_pred),
        rmse=rmse(all_true, all_pred),
        mae=mae(all_true, all_pred),
        r2_ci=cluster_bootstrap_r2(concat, n_boot=n_boot, seed=seed),
        per_subject_r2=per_subject_r2,
        agreement=bland_altman(all_true, all_pred),
        n_samples=int(all_true.size),
        n_subjects=len(concat),
        r2_loaded=r2_loaded,
        peak_mae_bw=peak_mae,
        peak_mape_pct=peak_mape,
        peak_agreement=bland_altman(peak_true, peak_pred),
        per_subject_r2_loaded=per_subject_r2_loaded,
    )
