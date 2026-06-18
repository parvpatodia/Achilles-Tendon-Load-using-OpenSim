"""Subject-wise k-fold cross-validation for the Achilles surrogate.

A single held-out split can be lucky or unlucky. K-fold subject-wise CV holds
every subject out exactly once, so the reported generalisation is over the
whole cohort, with a mean +/- std across folds that shows the spread. This is
the honest way to claim "it works on unseen people".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import torch

from achilles.biomech.achilles import AchillesLoadModel
from achilles.data.trial import GaitTrial
from achilles.ml.baselines import SequenceModel
from achilles.ml.dataset import AchillesSequenceDataset, build_samples
from achilles.ml.losses import LossWeights
from achilles.ml.trainer import Trainer, TrainConfig


@dataclass
class CVResult:
    fold_r2: list[float]
    fold_rmse: list[float]
    pooled_r2: float
    pooled_rmse_bw: float
    pooled_mae_bw: float
    phase: np.ndarray
    true_curves: list[np.ndarray]   # pooled across all held-out folds
    pred_curves: list[np.ndarray]
    n_subjects: int
    k: int

    @property
    def mean_r2(self) -> float:
        return float(np.mean(self.fold_r2))

    @property
    def std_r2(self) -> float:
        return float(np.std(self.fold_r2))


def subject_kfold(
    trials: list[GaitTrial],
    k: int = 5,
    weights: LossWeights | None = None,
    epochs: int = 200,
    seed: int = 0,
    load_model: AchillesLoadModel | None = None,
    verbose: bool = True,
) -> CVResult:
    load_model = load_model or AchillesLoadModel()
    subjects = sorted({t.subject_id for t in trials})
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    folds = [list(f) for f in np.array_split(subjects, k)]

    fold_r2, fold_rmse = [], []
    true_all, pred_all = [], []

    for i, test_subj in enumerate(folds):
        test_set = set(test_subj)
        train_t = [t for t in trials if t.subject_id not in test_set]
        test_t = [t for t in trials if t.subject_id in test_set]

        train_ds = AchillesSequenceDataset(build_samples(train_t, load_model))
        test_ds = AchillesSequenceDataset(build_samples(test_t, load_model),
                                          train_ds.feat_mean, train_ds.feat_std)
        cfg = TrainConfig(epochs=epochs, weights=weights or LossWeights(), seed=seed)
        trainer = Trainer(train_ds, test_ds, cfg)
        trainer.train(verbose=False)
        ev = trainer.evaluate()
        fold_r2.append(ev.r2)
        fold_rmse.append(ev.rmse_bw)
        true_all.extend(ev.true_curves)
        pred_all.extend(ev.pred_curves)
        if verbose:
            print(f"  fold {i+1}/{k}  held-out subjects={len(test_subj)}  "
                  f"R2={ev.r2:.3f}  RMSE={ev.rmse_bw:.2f} BW")

    t = np.concatenate(true_all)
    p = np.concatenate(pred_all)
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    pooled_r2 = 1.0 - ss_res / ss_tot
    pooled_rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    pooled_mae = float(np.mean(np.abs(t - p)))

    return CVResult(
        fold_r2=fold_r2, fold_rmse=fold_rmse,
        pooled_r2=pooled_r2, pooled_rmse_bw=pooled_rmse, pooled_mae_bw=pooled_mae,
        phase=np.linspace(0, 100, len(true_all[0])),
        true_curves=true_all, pred_curves=pred_all,
        n_subjects=len(subjects), k=k,
    )


# -- model comparison on shared folds ---------------------------------------
def _subject_folds(trials: list[GaitTrial], k: int, seed: int) -> list[list[str]]:
    subjects = sorted({t.subject_id for t in trials})
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    return [list(f) for f in np.array_split(subjects, k)]


def _dataset_arrays(ds: AchillesSequenceDataset):
    """Stack standardised inputs (N,C,T), targets (N,T) and per-sample subjects."""
    X = np.stack([ds[i]["x"].numpy() for i in range(len(ds))])
    Y = np.stack([ds[i]["y"].numpy() for i in range(len(ds))])
    subj = [s.subject_id for s in ds.samples]
    return X, Y, subj


def compare_models_kfold(
    trials: list[GaitTrial],
    baselines: list[SequenceModel],
    k: int = 5,
    epochs: int = 200,
    seed: int = 0,
    weights: LossWeights | None = None,
    include_cnn: bool = True,
) -> dict[str, dict]:
    """Score the CNN and each baseline on identical subject-wise folds.

    Returns model_name -> {subject_ids, true_curves, pred_curves} pooled over
    held-out folds, so every model is judged on exactly the same unseen people.
    """
    load_model = AchillesLoadModel()
    folds = _subject_folds(trials, k, seed)
    out: dict[str, dict] = {}

    def _collect(name, subj, true, pred):
        d = out.setdefault(name, {"subject_ids": [], "true_curves": [], "pred_curves": []})
        d["subject_ids"].extend(subj)
        d["true_curves"].extend(list(true))
        d["pred_curves"].extend(list(np.clip(pred, 0.0, None)))  # tendon force >= 0

    for test_subj in folds:
        test_set = set(test_subj)
        train_t = [t for t in trials if t.subject_id not in test_set]
        test_t = [t for t in trials if t.subject_id in test_set]
        train_ds = AchillesSequenceDataset(build_samples(train_t, load_model))
        test_ds = AchillesSequenceDataset(build_samples(test_t, load_model),
                                          train_ds.feat_mean, train_ds.feat_std)
        Xtr, Ytr, _ = _dataset_arrays(train_ds)
        Xte, Yte, subj_te = _dataset_arrays(test_ds)

        for b in baselines:
            b.fit(Xtr, Ytr)
            _collect(b.name, subj_te, Yte, b.predict(Xte))

        if include_cnn:
            cfg = TrainConfig(epochs=epochs, weights=weights or LossWeights(), seed=seed)
            trainer = Trainer(train_ds, test_ds, cfg)
            trainer.train(verbose=False)
            trainer.model.eval()
            with torch.no_grad():
                pred = trainer.model(torch.from_numpy(Xte.astype(np.float32))).numpy()
            _collect("physics-guided CNN", subj_te, Yte, pred)

    return out
