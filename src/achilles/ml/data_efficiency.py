"""Data-efficiency experiment: does the physics-guided loss earn its place when
training subjects are scarce?

On the full cohort the physics-guided CNN ties a linear baseline (see the model
comparison), so the recommended deployable model is the compact linear map. But
the realistic deployment is a SMALL calibration cohort (about 30-50 athletes;
see README section 9), and a physics prior is a regulariser that should help
most exactly when labelled data is scarce. This measures it directly.

With a FIXED held-out test set of subjects (constant across every training size
and seed, so the metric is comparable), we grow the number of training subjects
and score three models on identical data:

  - physics-guided CNN  (data + non-negativity + moment-consistency + smoothness)
  - data-only CNN       (the SAME network and init, physics weights set to 0)
  - linear (ridge)      (the deployable baseline)

The CNN pair shares its seed, so init and batch order match and the only
difference is the loss: a clean ablation of the physics terms. The metric is
loaded-phase R^2 (the honest one; the swing phase is trivially near-zero). Each
(model, size) point is averaged over several seeds that resample WHICH subjects
land in the training pool, with the spread reported, because at small sample
size the seed-to-seed variance is the story.

The verdict is reported honestly whichever way it falls: a physics gain at small
n is a deployment-relevant finding; no gain strengthens the "linear is enough"
recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from achilles.biomech.achilles import AchillesLoadModel
from achilles.data.trial import GaitTrial
from achilles.ml.baselines import RidgeSequenceModel
from achilles.ml.dataset import AchillesSequenceDataset, build_samples
from achilles.ml.evaluation import LOADED_THRESHOLD_BW, r2_score
from achilles.ml.losses import LossWeights
from achilles.ml.trainer import Trainer, TrainConfig

PHYSICS = "physics-guided CNN"
DATA_ONLY = "data-only CNN"
LINEAR = "linear (ridge)"
MODELS = [PHYSICS, DATA_ONLY, LINEAR]

# WHY: the physics-guided weights are the package default; the data-only ablation
# zeroes every physics term so only the data MSE remains (same net, same init).
_PHYSICS_W = LossWeights()
_DATA_ONLY_W = LossWeights(data=1.0, non_neg=0.0, moment=0.0, smooth=0.0)


@dataclass
class DataEfficiencyResult:
    sizes: list[int]                         # training-subject counts (x-axis)
    models: list[str]                        # model labels in plot order
    mean_loaded_r2: dict[str, np.ndarray]    # label -> (len(sizes),)
    std_loaded_r2: dict[str, np.ndarray]     # label -> (len(sizes),) across seeds
    n_seeds: int
    n_test_subjects: int
    epochs: int

    def gap_physics_minus_dataonly(self) -> np.ndarray:
        """Loaded-R^2 advantage of the physics terms at each training size."""
        return self.mean_loaded_r2[PHYSICS] - self.mean_loaded_r2[DATA_ONLY]

    def table(self) -> str:
        lines = [f"{'train subjects':>14s} " + " ".join(f"{m:>20s}" for m in self.models)
                 + f"{'physics-data-only':>20s}"]
        gap = self.gap_physics_minus_dataonly()
        for j, n in enumerate(self.sizes):
            cells = " ".join(
                f"{self.mean_loaded_r2[m][j]:.3f}+/-{self.std_loaded_r2[m][j]:.3f}".rjust(20)
                for m in self.models
            )
            lines.append(f"{n:>14d} {cells}{gap[j]:>+20.3f}")
        return "\n".join(lines)


def _loaded_r2(true_curves: list[np.ndarray], pred_curves: list[np.ndarray]) -> float:
    """Pooled R^2 over the loaded phase only (same definition as evaluation.py)."""
    t = np.concatenate([np.asarray(c).ravel() for c in true_curves])
    p = np.concatenate([np.asarray(c).ravel() for c in pred_curves])
    mask = t > LOADED_THRESHOLD_BW
    return r2_score(t[mask], p[mask]) if mask.any() else float("nan")


def _curves(pred: np.ndarray) -> list[np.ndarray]:
    return [pred[i] for i in range(len(pred))]


def data_efficiency_curve(
    trials: list[GaitTrial],
    sizes: tuple[int, ...] = (4, 8, 12, 16, 20),
    n_seeds: int = 5,
    n_test_subjects: int = 8,
    epochs: int = 120,
    base_seed: int = 0,
) -> DataEfficiencyResult:
    load_model = AchillesLoadModel()
    all_subjects = sorted({t.subject_id for t in trials})

    # WHY: a single fixed test set (deterministic permutation, never trained on)
    # makes the score comparable across every training size and seed.
    perm = list(np.random.default_rng(base_seed).permutation(all_subjects))
    test_subj = set(perm[:n_test_subjects])
    pool = perm[n_test_subjects:]
    sizes = tuple(s for s in sizes if s <= len(pool))

    test_t = [t for t in trials if t.subject_id in test_subj]
    test_samples_template = build_samples(test_t, load_model)
    true_curves = [s.y_bw for s in test_samples_template]

    acc = {m: {n: [] for n in sizes} for m in MODELS}

    for n in sizes:
        for s in range(n_seeds):
            # which subjects make up this training set (seeded by size+seed)
            rng = np.random.default_rng(10_000 * n + s)
            train_subj = set(rng.choice(pool, size=n, replace=False).tolist())
            train_t = [t for t in trials if t.subject_id in train_subj]

            train_ds = AchillesSequenceDataset(build_samples(train_t, load_model))
            test_ds = AchillesSequenceDataset(
                test_samples_template, train_ds.feat_mean, train_ds.feat_std)
            Xte = np.stack([test_ds[i]["x"].numpy() for i in range(len(test_ds))]).astype(np.float32)

            # the two CNNs share seed -> identical init and batch order; only the loss differs
            for label, w in ((PHYSICS, _PHYSICS_W), (DATA_ONLY, _DATA_ONLY_W)):
                tr = Trainer(train_ds, test_ds, TrainConfig(epochs=epochs, weights=w, seed=s))
                tr.train(verbose=False)
                tr.model.eval()
                with torch.no_grad():
                    pred = np.clip(tr.model(torch.from_numpy(Xte)).numpy(), 0.0, None)
                acc[label][n].append(_loaded_r2(true_curves, _curves(pred)))

            # linear baseline on the identical standardised data
            Xtr = np.stack([train_ds[i]["x"].numpy() for i in range(len(train_ds))])
            Ytr = np.stack([train_ds[i]["y"].numpy() for i in range(len(train_ds))])
            lin = RidgeSequenceModel(channels=None, label=LINEAR).fit(Xtr, Ytr)
            pred_lin = np.clip(lin.predict(Xte), 0.0, None)
            acc[LINEAR][n].append(_loaded_r2(true_curves, _curves(pred_lin)))

    mean = {m: np.array([float(np.mean(acc[m][n])) for n in sizes]) for m in MODELS}
    std = {m: np.array([float(np.std(acc[m][n])) for n in sizes]) for m in MODELS}
    return DataEfficiencyResult(list(sizes), MODELS, mean, std,
                                n_seeds, len(test_subj), epochs)
