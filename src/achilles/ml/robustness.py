"""Input-degradation experiment: how the surrogate behaves on real-insole-grade
signals, not pristine lab inputs.

We train on clean lab data and then test on progressively corrupted inputs
(sensor noise, lost temporal resolution, low-bit quantisation). This is the
honest answer to "your R^2 is a clean-data artifact": it measures the
lab->field domain shift and shows where accuracy actually lands once the signal
looks like an insole's. Retraining on matched noise would recover some of this;
we report the harder train-clean/test-degraded case on purpose.
"""
from __future__ import annotations

import numpy as np
import torch

from achilles.biomech.achilles import AchillesLoadModel
from achilles.ml.dataset import AchillesSequenceDataset, build_samples
from achilles.ml.evaluation import evaluate_predictions
from achilles.ml.trainer import Trainer, TrainConfig


def degrade(x_raw: np.ndarray, kind: str, level: float,
            ch_std: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Corrupt one (C, T) raw feature array. ch_std is per-channel std (for noise)."""
    x = x_raw.copy()
    T = x.shape[1]
    if kind == "noise":                     # sensor noise: level = fraction of channel std
        x = x + rng.normal(0, 1, x.shape) * (level * ch_std[:, None])
    elif kind == "downsample":              # lose temporal resolution: keep every f-th sample
        f = max(int(level), 1)
        idx = np.arange(0, T, f)
        for c in range(x.shape[0]):
            x[c] = np.interp(np.arange(T), idx, x[c, idx])
    elif kind == "quantize":                # low-bit ADC: level = bits
        bits = max(int(level), 1)
        for c in range(x.shape[0]):
            lo, hi = x[c].min(), x[c].max()
            if hi > lo:
                q = np.round((x[c] - lo) / (hi - lo) * (2**bits - 1))
                x[c] = lo + q / (2**bits - 1) * (hi - lo)
    else:
        raise ValueError(f"unknown degradation {kind!r}")
    return x


def run_degradation(trials, schedule: dict[str, list[float]], k: int = 5,
                    epochs: int = 200, seed: int = 0) -> dict:
    """Train clean per fold; evaluate on degraded test inputs.

    schedule maps kind -> list of levels (the first level should be the clean
    baseline, e.g. noise 0.0, downsample 1, quantize a high bit-depth).
    Returns {kind: {"levels": [...], "r2_loaded": [...], "peak_mape": [...]}}.
    """
    load_model = AchillesLoadModel()
    subjects = sorted({t.subject_id for t in trials})
    rng_fold = np.random.default_rng(seed)
    rng_fold.shuffle(subjects)
    folds = [list(f) for f in np.array_split(subjects, k)]

    # accumulate held-out preds per (kind, level)
    acc: dict = {kind: {lvl: {"sid": [], "true": [], "pred": []} for lvl in lvls}
                 for kind, lvls in schedule.items()}

    for test_subj in folds:
        ts = set(test_subj)
        train_t = [t for t in trials if t.subject_id not in ts]
        test_t = [t for t in trials if t.subject_id in ts]
        train_ds = AchillesSequenceDataset(build_samples(train_t, load_model))
        test_ds = AchillesSequenceDataset(build_samples(test_t, load_model),
                                          train_ds.feat_mean, train_ds.feat_std)
        cfg = TrainConfig(epochs=epochs, seed=seed)
        trainer = Trainer(train_ds, test_ds, cfg)
        trainer.train(verbose=False)
        trainer.model.eval()

        fmean, fstd = train_ds.feat_mean, train_ds.feat_std
        ch_std = (np.stack([s.x for s in train_ds.samples]).std(axis=(0, 2)))
        rng = np.random.default_rng(seed)
        for kind, lvls in schedule.items():
            for lvl in lvls:
                X = []
                for s in test_ds.samples:
                    xd = degrade(s.x, kind, lvl, ch_std, rng)
                    X.append((xd - fmean) / fstd)
                X = np.stack(X).astype(np.float32)
                with torch.no_grad():
                    pred = np.clip(trainer.model(torch.from_numpy(X)).numpy(), 0, None)
                a = acc[kind][lvl]
                a["sid"].extend([s.subject_id for s in test_ds.samples])
                a["true"].extend([s.y_bw for s in test_ds.samples])
                a["pred"].extend(list(pred))

    out: dict = {}
    for kind, lvls in schedule.items():
        out[kind] = {"levels": [], "r2_loaded": [], "peak_mape": []}
        for lvl in lvls:
            a = acc[kind][lvl]
            m = evaluate_predictions(a["sid"], a["true"], a["pred"], n_boot=1)
            out[kind]["levels"].append(lvl)
            out[kind]["r2_loaded"].append(m.r2_loaded)
            out[kind]["peak_mape"].append(m.peak_mape_pct)
    return out
