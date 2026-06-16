"""Torch dataset and subject-wise splitting for the Achilles surrogate.

Subject-wise split is the honest generalisation test: every trial from a given
subject goes entirely to train or entirely to test, so the model is scored on
people it has never seen. This mirrors the covariate-shift discipline that
matters for a wearable that will meet new athletes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from achilles.biomech.achilles import AchillesLoadModel
from achilles.data.trial import GaitTrial
from achilles.ml.features import build_features


@dataclass
class TrialSample:
    x: np.ndarray            # (C, T) wearable features
    y_bw: np.ndarray         # (T,) Achilles force in body weights (target)
    moment_nm: np.ndarray    # (T,) measured plantarflexion moment (Nm)
    moment_arm_m: np.ndarray  # (T,) moment arm (m)
    body_weight_n: float
    subject_id: str


class AchillesSequenceDataset(Dataset):
    """Wraps a list of TrialSample, applying input standardisation."""

    def __init__(self, samples: list[TrialSample], feat_mean=None, feat_std=None,
                 noise_std: float = 0.0):
        self.samples = samples
        self.noise_std = float(noise_std)  # simulated sensor noise (std-units)
        if feat_mean is None or feat_std is None:
            stacked = np.stack([s.x for s in samples])          # (N, C, T)
            feat_mean = stacked.mean(axis=(0, 2), keepdims=True)[0]  # (C,1)
            feat_std = stacked.std(axis=(0, 2), keepdims=True)[0] + 1e-6
        self.feat_mean = feat_mean.astype(np.float32)
        self.feat_std = feat_std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        x = (s.x - self.feat_mean) / self.feat_std
        if self.noise_std > 0:
            # deterministic per-item sensor noise (reproducible eval)
            rng = np.random.default_rng(1000 + i)
            x = x + rng.normal(0, self.noise_std, x.shape).astype(np.float32)
        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "y": torch.from_numpy(s.y_bw.astype(np.float32)),
            "moment_nm": torch.from_numpy(s.moment_nm.astype(np.float32)),
            "moment_arm_m": torch.from_numpy(s.moment_arm_m.astype(np.float32)),
            "bw": torch.tensor(s.body_weight_n, dtype=torch.float32),
        }


def build_samples(trials: list[GaitTrial], load_model: AchillesLoadModel) -> list[TrialSample]:
    samples = []
    for t in trials:
        res = load_model.compute(t)
        samples.append(TrialSample(
            x=build_features(t),
            y_bw=res.force_bw,
            moment_nm=np.clip(t.ankle_moment_nm, 0.0, None),
            moment_arm_m=res.moment_arm_m,
            body_weight_n=t.body_weight_n,
            subject_id=t.subject_id,
        ))
    return samples


def subject_wise_split(trials: list[GaitTrial], test_frac: float = 0.25, seed: int = 0):
    """Split trials into (train, test) with no subject appearing in both."""
    subjects = sorted({t.subject_id for t in trials})
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_test = max(1, round(len(subjects) * test_frac))
    test_subj = set(subjects[:n_test])
    train = [t for t in trials if t.subject_id not in test_subj]
    test = [t for t in trials if t.subject_id in test_subj]
    return train, test, sorted(test_subj)
