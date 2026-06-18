"""Reference models the surrogate must beat to justify its complexity.

A high R^2 means nothing without a baseline. These give the panel the honest
context: if a linear model already scores 0.97, the CNN's extra capacity is not
worth it. All share one interface so the cross-validation harness scores them on
identical subject-wise folds.

  MeanCurveModel   predicts the average training force curve for everyone
                   (the "no-skill" floor; R^2 ~ 0 by construction off-cohort).
  RidgeSequenceModel  a linear map from the (optionally channel-subset) input
                   sequence to the output sequence. The real "do you need a
                   neural net?" baseline, and the vehicle for input ablations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import Ridge


class SequenceModel(ABC):
    """Maps a batch of input sequences (N, C, T) to outputs (N, T)."""

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "SequenceModel":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class MeanCurveModel(SequenceModel):
    def __init__(self):
        self._mean_curve: np.ndarray | None = None

    def fit(self, X, Y):
        self._mean_curve = np.asarray(Y).mean(axis=0)
        return self

    def predict(self, X):
        return np.tile(self._mean_curve, (len(X), 1))

    @property
    def name(self) -> str:
        return "mean-curve (no-skill floor)"


class RidgeSequenceModel(SequenceModel):
    """Multi-output ridge regression on the flattened input sequence.

    channels=None uses all input channels; passing a subset (e.g. only the GRF
    channel) turns this into a clean input-ablation baseline.
    """

    def __init__(self, alpha: float = 10.0, channels: tuple[int, ...] | None = None,
                 label: str = "ridge (linear, all inputs)"):
        self.alpha = alpha
        self.channels = channels
        self._label = label
        self._model = Ridge(alpha=alpha)

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if self.channels is not None:
            X = X[:, self.channels, :]
        return X.reshape(len(X), -1)

    def fit(self, X, Y):
        self._model.fit(self._flatten(X), np.asarray(Y))
        return self

    def predict(self, X):
        return self._model.predict(self._flatten(X))

    @property
    def name(self) -> str:
        return self._label
