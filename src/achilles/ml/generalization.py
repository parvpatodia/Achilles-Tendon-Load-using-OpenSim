"""Load-regime generalization: does the surrogate work at a running speed it
never trained on?

The running set has three speeds (different load regimes). Leave-one-speed-out
trains on two speeds and tests on the held-out one, so it measures whether the
signal-to-force mapping extrapolates to a load magnitude it has not seen. The
lowest speed is the extrapolation toward lower, walking-like loads, the most
relevant cell for a walking rehab cohort (Mirai's near-term users), since the
surrogate itself cannot be tested on the public walking set (no time-normalised
GRF, the wearable input it needs).

Honest scope: subjects appear at every speed, so this isolates load-regime shift
and does NOT also hold subjects out (subject-wise generalization is the separate
k-fold result). It is encouraging for, not proof of, transfer to real walking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from achilles.biomech.achilles import AchillesLoadModel
from achilles.data.trial import GaitTrial
from achilles.ml.baselines import RidgeSequenceModel, SequenceModel
from achilles.ml.dataset import AchillesSequenceDataset, build_samples
from achilles.ml.evaluation import LOADED_THRESHOLD_BW, r2_score


@dataclass
class SpeedGeneralizationResult:
    speeds: list[float]
    loaded_r2: dict[float, float] = field(default_factory=dict)
    peak_mape_pct: dict[float, float] = field(default_factory=dict)
    peak_bias_bw: dict[float, float] = field(default_factory=dict)
    n_test: dict[float, int] = field(default_factory=dict)

    def table(self) -> str:
        lines = [f"{'held-out speed':>15s} {'loaded R2':>10s} {'peak MAPE':>10s} "
                 f"{'peak bias':>10s} {'n_test':>7s}"]
        lo = min(self.speeds)
        for s in self.speeds:
            tag = "  <- toward walking loads" if s == lo else ""
            lines.append(f"{s:>13.1f}   {self.loaded_r2[s]:>10.3f} "
                         f"{self.peak_mape_pct[s]:>9.1f}% {self.peak_bias_bw[s]:>+9.2f}BW "
                         f"{self.n_test[s]:>7d}{tag}")
        return "\n".join(lines)


def _loaded_r2(true: list[np.ndarray], pred: list[np.ndarray]) -> float:
    t = np.concatenate([c.ravel() for c in true]); p = np.concatenate([c.ravel() for c in pred])
    m = t > LOADED_THRESHOLD_BW
    return r2_score(t[m], p[m]) if m.any() else float("nan")


def leave_one_speed_out(
    trials: list[GaitTrial],
    model_factory: Callable[[], SequenceModel] = lambda: RidgeSequenceModel(channels=None),
) -> SpeedGeneralizationResult:
    load_model = AchillesLoadModel()
    speeds = sorted({t.speed_ms for t in trials})
    res = SpeedGeneralizationResult(speeds=speeds)

    for held in speeds:
        train_t = [t for t in trials if t.speed_ms != held]
        test_t = [t for t in trials if t.speed_ms == held]
        tr = AchillesSequenceDataset(build_samples(train_t, load_model))
        te = AchillesSequenceDataset(build_samples(test_t, load_model), tr.feat_mean, tr.feat_std)
        Xtr = np.stack([tr[i]["x"].numpy() for i in range(len(tr))])
        Ytr = np.stack([tr[i]["y"].numpy() for i in range(len(tr))])
        Xte = np.stack([te[i]["x"].numpy() for i in range(len(te))])
        true = [s.y_bw for s in te.samples]

        model = model_factory().fit(Xtr, Ytr)
        pred = np.clip(model.predict(Xte), 0.0, None)
        pred_curves = [pred[i] for i in range(len(pred))]

        peak_true = np.array([c.max() for c in true])
        peak_pred = np.array([c.max() for c in pred_curves])
        res.loaded_r2[held] = _loaded_r2(true, pred_curves)
        res.peak_mape_pct[held] = float(np.mean(np.abs(peak_pred - peak_true) / peak_true) * 100)
        res.peak_bias_bw[held] = float(np.mean(peak_pred - peak_true))
        res.n_test[held] = len(test_t)
    return res
