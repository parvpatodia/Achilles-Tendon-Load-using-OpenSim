"""Trainer for the Achilles surrogate: fit, evaluate on held-out subjects."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from achilles.ml.dataset import AchillesSequenceDataset
from achilles.ml.losses import LossWeights, PhysicsInformedLoss
from achilles.ml.network import AchillesSurrogate


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 32
    lr: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 0
    weights: LossWeights = field(default_factory=LossWeights)


@dataclass
class EvalResult:
    r2: float
    rmse_bw: float
    mae_bw: float
    phase: np.ndarray
    true_curves: list[np.ndarray]
    pred_curves: list[np.ndarray]
    test_subjects: list[str]


class Trainer:
    def __init__(self, train_ds: AchillesSequenceDataset,
                 test_ds: AchillesSequenceDataset,
                 cfg: TrainConfig | None = None):
        self.cfg = cfg or TrainConfig()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.train_ds = train_ds
        self.test_ds = test_ds
        in_ch = train_ds.samples[0].x.shape[0]
        self.model = AchillesSurrogate(in_channels=in_ch)
        self.loss_fn = PhysicsInformedLoss(self.cfg.weights)

    def train(self, verbose: bool = True) -> list[dict]:
        loader = DataLoader(self.train_ds, batch_size=self.cfg.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr,
                               weight_decay=self.cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.cfg.epochs)
        history = []
        for ep in range(self.cfg.epochs):
            self.model.train()
            agg = {}
            for batch in loader:
                opt.zero_grad()
                pred = self.model(batch["x"])
                loss, parts = self.loss_fn(pred, batch)
                loss.backward()
                opt.step()
                for k, v in parts.items():
                    agg[k] = agg.get(k, 0.0) + v
            sched.step()
            agg = {k: v / len(loader) for k, v in agg.items()}
            history.append(agg)
            if verbose and (ep % 25 == 0 or ep == self.cfg.epochs - 1):
                print(f"  epoch {ep:3d}  total={agg['total']:.4f}  "
                      f"data={agg['data']:.4f}  moment={agg['moment']:.4f}")
        return history

    @torch.no_grad()
    def evaluate(self) -> EvalResult:
        self.model.eval()
        loader = DataLoader(self.test_ds, batch_size=len(self.test_ds))
        batch = next(iter(loader))
        pred = self.model(batch["x"]).numpy()
        true = batch["y"].numpy()

        # clamp predictions at 0 for reporting (tendon force >= 0)
        pred = np.clip(pred, 0.0, None)

        t_flat, p_flat = true.ravel(), pred.ravel()
        ss_res = float(np.sum((t_flat - p_flat) ** 2))
        ss_tot = float(np.sum((t_flat - t_flat.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        rmse = float(np.sqrt(np.mean((t_flat - p_flat) ** 2)))
        mae = float(np.mean(np.abs(t_flat - p_flat)))

        phase = np.linspace(0, 100, true.shape[1])
        subj = [s.subject_id for s in self.test_ds.samples]
        return EvalResult(
            r2=r2, rmse_bw=rmse, mae_bw=mae, phase=phase,
            true_curves=[true[i] for i in range(len(true))],
            pred_curves=[pred[i] for i in range(len(pred))],
            test_subjects=sorted(set(subj)),
        )
