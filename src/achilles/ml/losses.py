"""Physics-informed loss for the Achilles surrogate.

Total loss = data fit + physical priors. Each term is documented with its
physical meaning:

  data        MSE on the Achilles force waveform (body weights). The supervision.
  non_neg     penalise negative tendon force. A tendon can only pull, never
              push, so F >= 0 is a hard physical fact.
  moment      enforce the biomechanical identity F * r = M_plantarflexion in
              joint-moment units (Nm). Anchors the output to the measured
              inverse-dynamics moment independent of the BW normalisation.
  smooth      penalise the second time-difference of the force. Tendon force
              cannot change arbitrarily fast (bounded loading rate); this is a
              physically motivated temporal regulariser.

Weights are configurable so the contribution of the physics terms can be
ablated (train with weights=0 to recover a plain data-driven baseline).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    data: float = 1.0
    non_neg: float = 0.5
    moment: float = 0.2
    smooth: float = 0.05


class PhysicsInformedLoss:
    def __init__(self, weights: LossWeights | None = None):
        self.w = weights or LossWeights()

    def __call__(self, pred_bw, batch) -> tuple[torch.Tensor, dict]:
        y = batch["y"]                      # (B, T) BW
        bw = batch["bw"].unsqueeze(1)       # (B, 1) N
        r = batch["moment_arm_m"]           # (B, T) m
        m = batch["moment_nm"]              # (B, T) Nm

        # data fit
        l_data = F.mse_loss(pred_bw, y)

        # non-negativity (in newtons so it is in physical units)
        pred_n = pred_bw * bw
        l_nonneg = torch.relu(-pred_n).mean() / 1000.0  # scale to ~O(1)

        # moment consistency: F * r should equal the measured plantarflexion moment
        moment_pred = pred_n * r
        l_moment = F.mse_loss(moment_pred, m) / (100.0 ** 2)  # scale Nm^2 to ~O(1)

        # smoothness: second difference of the force waveform
        d2 = pred_bw[:, 2:] - 2 * pred_bw[:, 1:-1] + pred_bw[:, :-2]
        l_smooth = (d2 ** 2).mean()

        total = (self.w.data * l_data + self.w.non_neg * l_nonneg
                 + self.w.moment * l_moment + self.w.smooth * l_smooth)
        parts = {"data": float(l_data), "non_neg": float(l_nonneg),
                 "moment": float(l_moment), "smooth": float(l_smooth),
                 "total": float(total)}
        return total, parts
