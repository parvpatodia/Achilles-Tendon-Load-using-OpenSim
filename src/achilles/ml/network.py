"""AchillesSurrogate: a small 1D-CNN that maps a wearable signal to the Achilles
force waveform (sequence in -> sequence out).

A temporal CNN suits this problem: the mapping from plantar load + ankle angle
to tendon force is local in time (loading at phase p depends on the signal near
p), so modest receptive fields with shared weights generalise across subjects
with few parameters. Output is left unbounded so the non-negativity physics loss
is an active constraint rather than a no-op.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AchillesSurrogate(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32, kernel: int = 7,
                 depth: int = 3):
        super().__init__()
        pad = kernel // 2  # 'same' length
        layers: list[nn.Module] = []
        c_in = in_channels
        for _ in range(depth):
            layers += [
                nn.Conv1d(c_in, hidden, kernel, padding=pad),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
            ]
            c_in = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, T)
        h = self.backbone(x)
        return self.head(h).squeeze(1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
