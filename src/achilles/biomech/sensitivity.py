"""Moment-arm sensitivity analysis.

The Achilles moment arm is the single assumption the whole estimate leans on
(force = moment / arm), and the literature spread is wide (~4-6 cm). Rather than
hide behind one fixed value, this sweeps the moment arm across that range and
reports how much the peak tendon force and stress move. It turns "we assumed
5 cm" into "here is exactly how sensitive the answer is, and where a validated
model and per-athlete measurement would pin it down".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from achilles.biomech.achilles import AchillesLoadModel
from achilles.biomech.moment_arm import ConstantMomentArm
from achilles.config import TENDON, TendonProperties
from achilles.data.trial import GaitTrial


@dataclass
class SensitivityResult:
    arms_cm: np.ndarray
    peak_force_bw_mean: np.ndarray
    peak_force_bw_sd: np.ndarray
    peak_stress_mpa_mean: np.ndarray
    peak_stress_mpa_sd: np.ndarray

    def force_swing_pct(self) -> float:
        """How much peak force changes across the swept arm range, as a % of the
        mid-range value (the headline sensitivity number)."""
        lo, hi = self.peak_force_bw_mean[0], self.peak_force_bw_mean[-1]
        mid = self.peak_force_bw_mean[len(self.peak_force_bw_mean) // 2]
        return float(abs(hi - lo) / mid * 100.0)


def moment_arm_sensitivity(
    trials: list[GaitTrial],
    arms_cm: np.ndarray | None = None,
    tendon: TendonProperties = TENDON,
) -> SensitivityResult:
    arms_cm = np.linspace(4.0, 6.0, 11) if arms_cm is None else np.asarray(arms_cm)
    pf_mean, pf_sd, ps_mean, ps_sd = [], [], [], []
    for r_cm in arms_cm:
        model = AchillesLoadModel(moment_arm=ConstantMomentArm(r_cm / 100.0),
                                  tendon=tendon)
        res = [model.compute(t) for t in trials]
        pf = np.array([x.peak_force_bw for x in res])
        ps = np.array([x.peak_stress_mpa for x in res])
        pf_mean.append(pf.mean()); pf_sd.append(pf.std())
        ps_mean.append(ps.mean()); ps_sd.append(ps.std())
    return SensitivityResult(
        arms_cm=arms_cm,
        peak_force_bw_mean=np.array(pf_mean), peak_force_bw_sd=np.array(pf_sd),
        peak_stress_mpa_mean=np.array(ps_mean), peak_stress_mpa_sd=np.array(ps_sd),
    )
