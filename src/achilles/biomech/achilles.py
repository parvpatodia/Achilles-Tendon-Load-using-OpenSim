"""Analytical Achilles tendon load: force -> stress -> strain (Stage 1 core).

The scientific spine:
    ankle plantarflexion moment  (inverse dynamics, from the dataset)
      / Achilles moment arm        -> tendon force        (N)
      / cross-sectional area       -> tendon stress       (Pa)
      / linear-region modulus      -> tendon strain        (-)

Only the plantarflexion (positive) moment loads the Achilles; when the net
ankle moment is dorsiflexor the tendon is treated as unloaded (the dorsiflexors,
not the Achilles, carry it). This is a standard simplifying assumption.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from achilles.config import TENDON, TendonProperties
from achilles.biomech.moment_arm import AngleDependentMomentArm, MomentArmModel
from achilles.data.trial import GaitTrial


@dataclass(frozen=True)
class AchillesLoadResult:
    """Per-sample Achilles load signals over one gait cycle."""

    trial: GaitTrial
    moment_arm_m: np.ndarray
    force_n: np.ndarray
    force_bw: np.ndarray
    stress_pa: np.ndarray
    strain: np.ndarray
    moment_arm_name: str

    @property
    def peak_force_n(self) -> float:
        return float(np.max(self.force_n))

    @property
    def peak_force_bw(self) -> float:
        return float(np.max(self.force_bw))

    @property
    def peak_stress_mpa(self) -> float:
        return float(np.max(self.stress_pa) / 1e6)

    @property
    def peak_strain_pct(self) -> float:
        return float(np.max(self.strain) * 100.0)

    def loading_impulse_ns(self) -> float:
        """Integral of tendon force over the gait cycle, in N*s.

        Uses the trial period implied by stride. We integrate over normalised
        phase and scale by an assumed stride time so the value is a consistent
        relative quantity across trials (absolute value is indicative only).
        """
        # trapezoidal over phase fraction (0..1), giving N * (cycle fraction).
        frac = self.trial.gait_phase / 100.0
        trapz = getattr(np, "trapezoid", np.trapz)  # np 2.x renamed trapz
        return float(trapz(self.force_n, frac))


class AchillesLoadModel:
    """Compute Achilles tendon load from a gait trial.

    Single responsibility: turn (ankle moment, ankle angle) into tendon
    force/stress/strain given a moment-arm strategy and tendon properties.
    """

    def __init__(
        self,
        moment_arm: MomentArmModel | None = None,
        tendon: TendonProperties = TENDON,
    ):
        self.moment_arm = moment_arm or AngleDependentMomentArm()
        self.tendon = tendon

    def compute(self, trial: GaitTrial) -> AchillesLoadResult:
        r = self.moment_arm.moment_arm_m(trial.ankle_angle_deg)  # (m)

        # Only the plantarflexion (positive) moment loads the Achilles.
        plantarflexion_moment_nm = np.clip(trial.ankle_moment_nm, 0.0, None)
        force_n = plantarflexion_moment_nm / r  # F = M / r

        stress_pa = force_n / self.tendon.csa_m2
        strain = stress_pa / self.tendon.modulus_pa  # linear-region estimate

        return AchillesLoadResult(
            trial=trial,
            moment_arm_m=r,
            force_n=force_n,
            force_bw=force_n / trial.body_weight_n,
            stress_pa=stress_pa,
            strain=strain,
            moment_arm_name=self.moment_arm.name,
        )
