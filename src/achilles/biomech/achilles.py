"""Analytical Achilles tendon load: force -> stress -> strain (Stage 1 core).

The scientific spine:
    ankle plantarflexion moment  (inverse dynamics, from the dataset)
      / Achilles moment arm        -> tendon force        (N)
      / cross-sectional area       -> tendon stress       (Pa)
      via tendon constitutive law  -> tendon strain        (-)

Only the plantarflexion (positive) moment loads the Achilles; when the net
ankle moment is dorsiflexor the tendon is treated as unloaded (the dorsiflexors,
not the Achilles, carry it). This is a standard simplifying assumption.

The moment-arm strategy (how force is derived) and the tendon material model
(how strain is derived) are both injected, so each modelling assumption is
swappable and its effect on the result is auditable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from achilles.config import TENDON, TendonProperties
from achilles.biomech.moment_arm import AngleDependentMomentArm, MomentArmModel
from achilles.biomech.tendon import TendonMaterialModel, ToeLinearTendon
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
    material_name: str
    tendon: TendonProperties

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

    @property
    def stress_safety_factor(self) -> float:
        """Ultimate tensile stress / peak operating stress (>1 means a margin)."""
        peak = np.max(self.stress_pa)
        return float(self.tendon.ultimate_stress_pa / peak) if peak > 0 else float("inf")

    def cyclic_load_index(self) -> float:
        """Per-stride loading exposure: tendon force integrated over the gait cycle.

        Integrated over the normalised gait cycle (phase fraction 0..1), so it is
        a RELATIVE per-cycle quantity (units N x cycle-fraction), not an absolute
        impulse in N*s. Any constant stride time cancels when this is used as a
        relative index (Stage 4 asymmetry and accumulation), which is its only
        use. Reported as relative, never as an absolute impulse.
        """
        frac = self.trial.gait_phase / 100.0
        trapz = getattr(np, "trapezoid", np.trapz)  # np 2.x renamed trapz
        return float(trapz(self.force_n, frac))


class AchillesLoadModel:
    """Compute Achilles tendon load from a gait trial.

    Single responsibility: turn (ankle moment, ankle angle) into tendon
    force/stress/strain given a moment-arm strategy, a tendon material model,
    and tendon geometry.
    """

    def __init__(
        self,
        moment_arm: MomentArmModel | None = None,
        material: TendonMaterialModel | None = None,
        tendon: TendonProperties = TENDON,
    ):
        self.moment_arm = moment_arm or AngleDependentMomentArm()
        self.material = material or ToeLinearTendon(tendon)
        self.tendon = tendon

    def compute(self, trial: GaitTrial) -> AchillesLoadResult:
        # angle-dependent shape x subject-specific anthropometric scale
        r = self.moment_arm.moment_arm_m(trial.ankle_angle_deg) * trial.moment_arm_scale

        # Only the plantarflexion (positive) moment loads the Achilles.
        plantarflexion_moment_nm = np.clip(trial.ankle_moment_nm, 0.0, None)
        force_n = plantarflexion_moment_nm / r  # F = M / r

        stress_pa = force_n / self.tendon.csa_m2
        strain = self.material.strain(stress_pa)  # non-linear constitutive law

        return AchillesLoadResult(
            trial=trial,
            moment_arm_m=r,
            force_n=force_n,
            force_bw=force_n / trial.body_weight_n,
            stress_pa=stress_pa,
            strain=strain,
            moment_arm_name=self.moment_arm.name,
            material_name=self.material.name,
            tendon=self.tendon,
        )
