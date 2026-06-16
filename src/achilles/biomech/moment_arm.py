"""Achilles tendon moment arm models (strategy pattern).

The moment arm r is the lever converting ankle plantarflexion moment to tendon
force: F_achilles = M_plantarflexion / r. Two interchangeable strategies:

  ConstantMomentArm       - fixed r (simple, transparent).
  AngleDependentMomentArm - r varies with ankle angle (closer to in-vivo).

Downstream code depends on the MomentArmModel interface, so swapping the
assumption is a one-line change and its effect on the result is auditable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from achilles.config import MOMENT_ARM, MomentArmParams


class MomentArmModel(ABC):
    @abstractmethod
    def moment_arm_m(self, ankle_angle_deg: np.ndarray) -> np.ndarray:
        """Return moment arm (m) for each ankle angle sample (+ dorsiflexion)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class ConstantMomentArm(MomentArmModel):
    """Fixed moment arm regardless of ankle angle.

    REF: Rugg et al. 1990; Maganaris et al. 2000 (adult Achilles ~4-6 cm).
    """

    def __init__(self, r_m: float = MOMENT_ARM.constant_m):
        self.r_m = float(r_m)

    def moment_arm_m(self, ankle_angle_deg: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(ankle_angle_deg, dtype=float), self.r_m)

    @property
    def name(self) -> str:
        return f"constant ({self.r_m * 100:.1f} cm)"


class AngleDependentMomentArm(MomentArmModel):
    """Quadratic r(theta) clamped to a physiological range.

    r(theta) = r0 + c1*theta + c2*theta^2, theta in degrees (+ dorsiflexion).
    The trend (smaller moment arm toward extreme dorsiflexion) follows in-vivo
    measurements. REF: Maganaris et al. 2000; Rugg et al. 1990. This is an
    approximation, not a subject-specific measurement.
    """

    def __init__(self, params: MomentArmParams = MOMENT_ARM):
        self.p = params

    def moment_arm_m(self, ankle_angle_deg: np.ndarray) -> np.ndarray:
        theta = np.asarray(ankle_angle_deg, dtype=float)
        r = self.p.r0_m + self.p.c1_m_per_deg * theta + self.p.c2_m_per_deg2 * theta**2
        return np.clip(r, self.p.min_m, self.p.max_m)

    @property
    def name(self) -> str:
        return "angle-dependent (Maganaris-style)"
