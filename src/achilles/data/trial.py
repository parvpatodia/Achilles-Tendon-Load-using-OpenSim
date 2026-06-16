"""GaitTrial: the value object every downstream stage consumes.

A trial is one (subject, side, speed) recording, time-normalised to the gait
cycle. Biomechanics, the ML surrogate, and the product views all depend on
this small interface, never on a specific dataset's file format.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from achilles.config import GRAVITY_M_S2


@dataclass(frozen=True)
class GaitTrial:
    """One time-normalised gait cycle for one limb at one speed.

    Arrays are all length n_samples (101) over 0-100% of the gait cycle.
    Signs follow the dataset convention:
      ankle_angle_deg          + dorsiflexion
      ankle_moment_nm_per_kg   + plantarflexion (the Achilles-generated moment)
      vgrf_n_per_kg            vertical ground reaction force, >= 0
    """

    subject_id: str
    side: str            # "R" or "L"
    speed_ms: float
    body_mass_kg: float
    gait_phase: np.ndarray
    ankle_angle_deg: np.ndarray
    ankle_moment_nm_per_kg: np.ndarray
    vgrf_n_per_kg: np.ndarray
    source: str = "unknown"  # provenance tag, e.g. "fukuchi" or "synthetic"

    def __post_init__(self) -> None:
        n = len(self.gait_phase)
        for name in ("ankle_angle_deg", "ankle_moment_nm_per_kg", "vgrf_n_per_kg"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name} length {len(arr)} != gait_phase length {n}")
        if self.side not in ("R", "L"):
            raise ValueError(f"side must be 'R' or 'L', got {self.side!r}")
        if self.body_mass_kg <= 0:
            raise ValueError(f"body_mass_kg must be > 0, got {self.body_mass_kg}")

    # Absolute (de-normalised) signals -------------------------------------
    @property
    def ankle_moment_nm(self) -> np.ndarray:
        """Plantarflexion moment in Nm (mass-normalised value x body mass)."""
        return self.ankle_moment_nm_per_kg * self.body_mass_kg

    @property
    def vgrf_n(self) -> np.ndarray:
        """Vertical GRF in newtons."""
        return self.vgrf_n_per_kg * self.body_mass_kg

    @property
    def vgrf_bw(self) -> np.ndarray:
        """Vertical GRF in body weights (dimensionless)."""
        return self.vgrf_n_per_kg / GRAVITY_M_S2

    @property
    def body_weight_n(self) -> float:
        return self.body_mass_kg * GRAVITY_M_S2

    @property
    def trial_id(self) -> str:
        return f"{self.subject_id}_{self.side}_{self.speed_ms:g}"
