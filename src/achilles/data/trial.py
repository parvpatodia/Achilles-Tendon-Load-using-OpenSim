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
    # Vertical GRF is optional: Stage 1 (force/stress/strain) does not need it,
    # and some sources (the walking set) provide normalised joint moments but
    # not normalised GRF. The PINN (Stage 2) uses only trials that have it.
    vgrf_n_per_kg: np.ndarray | None = None
    source: str = "unknown"  # provenance tag, e.g. "fukuchi" or "synthetic"
    # Subject-specific scale on the Achilles moment arm (a length, so it scales
    # with body size). 1.0 = cohort-average build; set from height by the data
    # source. Lets each athlete have their own lever instead of one fixed value.
    moment_arm_scale: float = 1.0
    task: str = "run"   # "run" or "walk"

    def __post_init__(self) -> None:
        n = len(self.gait_phase)
        names = ["ankle_angle_deg", "ankle_moment_nm_per_kg"]
        if self.vgrf_n_per_kg is not None:
            names.append("vgrf_n_per_kg")
        for name in names:
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name} length {len(arr)} != gait_phase length {n}")
        if self.side not in ("R", "L"):
            raise ValueError(f"side must be 'R' or 'L', got {self.side!r}")
        if self.body_mass_kg <= 0:
            raise ValueError(f"body_mass_kg must be > 0, got {self.body_mass_kg}")

    @property
    def has_grf(self) -> bool:
        return self.vgrf_n_per_kg is not None

    # Absolute (de-normalised) signals -------------------------------------
    @property
    def ankle_moment_nm(self) -> np.ndarray:
        """Plantarflexion moment in Nm (mass-normalised value x body mass)."""
        return self.ankle_moment_nm_per_kg * self.body_mass_kg

    @property
    def vgrf_n(self) -> np.ndarray:
        """Vertical GRF in newtons (None if this source has no GRF)."""
        return None if self.vgrf_n_per_kg is None else self.vgrf_n_per_kg * self.body_mass_kg

    @property
    def vgrf_bw(self) -> np.ndarray:
        """Vertical GRF in body weights (None if this source has no GRF)."""
        return None if self.vgrf_n_per_kg is None else self.vgrf_n_per_kg / GRAVITY_M_S2

    @property
    def body_weight_n(self) -> float:
        return self.body_mass_kg * GRAVITY_M_S2

    @property
    def trial_id(self) -> str:
        return f"{self.subject_id}_{self.side}_{self.speed_ms:g}"
