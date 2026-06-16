"""SyntheticGaitSource: parametric fallback when the real dataset is missing.

Generates physiologically shaped running signals (half-sine vertical GRF scaled
to 2-3 body weights, a stance-phase plantarflexion moment, a realistic ankle
angle trace). Clearly tagged source="synthetic" so figures and the README can
label it as illustrative, not measured.

This exists so the full pipeline and PINN still demonstrate correctly if the
Figshare download fails (a stated risk in the build spec).
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from achilles.config import CONVENTIONS, GRAVITY_M_S2
from achilles.data.base import GaitDataSource
from achilles.data.trial import GaitTrial


class SyntheticGaitSource(GaitDataSource):
    def __init__(
        self,
        n_subjects: int = 20,
        speeds: tuple[float, ...] = (2.5, 3.5, 4.5),
        seed: int = 0,
    ):
        self.n_subjects = n_subjects
        self.speeds = speeds
        self.rng = np.random.default_rng(seed)
        self.n = CONVENTIONS.n_samples

    def _stance_mask(self, stance_frac: float) -> np.ndarray:
        phase = np.linspace(0.0, 1.0, self.n)
        return phase <= stance_frac, phase

    def _trial(self, subject: str, side: str, speed: float, mass: float,
               peak_bw: float, peak_mom: float, asym: float) -> GaitTrial:
        # Faster running -> shorter stance, higher peak GRF and moment.
        stance_frac = 0.40 - 0.03 * (speed - 2.5)
        in_stance, phase = self._stance_mask(stance_frac)
        s = np.clip(phase / stance_frac, 0.0, 1.0)  # 0..1 within stance

        # Vertical GRF: skewed half-sine over stance (N/kg). Scale by side asymmetry.
        vgrf_bw = peak_bw * asym * np.sin(np.pi * s) ** 1.1
        vgrf_bw[~in_stance] = 0.0
        vgrf_n_per_kg = vgrf_bw * GRAVITY_M_S2

        # Plantarflexion moment (Nm/kg): peaks slightly later than GRF (push-off).
        mom = peak_mom * asym * np.sin(np.pi * np.clip((s - 0.05) / 0.95, 0, 1)) ** 1.4
        mom[~in_stance] = 0.0

        # Ankle angle (deg, + dorsiflexion): dorsiflexes through stance to ~20 deg,
        # plantarflexes at toe-off to ~-25 deg, recovers to neutral during swing.
        toeoff_val = -25.0
        swing_phase = (phase - stance_frac) / (1 - stance_frac)
        swing = toeoff_val * (1 - swing_phase) + 5.0 * np.sin(np.pi * swing_phase)
        angle = np.where(in_stance, 22.0 * np.sin(np.pi * s) - 30.0 * s ** 3, swing)

        noise = self.rng.normal(0, 0.01, self.n)
        return GaitTrial(
            subject_id=subject,
            side=side,
            speed_ms=speed,
            body_mass_kg=mass,
            gait_phase=phase * 100.0,
            ankle_angle_deg=angle + noise * 5,
            ankle_moment_nm_per_kg=np.clip(mom + noise, 0.0, None),
            vgrf_n_per_kg=np.clip(vgrf_n_per_kg + noise * GRAVITY_M_S2, 0.0, None),
            source="synthetic",
        )

    def iter_trials(self) -> Iterator[GaitTrial]:
        for i in range(self.n_subjects):
            subject = f"SYN{i + 1:03d}"
            mass = float(self.rng.uniform(55, 85))
            # per-subject baseline peaks with mild spread
            base_bw = self.rng.uniform(2.2, 2.7)
            base_mom = self.rng.uniform(2.3, 3.0)
            # small left/right asymmetry per subject (1.0 = symmetric)
            asym = {"R": 1.0, "L": float(self.rng.uniform(0.92, 1.0))}
            for speed in self.speeds:
                speed_bw = base_bw + 0.12 * (speed - 3.5)
                speed_mom = base_mom + 0.15 * (speed - 3.5)
                for side in ("R", "L"):
                    yield self._trial(subject, side, speed, mass,
                                      speed_bw, speed_mom, asym[side])
