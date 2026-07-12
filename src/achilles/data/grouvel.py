"""GrouvelDataSource: measured-insole walking trials (Stage C).

REF: Grouvel G, Moissenet F, et al. (2023). "A dataset of asymptomatic human
gait and movements obtained from marker-based motion capture, force plates,
inertial sensors and plantar-pressure insoles." Yareta (CC-BY),
https://doi.org/10.26037/yareta:... (Moticon Sensor Insole3, 16 zones, 100 Hz).

This is the source the whole point of Stage C hangs on: every other source hands
the surrogate a *modelled* CoP proxy for the insole channels (features.py),
whereas each Grouvel trial carries the REAL 16->4 plantar-pressure split measured
under the foot, plus the Achilles target we computed ourselves with OpenSim
(scale -> IK -> ID; see scripts/grouvel_opensim.py). It reads the small,
committed per-stance CSVs under data/grouvel_processed/, so consuming it needs
neither OpenSim nor the 2.6 GB raw archive.

Each CSV is one on-plate single-limb stance, time-normalised to 101 points:
    phase, ankle_angle_deg, ankle_moment_nm, vgrf_n,
    zone_heel, zone_arch, zone_forefoot, zone_bigtoe, insole_total_n[, speed_ms]
The filename encodes provenance: ``<subject>_<trial>_p<plate>_<side>.csv``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from achilles.config import REPO_ROOT
from achilles.data.base import GaitDataSource
from achilles.data.trial import GaitTrial
from achilles.ml.features import ZONE_NAMES

PROCESSED_DIR = REPO_ROOT / "data" / "grouvel_processed"
# <subject>_<trial>_<SIDE><segment>.csv, e.g. P02_S01_Gait_01_R1.csv
_STANCE_RE = re.compile(r"^(P\d+)_.*_([RL])(\d+)\.csv$")
_ZONE_COLS = tuple(f"zone_{z}" for z in ZONE_NAMES)  # heel, arch, forefoot, bigtoe
_DEFAULT_WALK_SPEED_MS = 1.3  # self-selected overground walking, if not recorded

# Physiological QC (walking): drop stances outside these before they reach a model.
_MIN_PEAK_ACHILLES_BW = 0.5
_MAX_PEAK_ACHILLES_BW = 4.0


class GrouvelDataSource(GaitDataSource):
    def __init__(self, processed_dir: Path | str = PROCESSED_DIR):
        self.processed_dir = Path(processed_dir)
        self.stances_dir = self.processed_dir / "stances"
        self._mass, self._height = self._load_subjects(self.processed_dir / "subjects.csv")
        heights = [h for h in self._height.values() if h > 0]
        self._mean_height_mm = float(np.mean(heights)) if heights else 0.0

    @staticmethod
    def _load_subjects(path: Path) -> tuple[dict[str, float], dict[str, float]]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Grouvel demographics are committed; run "
                "scripts/grouvel_batch.py to (re)build the processed stances."
            )
        df = pd.read_csv(path)
        mass = {str(r["subject"]): float(r["body_mass_kg"]) for _, r in df.iterrows()}
        height = {str(r["subject"]): float(r["height_mm"]) for _, r in df.iterrows()}
        return mass, height

    def _moment_arm_scale(self, subject: str) -> float:
        """Height-scaled per-athlete moment-arm proxy, centred on the cohort mean.

        Same weak first-order proxy the running source uses (see fukuchi.py for
        the honest caveat); replace with a measured moment arm when available.
        """
        h = self._height.get(subject, 0.0)
        if h <= 0 or self._mean_height_mm <= 0:
            return 1.0
        return h / self._mean_height_mm

    def _stance_files(self) -> list[Path]:
        if not self.stances_dir.exists():
            return []
        return sorted(p for p in self.stances_dir.glob("P*.csv") if _STANCE_RE.match(p.name))

    def iter_trials(self) -> Iterator[GaitTrial]:
        for path in self._stance_files():
            m = _STANCE_RE.match(path.name)
            if not m:
                continue
            subject, side = m.group(1), m.group(2)
            mass = self._mass.get(subject)
            if mass is None:
                continue  # no body mass -> cannot de-normalise; skip honestly
            df = pd.read_csv(path)
            phase = df["phase"].to_numpy(dtype=float)
            moment_nm = np.clip(df["ankle_moment_nm"].to_numpy(dtype=float), 0.0, None)
            vgrf_n = np.clip(df["vgrf_n"].to_numpy(dtype=float), 0.0, None)
            zones = np.vstack([df[c].to_numpy(dtype=float) for c in _ZONE_COLS])  # (4, n)
            speed = float(df["speed_ms"].iloc[0]) if "speed_ms" in df.columns else _DEFAULT_WALK_SPEED_MS

            trial = GaitTrial(
                subject_id=subject,
                side=side,
                speed_ms=speed,
                body_mass_kg=mass,
                gait_phase=phase,
                ankle_angle_deg=df["ankle_angle_deg"].to_numpy(dtype=float),
                ankle_moment_nm_per_kg=moment_nm / mass,
                vgrf_n_per_kg=vgrf_n / mass,
                source="grouvel",
                moment_arm_scale=self._moment_arm_scale(subject),
                task="walk",
                measured_zones=zones,
            )
            if self._is_physiological(trial):
                yield trial

    @staticmethod
    def _is_physiological(trial: GaitTrial) -> bool:
        # peak Achilles ~= peak plantarflexion moment / moment arm / body weight.
        # A cheap bound using the smallest plausible arm (0.035 m) as denominator.
        peak_bw = float(np.max(trial.ankle_moment_nm)) / 0.035 / trial.body_weight_n
        return _MIN_PEAK_ACHILLES_BW <= peak_bw <= _MAX_PEAK_ACHILLES_BW
