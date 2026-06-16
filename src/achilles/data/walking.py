"""WalkingDataSource: the Fukuchi et al. 2018 treadmill-walking dataset.

REF: Fukuchi CA, Fukuchi RK, Duarte M (2018). "A public dataset of overground
and treadmill walking kinematics and kinetics in healthy individuals."
PeerJ 6:e4640. Data: figshare 10.6084/m9.figshare.5722711.

Why it matters here: Mirai's published insole work is on WALKING in
rehabilitation patients, not running. This dataset (42 adults, young and older,
treadmill walking across 8 speeds) lets the same pipeline run on the gait mode
that matches her cohort.

Format note: the per-trial angle ('ang') and kinetics ('knt') files are already
normalised to 101-point gait cycles (same convention as the running set:
sagittal ankle = the Z axis, moments in Nm/kg). The GRF is only available raw,
so walking trials carry no vertical GRF and are used for Stage 1 analysis, not
the Stage 2 surrogate (which needs a wearable GRF input).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from achilles.config import DATA_RAW
from achilles.data.base import GaitDataSource
from achilles.data.trial import GaitTrial

_TRIAL_RE = re.compile(r"(WBDS\d+walkT\d+)ang\.txt$")

# Plausible walking ankle plantarflexion moment (Nm/kg): peak ~1.0-1.8.
_MIN_WALK_MOMENT = 0.5
_MAX_WALK_MOMENT = 2.5


class WalkingDataSource(GaitDataSource):
    def __init__(self, data_dir: Path | str = DATA_RAW / "wbds",
                 info_path: Path | str = DATA_RAW / "WBDSinfo.xlsx",
                 sides: tuple[str, ...] = ("R", "L")):
        self.data_dir = Path(data_dir)
        self.sides = sides
        self._mass, self._height, self._speed = self._load_meta(Path(info_path))
        heights = [h for h in self._height.values() if h > 0]
        self._mean_height_cm = float(np.mean(heights)) if heights else 0.0

    @staticmethod
    def _load_meta(info_path: Path):
        """Per-subject mass/height and per-trial gait speed from WBDSinfo.xlsx."""
        import warnings
        with warnings.catch_warnings():  # openpyxl notes a harmless xlsx extension
            warnings.simplefilter("ignore")
            info = pd.read_excel(info_path)
        mass, height, speed = {}, {}, {}
        for _, row in info.iterrows():
            subj = f"WBDS{int(row['Subject']):02d}"
            mass.setdefault(subj, float(row["Mass"]))
            height.setdefault(subj, float(row["Height"]))
            stem = str(row["FileName"]).rsplit(".", 1)[0]  # e.g. WBDS01walkT01
            try:
                speed[stem] = float(row["GaitSpeed(m/s)"])
            except (ValueError, TypeError):
                pass
        return mass, height, speed

    def _moment_arm_scale(self, subject: str) -> float:
        h = self._height.get(subject, 0.0)
        if h <= 0 or self._mean_height_cm <= 0:
            return 1.0
        return h / self._mean_height_cm

    def iter_trials(self) -> Iterator[GaitTrial]:
        for ang_path in sorted(self.data_dir.glob("WBDS*walkT*ang.txt")):
            m = _TRIAL_RE.search(ang_path.name)
            if not m:
                continue
            stem = m.group(1)                         # WBDS01walkT01
            subject = stem.split("walk")[0]           # WBDS01
            knt_path = ang_path.with_name(stem + "knt.txt")
            if not knt_path.exists():
                continue
            mass = self._mass.get(subject)
            speed = self._speed.get(stem)
            if mass is None or speed is None:
                continue
            ang = pd.read_csv(ang_path, sep="\t")
            knt = pd.read_csv(knt_path, sep="\t")
            n = len(ang)
            phase = np.linspace(0.0, 100.0, n)
            for side in self.sides:
                ang_col, mom_col = f"{side}AnkleAngleZ", f"{side}AnkleMomentZ"
                if ang_col not in ang.columns or mom_col not in knt.columns:
                    continue
                moment = np.nan_to_num(knt[mom_col].to_numpy(dtype=float), nan=0.0)
                if not (_MIN_WALK_MOMENT <= float(np.max(moment)) <= _MAX_WALK_MOMENT):
                    continue  # QC: reject non-physiological walking moments
                yield GaitTrial(
                    subject_id=subject,
                    side=side,
                    speed_ms=speed,
                    body_mass_kg=mass,
                    gait_phase=phase,
                    ankle_angle_deg=np.nan_to_num(ang[ang_col].to_numpy(dtype=float)),
                    ankle_moment_nm_per_kg=moment,
                    vgrf_n_per_kg=None,           # no normalised GRF for walking
                    source="fukuchi-walking",
                    moment_arm_scale=self._moment_arm_scale(subject),
                    task="walk",
                )
