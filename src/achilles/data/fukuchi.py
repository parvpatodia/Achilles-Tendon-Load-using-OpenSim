"""FukuchiDataSource: load the Fukuchi et al. 2017 running dataset.

REF: Fukuchi RK, Fukuchi CA, Duarte M (2017). "A public data set of running
biomechanics and the effects of running speed on lower extremity kinematics
and kinetics." PeerJ 5:e3298. Data: figshare 10.6084/m9.figshare.4543435.

We use the per-subject 'RBDS###processed.txt' files (time-normalised to 101
points over the gait cycle) and 'RBDSinfo.txt' for body mass. Column naming:
'<side><joint><quantity><axis><speed>', e.g. 'RankleMomZ35'. The sagittal axes
were identified empirically during setup (see config.GaitConventions).

Two schemas appear in the dataset: 31 subjects with all three speeds
(25/35/45) and 8 with 3.5 m/s only. We detect the speeds present per file and
emit a trial per (side, available speed), so both schemas are handled the same
way.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from achilles.config import CONVENTIONS, DATA_RAW
from achilles.data.base import GaitDataSource
from achilles.data.trial import GaitTrial

_SUBJECT_RE = re.compile(r"(RBDS\d+)processed\.txt$")


# Physiological QC bounds. Trials outside these are corrupt or a different
# encoding (the dataset's reduced-schema subjects have left-limb moments in the
# thousands, not Nm/kg) and are dropped rather than silently distorting results.
_MAX_PLAUSIBLE_MOMENT_NM_PER_KG = 8.0   # peak running ankle moment ~2-4 Nm/kg
_MIN_PLAUSIBLE_MOMENT_NM_PER_KG = 0.5
_MAX_PLAUSIBLE_VGRF_N_PER_KG = 40.0     # ~4 BW upper sanity bound


class FukuchiDataSource(GaitDataSource):
    def __init__(
        self,
        data_dir: Path | str = DATA_RAW,
        sides: tuple[str, ...] = ("R", "L"),
        require_all_speeds: bool = True,
    ):
        # require_all_speeds keeps only the 31 subjects with the complete
        # three-speed bilateral protocol (drops 8 reduced-schema subjects, some
        # of which also have inconsistent left-limb encoding).
        self.data_dir = Path(data_dir)
        self.sides = sides
        self.require_all_speeds = require_all_speeds
        self._mass_by_subject = self._load_masses(self.data_dir / "RBDSinfo.txt")

    # -- metadata -----------------------------------------------------------
    @staticmethod
    def _load_masses(info_path: Path) -> dict[str, float]:
        """Map subject id (e.g. 'RBDS001') -> body mass in kg."""
        if not info_path.exists():
            raise FileNotFoundError(
                f"{info_path} not found. Run scripts/download_data.py first."
            )
        info = pd.read_csv(info_path, sep="\t")
        masses: dict[str, float] = {}
        for _, row in info.iterrows():
            subj = f"RBDS{int(row['Subject']):03d}"
            masses[subj] = float(row["Mass"])
        return masses

    # -- discovery ----------------------------------------------------------
    def _processed_files(self) -> list[Path]:
        return sorted(self.data_dir.glob("RBDS*processed.txt"))

    @staticmethod
    def _available_speeds(columns: list[str]) -> list[str]:
        """Speed labels present, e.g. ['25','35','45'] or ['35']."""
        present = []
        for label in ("25", "35", "45"):
            if f"RankleMomZ{label}" in columns:
                present.append(label)
        return present

    # -- loading ------------------------------------------------------------
    def iter_trials(self) -> Iterator[GaitTrial]:
        ax_ang = CONVENTIONS.ankle_angle_axis
        ax_mom = CONVENTIONS.ankle_moment_axis
        ax_grf = CONVENTIONS.vgrf_axis
        for path in self._processed_files():
            m = _SUBJECT_RE.search(path.name)
            if not m:
                continue
            subject = m.group(1)
            mass = self._mass_by_subject.get(subject)
            if mass is None:
                continue  # no body mass -> cannot de-normalise; skip honestly
            df = pd.read_csv(path, sep="\t")
            speeds = self._available_speeds(list(df.columns))
            if self.require_all_speeds and len(speeds) < 3:
                continue
            phase = df["PercGcycle"].to_numpy(dtype=float)
            for label in speeds:
                speed_ms = CONVENTIONS.speed_map[label]
                for side in self.sides:
                    ang_col = f"{side}ankleAng{ax_ang}{label}"
                    mom_col = f"{side}ankleMom{ax_mom}{label}"
                    grf_col = f"{side}grf{ax_grf}{label}"
                    if not all(c in df.columns for c in (ang_col, mom_col, grf_col)):
                        continue
                    moment = np.nan_to_num(df[mom_col].to_numpy(dtype=float), nan=0.0)
                    # GRF is NaN during the flight/swing phase (no foot contact);
                    # no contact means zero vertical force.
                    vgrf = np.clip(
                        np.nan_to_num(df[grf_col].to_numpy(dtype=float), nan=0.0),
                        0.0, None,
                    )
                    if not self._is_physiological(moment, vgrf):
                        continue
                    yield GaitTrial(
                        subject_id=subject,
                        side=side,
                        speed_ms=speed_ms,
                        body_mass_kg=mass,
                        gait_phase=phase.copy(),
                        ankle_angle_deg=df[ang_col].to_numpy(dtype=float),
                        ankle_moment_nm_per_kg=moment,
                        vgrf_n_per_kg=vgrf,
                        source="fukuchi",
                    )

    @staticmethod
    def _is_physiological(moment_nm_per_kg: np.ndarray, vgrf_n_per_kg: np.ndarray) -> bool:
        """Reject trials whose peak signals are outside plausible running bounds."""
        peak_moment = float(np.max(moment_nm_per_kg))
        return (
            _MIN_PLAUSIBLE_MOMENT_NM_PER_KG <= peak_moment <= _MAX_PLAUSIBLE_MOMENT_NM_PER_KG
            and float(np.max(vgrf_n_per_kg)) <= _MAX_PLAUSIBLE_VGRF_N_PER_KG
        )
