"""Grouvel measured-pressure pipeline, stage 4: harvest training pairs.

Turns one trial's OpenSim output (scripts/grouvel_opensim.py) plus its synced
Moticon pressure (scripts/grouvel_pressure.py) into committable per-stance CSVs:
each on-plate single-limb stance, time-normalised to 101 points, carrying the
Achilles TARGET (ankle angle + plantarflexion moment from IK/ID + force plate)
and the measured INPUT (16->4 plantar-pressure zones + insole total force).

Only stances where a foot is fully on a plate are kept: elsewhere there is no
GRF for that limb, so the inverse-dynamics moment (hence the Achilles target) is
meaningless. Each plate is assigned to the foot whose heel is nearest its COP
over the plate's active window (validated in grouvel_opensim.py); a foot may
straddle two adjacent plates in one stance, so plates are grouped per foot and
their GRF summed.

A stance is emitted only if it passes a measured-input TRUST GATE: the insole
total force must track the reference force-plate GRF over the stance
(corr >= _MIN_GRF_CORR). This refuses to pair the Achilles target with pressure
that does not actually match the ground truth, so the surrogate is never trained
on corrupted input. For the public Grouvel SYNC_DATA this gate rejects the P02
walking stances (insole-vs-plate r ~= 0.1-0.3; see README 6j).

OpenSim + pandas only (never import ezc3d here; they segfault together).

Usage:
    python scripts/grouvel_harvest.py --osim <osim_out> --prep <prep> \
        --stem P02_S01_Gait_01 --pressure <SYNC_DATA/..._Gait_01.csv> \
        --out data/grouvel_processed/stances
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grouvel_opensim import _col, assign_plates, moment_arm_m  # noqa: E402
from grouvel_pressure import read_pressure  # noqa: E402

import opensim as osim  # noqa: E402

GRAVITY = 9.81
N_PHASE = 101
GRF_THRESH_N = 40.0

# 16-zone Moticon insole -> the repo's 4 anatomical regions (features.ZONE_NAMES
# order). Sensors are numbered heel->toe; grouping validated by the roll-over QC
# below (heel peaks early in stance, forefoot mid, toes at push-off), consistent
# with the Stage-3 heatmap (sensors 1-3 heel, 10-14 forefoot).
ZONE_SENSORS = {
    "heel": [1, 2, 3],
    "arch": [4, 5, 6, 7],
    "forefoot": [8, 9, 10, 11, 12, 13],
    "bigtoe": [14, 15, 16],
}
ZONE_ORDER = ("heel", "arch", "forefoot", "bigtoe")

# Physiological QC (walking).
_MIN_PEAK_BW, _MAX_PEAK_BW = 0.5, 4.0
_MIN_STANCE_S, _MAX_STANCE_S = 0.35, 1.2
# Measured-input trust gate: the insole total force must track the reference
# force-plate GRF over the stance, else the pressure cannot be honestly paired
# with the OpenSim target. Literature insole-vs-plate agreement is r>0.9; 0.70
# is a lenient floor. NOTE: the public Grouvel SYNC_DATA insole (~31 Hz, resampled
# onto the 100 Hz mocap grid) fails this for P02 walking (r~=0.1-0.3), so its
# per-stance pressure is NOT trustworthy as a training input; this gate refuses
# it rather than train the surrogate on corrupted pressure. See README 6j.
_MIN_GRF_CORR = 0.70


def _subject_mass(stem: str, subjects_csv: Path) -> tuple[str, float]:
    subject = stem.split("_")[0]
    df = pd.read_csv(subjects_csv)
    row = df[df["subject"] == subject]
    if row.empty:
        raise ValueError(f"{subject} not in {subjects_csv}")
    return subject, float(row["body_mass_kg"].iloc[0])


def _resample(t_src: np.ndarray, v_src: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Interpolate a signal onto N_PHASE points spanning the stance window."""
    return np.interp(np.linspace(t0, t1, N_PHASE), t_src, v_src)


def _aggregate_zones(zones16: np.ndarray) -> dict[str, np.ndarray]:
    """(N,16) N/cm^2 -> per-region mean pressure (N,) each, in ZONE_ORDER."""
    return {z: zones16[:, [s - 1 for s in ZONE_SENSORS[z]]].mean(axis=1) for z in ZONE_ORDER}


def _segments(t: np.ndarray, active: np.ndarray, min_dur: float, max_gap_s: float = 0.04):
    """Contiguous stance segments (t0, t1) where `active`, merging brief gaps.

    A foot can straddle two adjacent force plates in ONE stance (heel on one,
    toe on the next); the plates' summed GRF is continuous, so a single active
    run is one stance. Short drop-outs are merged; runs shorter than min_dur are
    dropped (double-support brushes, not a real stance)."""
    idx = np.where(active)[0]
    if len(idx) == 0:
        return []
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.01
    max_gap = max(1, int(round(max_gap_s / dt)))
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev <= max_gap:
            prev = i
        else:
            runs.append((start, prev)); start = prev = i
    runs.append((start, prev))
    return [(float(t[a]), float(t[b])) for a, b in runs if float(t[b] - t[a]) >= min_dur]


def harvest_trial(osim_dir: Path, prep: Path, stem: str, pressure_csv: Path,
                  out_dir: Path, subjects_csv: Path, verbose: bool = True) -> list[Path]:
    subject, mass = _subject_mass(stem, subjects_csv)
    bw = mass * GRAVITY
    out_dir.mkdir(parents=True, exist_ok=True)

    ik = osim.TimeSeriesTable(str(osim_dir / "ik.mot")); it = np.array(ik.getIndependentColumn())
    idt = osim.TimeSeriesTable(str(osim_dir / "id.sto")); dt = np.array(idt.getIndependentColumn())
    grf_mot = str(prep / f"{stem}_grf.mot")
    grf = osim.TimeSeriesTable(grf_mot); gt = np.array(grf.getIndependentColumn())
    pressure = read_pressure(str(pressure_csv)); pt = pressure.t

    plate_foot = assign_plates(grf_mot, str(prep / f"{stem}.trc"))
    # group plates by foot; a foot's total GRF is the sum of its plates (handles
    # the straddle where one stance spans two adjacent plates).
    side_plates: dict[str, list[int]] = {"r": [], "l": []}
    for p, foot in plate_foot.items():
        side_plates["r" if foot == "calcn_r" else "l"].append(p)

    written: list[Path] = []
    for side, plates in side_plates.items():
        if not plates:
            continue
        vy_sum = np.sum([_col(grf, f"ground_force_{p}_vy") for p in plates], axis=0)  # total foot GRF
        seg_idx = 0
        for t0, t1 in _segments(gt, vy_sum > GRF_THRESH_N, _MIN_STANCE_S):
            dur = t1 - t0
            if dur > _MAX_STANCE_S:
                if verbose:
                    print(f"  drop {stem} {side.upper()}: stance {dur:.2f}s too long")
                continue
            angle = _resample(it, _col(ik, f"ankle_angle_{side}"), t0, t1)
            moment = _resample(dt, _col(idt, f"ankle_angle_{side}_moment"), t0, t1)
            pf_moment = np.clip(-moment, 0.0, None)               # plantarflexion Nm >= 0
            vgrf = np.clip(_resample(gt, vy_sum, t0, t1), 0.0, None)  # summed foot GRF, N
            force_bw = pf_moment / moment_arm_m(angle) / bw
            peak_bw = float(force_bw.max())
            if not (_MIN_PEAK_BW <= peak_bw <= _MAX_PEAK_BW):
                if verbose:
                    print(f"  drop {stem} {side.upper()}: peak Achilles {peak_bw:.2f} BW implausible")
                continue

            insole_total = _resample(pt, pressure.total(side), t0, t1)
            # trust gate: measured insole force must agree with the reference GRF
            grf_corr = float(np.corrcoef(vgrf, insole_total)[0, 1])
            if not np.isfinite(grf_corr) or grf_corr < _MIN_GRF_CORR:
                if verbose:
                    print(f"  drop {stem} {side.upper()}: insole-vs-GRF r={grf_corr:.2f}"
                          f" < {_MIN_GRF_CORR} (measured pressure not trustworthy)")
                continue
            zagg = _aggregate_zones(pressure.zones(side))         # region -> (N,)
            stance_zones = {z: _resample(pt, zagg[z], t0, t1) for z in ZONE_ORDER}

            df = pd.DataFrame({
                "phase": np.linspace(0, 100, N_PHASE),
                "ankle_angle_deg": angle,
                "ankle_moment_nm": pf_moment,
                "vgrf_n": vgrf,
                **{f"zone_{z}": stance_zones[z] for z in ZONE_ORDER},
                "insole_total_n": insole_total,
            })
            seg_idx += 1
            path = out_dir / f"{stem}_{side.upper()}{seg_idx}.csv"
            df.to_csv(path, index=False, float_format="%.5g")
            written.append(path)
            if verbose:
                print(f"  {path.name}: peak Achilles {peak_bw:.2f} BW, insole peak "
                      f"{insole_total.max()/bw:.2f} BW, insole-GRF r={grf_corr:.2f}, "
                      f"stance {dur:.2f}s")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--osim", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--pressure", required=True)
    ap.add_argument("--out", default="data/grouvel_processed/stances")
    ap.add_argument("--subjects", default="data/grouvel_processed/subjects.csv")
    args = ap.parse_args()
    written = harvest_trial(Path(args.osim), Path(args.prep), args.stem,
                            Path(args.pressure), Path(args.out), Path(args.subjects))
    print(f"harvested {len(written)} stance(s) from {args.stem}")


if __name__ == "__main__":
    main()
