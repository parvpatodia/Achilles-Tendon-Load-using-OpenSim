"""Grouvel c3d -> OpenSim inputs: markers .trc + ground-reaction .mot.

Stage 1 of the Grouvel measured-pressure pipeline (see README, measured-pressure
section, WIP). Reads a Grouvel c3d with OpenSim's C3DFileAdapter (which also
computes force-plate centre of pressure) and writes an OpenSim-ready .trc
(markers) and .mot (GRF), converting the lab frame to the model frame.

Frame/units: Grouvel c3d is Z-up, millimetres; OpenSim is Y-up, metres. The
rotation (x, y, z)_c3d -> (x, z, -y)_osim is a -90 deg rotation about X that
preserves handedness. Validated on P02 (LFHD Y ~= 1.64 m, RHEE Y ~= 0.10 m).

IMPORTANT: OpenSim and ezc3d must NOT be imported in the same process (their
bundled native libraries conflict and segfault). This module uses OpenSim only.
Subject mass/anthropometry live in the c3d SUBJECTS parameters; read those with
ezc3d in a SEPARATE process (see scripts/grouvel_subject_info.py).

Usage:
    python scripts/grouvel_prep.py --c3d <trial.c3d> --outdir <dir> [--name <stem>]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import opensim as osim


def read_tables(path: str):
    a = osim.C3DFileAdapter()
    a.setLocationForForceExpression(osim.C3DFileAdapter.ForceLocation_CenterOfPressure)
    t = a.read(path)
    return a.getMarkersTable(t), a.getForcesTable(t)


def _np(tab, label: str) -> np.ndarray:
    cc = tab.getDependentColumn(label)
    return np.array([[cc[i].get(0), cc[i].get(1), cc[i].get(2)] for i in range(tab.getNumRows())])


def _rot(v: np.ndarray) -> np.ndarray:
    """(n,3) Grouvel c3d (Z-up) -> OpenSim (Y-up); units unchanged here."""
    return np.column_stack([v[:, 0], v[:, 2], -v[:, 1]])


def write_trc(mk, path: Path) -> None:
    labels = list(mk.getColumnLabels())
    time = np.array(mk.getIndependentColumn())
    n = len(time)
    rate = 1.0 / (time[1] - time[0])
    data = {l: _rot(_np(mk, l)) / 1000.0 for l in labels}  # mm -> m
    with open(path, "w") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{path.name}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{rate:.0f}\t{rate:.0f}\t{n}\t{len(labels)}\tm\t{rate:.0f}\t1\t{n}\n")
        f.write("Frame#\tTime\t" + "\t\t\t".join(labels) + "\t\t\n")
        f.write("\t\t" + "\t".join(f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(len(labels))) + "\n\n")
        for i in range(n):
            row = [str(i + 1), f"{time[i]:.5f}"]
            for l in labels:
                x, y, z = data[l][i]
                row += [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]
            f.write("\t".join(row) + "\n")


def write_grf_mot(fo, path: Path, n_plates: int = 3) -> float:
    time = np.array(fo.getIndependentColumn())
    n = len(time)
    cols: dict[str, np.ndarray] = {}
    for p in range(1, n_plates + 1):
        F = _rot(_np(fo, f"f{p}"))             # N
        C = _rot(_np(fo, f"p{p}")) / 1000.0    # COP mm -> m
        M = _rot(_np(fo, f"m{p}")) / 1000.0    # free moment N*mm -> N*m
        for axis, j in (("x", 0), ("y", 1), ("z", 2)):
            cols[f"ground_force_{p}_v{axis}"] = F[:, j]
            cols[f"ground_force_{p}_p{axis}"] = C[:, j]
            cols[f"ground_torque_{p}_{axis}"] = M[:, j]
    names = list(cols.keys())
    with open(path, "w") as f:
        f.write(f"{path.name}\nversion=1\nnRows={n}\nnColumns={len(names)+1}\n"
                "inDegrees=no\nendheader\n")
        f.write("time\t" + "\t".join(names) + "\n")
        for i in range(n):
            f.write(f"{time[i]:.5f}\t" + "\t".join(f"{cols[nm][i]:.6f}" for nm in names) + "\n")
    return float(sum(cols[f"ground_force_{p}_vy"] for p in range(1, n_plates + 1)).max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3d", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--name", default=None, help="output stem (default: c3d filename)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.name or Path(args.c3d).stem
    mk, fo = read_tables(args.c3d)

    trc = outdir / f"{stem}.trc"
    write_trc(mk, trc)
    head = _rot(_np(mk, "LFHD"))[:, 1].mean() / 1000.0 if "LFHD" in mk.getColumnLabels() else float("nan")
    print(f"wrote {trc}  ({mk.getNumRows()} frames; LFHD Y={head:.2f} m)")

    if fo.getNumColumns() > 0:
        mot = outdir / f"{stem}_grf.mot"
        peak = write_grf_mot(fo, mot)
        print(f"wrote {mot}  ({fo.getNumRows()} rows; summed vertical GRF peak {peak:.1f} N)")
    else:
        print("no force data in this c3d (static/other); skipped GRF .mot")


if __name__ == "__main__":
    main()
