"""Grouvel measured-pressure pipeline, stage 3: read the Moticon 16-zone insole
pressure from a per-trial SYNC_DATA csv.

This is the MEASURED input that replaces the repo's derived 4-zone proxy
(features.py). The SYNC_DATA csv is already time-synced to the trial's 100 Hz
marker frames (so it aligns to the OpenSim IK/ID output by frame index), and
carries, per foot: 16 pressure sensors (N/cm^2), a 6-axis IMU, total force (N),
and centre of pressure. Plain csv, so this needs neither OpenSim nor ezc3d.

Usage:
    python scripts/grouvel_pressure.py --csv <SYNC_DATA/..._Gait_01.csv> [--bw 490]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

RATE_HZ = 100.0


@dataclass
class InsolePressure:
    t: np.ndarray            # (N,) seconds
    left_zones: np.ndarray   # (N,16) N/cm^2
    right_zones: np.ndarray  # (N,16)
    left_total: np.ndarray   # (N,) total force N
    right_total: np.ndarray  # (N,)

    def total(self, side: str) -> np.ndarray:
        return self.left_total if side == "l" else self.right_total

    def zones(self, side: str) -> np.ndarray:
        return self.left_zones if side == "l" else self.right_zones


def read_pressure(csv_path: str) -> InsolePressure:
    df = pd.read_csv(csv_path)
    lz = df[[f"leftPressure{k}_N_cm___" for k in range(1, 17)]].to_numpy(dtype=float)
    rz = df[[f"rightPressure{k}_N_cm___" for k in range(1, 17)]].to_numpy(dtype=float)
    lt = df["leftTotalForce_N_"].to_numpy(dtype=float)
    rt = df["rightTotalForce_N_"].to_numpy(dtype=float)
    n = len(df)
    return InsolePressure(np.arange(n) / RATE_HZ, np.nan_to_num(lz), np.nan_to_num(rz),
                          np.nan_to_num(lt), np.nan_to_num(rt))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--bw", type=float, default=490.0, help="body weight (N) for the sanity check")
    ap.add_argument("--plot", default=None, help="optional png path")
    args = ap.parse_args()

    p = read_pressure(args.csv)
    print(f"frames {len(p.t)} @ {RATE_HZ:.0f}Hz ({p.t[-1]:.2f}s)")
    for side in ("l", "r"):
        tot = p.total(side)
        print(f"  {side}: total-force peak {tot.max():.0f} N ({tot.max()/args.bw:.2f} BW), "
              f"zones peak {p.zones(side).max():.1f} N/cm^2")
    # sanity: insole total force should peak near body weight during stance
    ratio = max(p.left_total.max(), p.right_total.max()) / args.bw
    print(f"insole peak vs BW = {ratio:.2f} (expect ~1.0-1.2 for walking stance)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(p.t, p.left_total, label="left", color="#F58518")
        ax[0].plot(p.t, p.right_total, label="right", color="#4C78A8")
        ax[0].axhline(args.bw, color="k", ls=":", lw=1, label="body weight")
        ax[0].set_title("Insole total force"); ax[0].set_xlabel("s"); ax[0].set_ylabel("N"); ax[0].legend()
        # a right-foot stance: heat of the 16 zones over time
        ax[1].imshow(p.right_zones.T, aspect="auto", origin="lower",
                     extent=[0, p.t[-1], 0.5, 16.5], cmap="magma")
        ax[1].set_title("Right insole, 16 zones"); ax[1].set_xlabel("s"); ax[1].set_ylabel("sensor")
        fig.tight_layout(); fig.savefig(args.plot, dpi=110)
        print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
