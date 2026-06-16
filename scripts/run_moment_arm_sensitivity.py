"""Moment-arm sensitivity: how much does the assumed lever change the answer?

Sweeps the Achilles moment arm across the literature range (4-6 cm) and reports
how peak tendon force and stress move. Quantifies the single biggest assumption.

Usage:
    python scripts/run_moment_arm_sensitivity.py [--source fukuchi|synthetic]
"""
from __future__ import annotations

import argparse

import numpy as np

from achilles.biomech.sensitivity import moment_arm_sensitivity
from achilles.config import FIGURES_DIR
from achilles.data.factory import resolve_source
from achilles.viz.plots import plot_moment_arm_sensitivity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fukuchi")
    args = ap.parse_args()

    src, resolved = resolve_source(args.source)
    trials = src.load_trials()
    sens = moment_arm_sensitivity(trials, arms_cm=np.linspace(4.0, 6.0, 11))

    print(f"\n=== Moment-arm sensitivity ({resolved} data, {len(trials)} trials) ===")
    for r, f, s in zip(sens.arms_cm, sens.peak_force_bw_mean, sens.peak_stress_mpa_mean):
        print(f"  arm {r:.1f} cm -> peak force {f:.2f} BW, peak stress {s:.0f} MPa")
    print(f"\npeak force swings {sens.force_swing_pct():.0f}% across 4-6 cm "
          f"-> the moment arm is the dominant uncertainty, which is why Stage 3 "
          f"cross-checks it and per-athlete measurement would pin it down.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "fig6_moment_arm_sensitivity.png"
    plot_moment_arm_sensitivity(sens, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
