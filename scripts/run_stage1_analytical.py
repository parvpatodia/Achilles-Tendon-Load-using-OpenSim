"""Stage 1 - analytical Achilles tendon load from real running data.

Computes tendon force / stress / strain across the gait cycle for every trial,
prints a literature-comparison summary, and saves the Stage 1 figure.

Usage:
    python scripts/run_stage1_analytical.py [--source fukuchi|synthetic]
"""
from __future__ import annotations

import argparse

import numpy as np

from achilles.biomech.achilles import AchillesLoadModel
from achilles.biomech.moment_arm import AngleDependentMomentArm
from achilles.config import FIGURES_DIR, TENDON
from achilles.data.factory import resolve_source
from achilles.viz.plots import plot_stage1_achilles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fukuchi")
    args = ap.parse_args()

    src, resolved = resolve_source(args.source)
    trials = src.load_trials()
    model = AchillesLoadModel(moment_arm=AngleDependentMomentArm())
    results = [model.compute(t) for t in trials]

    print(f"\n=== Stage 1: analytical Achilles load ({resolved} data) ===")
    print(f"subjects: {len(src.subjects())}   trials: {len(trials)}   "
          f"moment arm: {model.moment_arm.name}")

    pf = np.array([r.peak_force_bw for r in results])
    ps = np.array([r.peak_stress_mpa for r in results])
    print(f"peak Achilles force : mean {pf.mean():.2f} BW  "
          f"(range {pf.min():.2f}-{pf.max():.2f})   [literature: ~4-7 BW running]")
    print(f"peak tendon stress  : mean {ps.mean():.1f} MPa "
          f"(range {ps.min():.1f}-{ps.max():.1f})   [ultimate ~{TENDON.ultimate_stress_pa/1e6:.0f} MPa]")
    print(f"safety factor (mean): {TENDON.ultimate_stress_pa/1e6 / ps.mean():.2f}x to failure stress")

    print("\nspeed trend (peak force):")
    for sp in sorted({t.speed_ms for t in trials}):
        sub = [r.peak_force_bw for r in results if r.trial.speed_ms == sp]
        print(f"  {sp:g} m/s : {np.mean(sub):.2f} BW  (n={len(sub)})")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "fig1_stage1_achilles_load.png"
    plot_stage1_achilles(results, out,
                         title_suffix="" if resolved == "fukuchi" else "  [SYNTHETIC]")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
