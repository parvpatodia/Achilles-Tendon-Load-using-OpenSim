"""Stage 3 (bonus) - OpenSim cross-check.

Feeds the same measured ankle plantarflexion moment through two moment-arm
sources - our analytical angle-dependent model and OpenSim's validated Gait2392
triceps surae geometry - and compares the resulting Achilles force. This
cross-checks the single biggest modelling assumption (the moment arm).

Skips cleanly (exit 0) if OpenSim or the model file is unavailable, so it never
sinks the demo.

Usage:
    python scripts/run_stage3_opensim.py [--subject RBDS010]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from achilles.biomech.achilles import AchillesLoadModel
from achilles.biomech.moment_arm import AngleDependentMomentArm
from achilles.config import DATA_RAW, FIGURES_DIR
from achilles.data.factory import resolve_source
from achilles.viz.plots import plot_stage3_opensim

MODEL_PATH = DATA_RAW.parent / "opensim" / "gait2392_thelen2003muscle.osim"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="RBDS010")
    ap.add_argument("--speed", type=float, default=3.5)
    args = ap.parse_args()

    try:
        import opensim  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"[stage3] OpenSim not importable ({e}); skipping cross-check (bonus).")
        return 0
    if not MODEL_PATH.exists():
        print(f"[stage3] model {MODEL_PATH} missing; skipping cross-check (bonus). "
              f"To reproduce it, place the standard OpenSim Gait2392 model "
              f"(gait2392_thelen2003muscle.osim, shipped with the OpenSim application "
              f"under Resources/Models/Gait2392_Simbody) at that path.")
        return 0

    from achilles.opensim_xcheck.pipeline import OpenSimMomentArmModel

    src, resolved = resolve_source("fukuchi")
    trials = src.load_trials()
    subs = sorted({t.subject_id for t in trials})
    subj = args.subject if args.subject in subs else subs[len(subs) // 2]
    trial = next((t for t in trials if t.subject_id == subj and t.side == "R"
                  and t.speed_ms == args.speed), None)
    if trial is None:
        trial = next(t for t in trials if t.side == "R")

    osim_ma = OpenSimMomentArmModel(MODEL_PATH, side="R")
    analytical = AchillesLoadModel(moment_arm=AngleDependentMomentArm()).compute(trial)
    opensim = AchillesLoadModel(moment_arm=osim_ma).compute(trial)

    print(f"\n=== Stage 3: OpenSim cross-check ({subj} R, {trial.speed_ms:g} m/s) ===")
    print(f"OpenSim triceps surae moment arm: {osim_ma.summary()}")
    print(f"analytical moment arm (mean): {analytical.moment_arm_m.mean()*100:.2f} cm")
    print(f"peak Achilles force - analytical {analytical.peak_force_bw:.2f} BW  vs  "
          f"OpenSim {opensim.peak_force_bw:.2f} BW")
    # agreement on the loaded portion
    mask = analytical.force_bw > 0.2
    diff = np.abs(analytical.force_bw[mask] - opensim.force_bw[mask])
    rel = 100 * diff.mean() / analytical.force_bw[mask].mean()
    print(f"mean abs difference over stance: {diff.mean():.2f} BW ({rel:.1f}% of mean force)")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "fig5_stage3_opensim_xcheck.png"
    plot_stage3_opensim(trial.gait_phase, analytical.force_bw, opensim.force_bw, out,
                        subject_label=f"({subj} R, {trial.speed_ms:g} m/s)")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
