"""Subject-specific calibration: how few of an athlete's own labelled steps
remove the systematic per-person bias, and what it does to the worst-athlete
case (the number a wearable is judged on).

Usage:
    python scripts/run_calibration.py [--source fukuchi|synthetic]
                                      [--method affine|offset|identity] [--draws N]
"""
from __future__ import annotations

import argparse

from achilles.config import FIGURES_DIR
from achilles.data.factory import resolve_source
from achilles.ml.calibration import subject_calibration_sweep
from achilles.viz.plots import plot_calibration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fukuchi")
    ap.add_argument("--method", default="affine", choices=["identity", "offset", "affine"])
    ap.add_argument("--draws", type=int, default=5)
    args = ap.parse_args()

    src, resolved = resolve_source(args.source)
    trials = src.load_trials()
    print(f"\n=== Subject-specific calibration ({resolved}, {args.method}, "
          f"recommended linear model) ===")
    res = subject_calibration_sweep(trials, method=args.method, n_draws=args.draws)
    print(res.table())

    k = res.ks[-1]
    d_worst = res.cal_worst_loaded_r2[k] - res.uncal_worst_loaded_r2[k]
    print(f"\nReading it honestly: with {k} of the athlete's own labelled steps, the worst "
          f"held-out athlete's loaded R2 moves {res.uncal_worst_loaded_r2[k]:.2f} -> "
          f"{res.cal_worst_loaded_r2[k]:.2f} ({d_worst:+.2f}) and peak error "
          f"{res.uncal_peak_mape[k]:.1f}% -> {res.cal_peak_mape[k]:.1f}%. The base model never "
          f"trains on the held-out athlete; the calibration steps are excluded from scoring and, "
          f"at deployment, come once at onboarding from the lab reference. This turns the honest "
          f"weakness (a new athlete can be off by a systematic bias) into a fixable one-time step.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = plot_calibration(res, FIGURES_DIR / "fig13_calibration.png")
    print(f"saved: {fig_path}")


if __name__ == "__main__":
    main()
