"""Data-efficiency experiment: does the physics-guided loss earn its place when
training subjects are scarce (the real calibration-cohort regime, README sec 9)?

Grows the number of training subjects against a FIXED held-out test set and
scores, on identical data, the physics-guided CNN, the same net without the
physics terms, and the linear baseline. Reports loaded-phase R^2 mean +/- SD
over seeds, and the physics-minus-data-only gap at each size.

Usage:
    python scripts/run_data_efficiency.py [--source fukuchi|synthetic]
        [--sizes 4 8 12 16 20] [--seeds N] [--test-subjects N] [--epochs N]
"""
from __future__ import annotations

import argparse

import numpy as np

from achilles.config import FIGURES_DIR
from achilles.data.factory import resolve_source
from achilles.ml.data_efficiency import DATA_ONLY, PHYSICS, data_efficiency_curve
from achilles.viz.plots import plot_data_efficiency


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fukuchi")
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 12, 16, 20])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--test-subjects", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()

    src, resolved = resolve_source(args.source)
    trials = src.load_trials()
    print(f"\n=== Data efficiency ({resolved}, {args.seeds} seeds, "
          f"{args.test_subjects} held-out test subjects, {args.epochs} epochs) ===")

    result = data_efficiency_curve(
        trials, sizes=tuple(args.sizes), n_seeds=args.seeds,
        n_test_subjects=args.test_subjects, epochs=args.epochs)

    print("\nloaded-phase R^2 (held-out, mean +/- SD over seeds):")
    print(result.table())

    gap = result.gap_physics_minus_dataonly()
    small = gap[0]
    big = gap[-1]
    print(f"\nphysics-minus-data-only gap: {small:+.3f} at {result.sizes[0]} subjects, "
          f"{big:+.3f} at {result.sizes[-1]} subjects")
    if small > 0.01 and small > big:
        print("verdict: the physics prior helps most when subjects are few, and the "
              "advantage shrinks as data grows (the small-cohort deployment benefit).")
    elif abs(big) < 0.01 and abs(small) < 0.01:
        print("verdict: no measurable physics gain at any size; the physics terms act "
              "as a validity guardrail, not an accuracy gain. Linear remains sufficient.")
    else:
        print("verdict: reported as measured (see table); read the gap column literally.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "fig12_data_efficiency.png"
    plot_data_efficiency(result, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
