"""Uncertainty quantification: deep-ensemble predictive bands + calibration.

Usage:
    python scripts/run_uncertainty.py [--source fukuchi|synthetic]
                                      [--models K] [--epochs N]
"""
from __future__ import annotations

import argparse

from achilles.config import FIGURES_DIR
from achilles.data.factory import resolve_source
from achilles.ml.uncertainty import deep_ensemble_cv
from achilles.viz.plots import plot_uncertainty


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fukuchi")
    ap.add_argument("--models", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    src, resolved = resolve_source(args.source)
    trials = src.load_trials()
    print(f"\n=== Uncertainty quantification ({resolved}, deep ensemble of "
          f"{args.models} nets x {args.k}-fold) ===")
    uq = deep_ensemble_cv(trials, k=args.k, n_models=args.models, epochs=args.epochs)

    print(f"mean 90% conformal band half-width: {uq.mean_halfwidth_bw:.2f} BW")
    print("calibration (nominal -> empirical coverage on held-out subjects, cross-fold conformal):")
    for nom, emp in uq.coverage_table():
        print(f"  {nom:.0%} -> {emp:.0%}")
    gap = max(abs(n - e) for n, e in zip(uq.nominal, uq.empirical))
    print(f"max nominal-vs-empirical gap: {gap:.0%} "
          f"({'well calibrated' if gap <= 0.05 else 'close; reported honestly'})")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "fig11_uncertainty.png"
    plot_uncertainty(uq, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
