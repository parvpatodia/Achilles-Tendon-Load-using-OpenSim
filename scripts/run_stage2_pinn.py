"""Stage 2 - physics-guided surrogate, evaluated with subject-wise k-fold CV.

Trains a small temporal CNN to predict the Achilles force waveform from a
wearable-style input (vertical GRF + ankle angle + four insole-zone channels).
Every subject is held out exactly once (subject-wise k-fold), so the reported
accuracy is over the whole cohort on people the model never trained on.

Optionally ablates the physics-loss terms (--ablation) to test, with CV
statistics, whether they change generalisation.

Usage:
    python scripts/run_stage2_pinn.py [--source fukuchi|synthetic] [--k 5]
                                      [--epochs N] [--ablation]
"""
from __future__ import annotations

import argparse

import numpy as np

from achilles.config import FIGURES_DIR
from achilles.data.factory import resolve_source
from achilles.ml.cross_val import subject_kfold
from achilles.ml.losses import LossWeights
from achilles.viz.plots import plot_stage2_pinn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fukuchi")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablation", action="store_true")
    args = ap.parse_args()

    src, resolved = resolve_source(args.source)
    trials = src.load_trials()

    print(f"\n=== Stage 2: physics-guided surrogate, {args.k}-fold subject-wise CV "
          f"({resolved} data) ===")
    print(f"{len(trials)} trials, {len(src.subjects())} subjects, "
          f"every subject held out once\n")

    cv = subject_kfold(trials, k=args.k, weights=LossWeights(), epochs=args.epochs,
                       seed=args.seed)
    print(f"\nphysics-guided:  held-out R2 = {cv.mean_r2:.3f} +/- {cv.std_r2:.3f} "
          f"(across {cv.k} folds)")
    print(f"                 pooled R2 = {cv.pooled_r2:.3f}, "
          f"RMSE = {cv.pooled_rmse_bw:.2f} BW, MAE = {cv.pooled_mae_bw:.2f} BW")

    subtitle = (f"{args.k}-fold subject-wise CV: R$^2$ = {cv.mean_r2:.3f} "
                f"$\\pm$ {cv.std_r2:.3f}  (every subject held out once)")

    if args.ablation:
        data_only = LossWeights(data=1.0, non_neg=0.0, moment=0.0, smooth=0.0)
        cv_abl = subject_kfold(trials, k=args.k, weights=data_only,
                               epochs=args.epochs, seed=args.seed, verbose=False)
        print(f"\ndata-only (ablation): held-out R2 = {cv_abl.mean_r2:.3f} "
              f"+/- {cv_abl.std_r2:.3f}")
        delta = cv.mean_r2 - cv_abl.mean_r2
        pooled_std = (cv.std_r2 + cv_abl.std_r2) / 2
        verdict = ("within fold-to-fold noise" if abs(delta) < pooled_std
                   else "outside fold noise")
        print(f"physics effect on mean R2: {delta:+.3f}  ({verdict})")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "fig2_stage2_pinn.png"
    plot_stage2_pinn(cv.phase, cv.true_curves, cv.pred_curves, cv.pooled_r2,
                     cv.pooled_rmse_bw, out, subtitle=subtitle)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
