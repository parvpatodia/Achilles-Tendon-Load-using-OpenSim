"""Stage 2 - physics-informed surrogate.

Trains a small temporal CNN to predict the Achilles force waveform from a
wearable-style input (vertical GRF + ankle angle + four insole-zone channels),
evaluated on HELD-OUT subjects. Optionally ablates the physics-loss terms to
show their effect on generalisation.

Usage:
    python scripts/run_stage2_pinn.py [--source fukuchi|synthetic] [--epochs N]
                                      [--no-ablation]
"""
from __future__ import annotations

import argparse

from achilles.biomech.achilles import AchillesLoadModel
from achilles.config import FIGURES_DIR
from achilles.data.factory import resolve_source
from achilles.ml.dataset import (AchillesSequenceDataset, build_samples,
                                  subject_wise_split)
from achilles.ml.losses import LossWeights
from achilles.ml.trainer import Trainer, TrainConfig
from achilles.viz.plots import plot_stage2_pinn


def _make_datasets(trials, seed, train_subjects=None, noise_std=0.0):
    train_t, test_t, test_subj = subject_wise_split(trials, test_frac=0.25, seed=seed)
    if train_subjects is not None:
        # low-data regime: keep only the first K training subjects (the wearable
        # "cold-start" case, where physics priors should matter most).
        keep = sorted({t.subject_id for t in train_t})[:train_subjects]
        train_t = [t for t in train_t if t.subject_id in keep]
    model = AchillesLoadModel()
    train_s = build_samples(train_t, model)
    test_s = build_samples(test_t, model)
    train_ds = AchillesSequenceDataset(train_s, noise_std=noise_std)
    test_ds = AchillesSequenceDataset(test_s, train_ds.feat_mean, train_ds.feat_std,
                                      noise_std=noise_std)
    return train_ds, test_ds, test_subj


def _run(train_ds, test_ds, epochs, weights, label):
    cfg = TrainConfig(epochs=epochs, weights=weights)
    trainer = Trainer(train_ds, test_ds, cfg)
    print(f"\n[{label}] model params: {trainer.model.num_params():,}")
    trainer.train(verbose=True)
    ev = trainer.evaluate()
    print(f"[{label}] held-out R2={ev.r2:.3f}  RMSE={ev.rmse_bw:.2f} BW  MAE={ev.mae_bw:.2f} BW")
    return trainer, ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fukuchi")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--train-subjects", type=int, default=None,
                    help="cap training subjects (low-data / cold-start regime)")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="simulated sensor noise std (std-units) on wearable inputs")
    args = ap.parse_args()

    src, resolved = resolve_source(args.source)
    trials = src.load_trials()
    train_ds, test_ds, test_subj = _make_datasets(
        trials, args.seed, train_subjects=args.train_subjects, noise_std=args.noise)

    regime = "full data"
    if args.train_subjects or args.noise:
        regime = (f"low-data ({args.train_subjects} subj)" if args.train_subjects else "full") \
                 + (f", sensor noise {args.noise}sd" if args.noise else "")
    print(f"\n=== Stage 2: physics-informed surrogate ({resolved} data, {regime}) ===")
    print(f"train trials: {len(train_ds)}   held-out trials: {len(test_ds)}")
    print(f"held-out subjects: {test_subj}")

    # physics-informed model (full loss)
    trainer, ev = _run(train_ds, test_ds, args.epochs, LossWeights(), "physics-informed")

    # ablation: data-only loss (physics weights zeroed)
    if not args.no_ablation:
        data_only = LossWeights(data=1.0, non_neg=0.0, moment=0.0, smooth=0.0)
        _, ev_abl = _run(train_ds, test_ds, args.epochs, data_only, "data-only (ablation)")
        delta = ev.r2 - ev_abl.r2
        print(f"\nphysics terms change held-out R2 by {delta:+.3f} "
              f"({ev_abl.r2:.3f} -> {ev.r2:.3f})")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "fig2_stage2_pinn.png"
    plot_stage2_pinn(ev.phase, ev.true_curves, ev.pred_curves, ev.r2, ev.rmse_bw, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
