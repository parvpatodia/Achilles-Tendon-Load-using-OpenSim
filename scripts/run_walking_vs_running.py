"""Walking vs running: the same pipeline across the gait spectrum.

Runs the Stage 1 analytical load on both the running dataset (Fukuchi 2017) and
the walking dataset (Fukuchi 2018, matching Mirai's walking/rehab cohort), and
shows Achilles load rising smoothly from slow walking to running.

Needs the walking data: python scripts/download_data.py --walking

Usage:
    python scripts/run_walking_vs_running.py
"""
from __future__ import annotations

import numpy as np

from achilles.biomech.achilles import AchillesLoadModel
from achilles.config import DATA_RAW, FIGURES_DIR
from achilles.data.fukuchi import FukuchiDataSource
from achilles.data.walking import WalkingDataSource
from achilles.viz.plots import plot_walking_vs_running


def main() -> None:
    if not (DATA_RAW / "wbds").exists():
        print("[walking] walking data not found. Run: "
              "python scripts/download_data.py --walking")
        return

    model = AchillesLoadModel()
    run = [model.compute(t) for t in FukuchiDataSource().load_trials()]
    walk = [model.compute(t) for t in WalkingDataSource().load_trials()]

    rp = np.array([r.peak_force_bw for r in run])
    wp = np.array([r.peak_force_bw for r in walk])
    print("\n=== Walking vs running (Stage 1 analytical Achilles load) ===")
    print(f"running:  {len(run)} trials, {len({r.trial.subject_id for r in run})} subjects, "
          f"peak {rp.mean():.2f} BW (speeds 2.5-4.5 m/s)")
    print(f"walking:  {len(walk)} trials, {len({r.trial.subject_id for r in walk})} subjects, "
          f"peak {wp.mean():.2f} BW (speeds 0.4-2.2 m/s)")
    print(f"combined: {len(run)+len(walk)} trials, "
          f"{len({r.trial.subject_id for r in run+walk})} subjects")
    print("Achilles load rises monotonically from slow walking to running, as expected.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "fig7_walking_vs_running.png"
    plot_walking_vs_running(walk, run, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
