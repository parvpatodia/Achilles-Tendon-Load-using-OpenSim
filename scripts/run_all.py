"""Run the full pipeline end to end and produce every figure.

    python scripts/run_all.py [--source fukuchi|synthetic] [--epochs N]

Stages 1, 2, 4 are the MVP; Stage 3 (OpenSim) runs if available and is skipped
cleanly otherwise.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from achilles.config import FIGURES_DIR
from achilles.viz.plots import plot_pipeline_diagram

SCRIPTS = Path(__file__).resolve().parent


def _run(name: str, extra: list[str]) -> None:
    print(f"\n{'='*70}\n>>> {name}\n{'='*70}")
    subprocess.run([sys.executable, str(SCRIPTS / name), *extra], check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fukuchi")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_pipeline_diagram(FIGURES_DIR / "fig0_pipeline.png")
    print(f"saved: {FIGURES_DIR / 'fig0_pipeline.png'}")

    _run("run_stage1_analytical.py", ["--source", args.source])
    _run("run_stage2_pinn.py", ["--source", args.source, "--epochs", str(args.epochs),
                                "--k", str(args.k)])
    _run("run_model_comparison.py", ["--source", args.source, "--epochs", str(args.epochs),
                                     "--k", str(args.k)])
    _run("run_moment_arm_sensitivity.py", ["--source", args.source])
    _run("run_robustness.py", ["--source", args.source, "--epochs", str(args.epochs)])
    _run("run_uncertainty.py", ["--source", args.source])
    _run("run_stage4_product.py", ["--source", args.source])
    _run("run_walking_vs_running.py", [])  # needs --walking data; skips cleanly if absent
    _run("run_stage3_opensim.py", [])      # bonus; skips cleanly if unavailable

    figs = sorted(FIGURES_DIR.glob("*.png"))
    print(f"\n{'='*70}\nDONE. {len(figs)} figures in {FIGURES_DIR}:")
    for f in figs:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
