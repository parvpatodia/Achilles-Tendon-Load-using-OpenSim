"""Stage 4 - the product view: relative load index, asymmetry, accumulation.

Seeds from one real bilateral recording (left + right Achilles loading impulse
at 3.5 m/s) and simulates a multi-session training block to show the
longitudinal, tissue-aware load monitoring a continuous insole would enable.
Relative and indicative, not absolute or predictive.

Usage:
    python scripts/run_stage4_product.py [--source fukuchi|synthetic] [--subject RBDS010]
"""
from __future__ import annotations

import argparse

import numpy as np

from achilles.biomech.achilles import AchillesLoadModel
from achilles.config import FIGURES_DIR
from achilles.data.factory import resolve_source
from achilles.product.load_index import (AccumulationTimeline, AsymmetryAnalyzer,
                                          SessionSimulator)
from achilles.viz.plots import plot_stage4_accumulation, plot_stage4_asymmetry


def _pick_subject(trials, subject, speed=3.5):
    subs = sorted({t.subject_id for t in trials})
    target = subject if subject in subs else subs[len(subs) // 2]
    left = next((t for t in trials if t.subject_id == target and t.side == "L"
                 and t.speed_ms == speed), None)
    right = next((t for t in trials if t.subject_id == target and t.side == "R"
                  and t.speed_ms == speed), None)
    if left is None or right is None:  # synthetic / single-speed fallback
        speed = sorted({t.speed_ms for t in trials})[0]
        left = next(t for t in trials if t.subject_id == target and t.side == "L"
                    and t.speed_ms == speed)
        right = next(t for t in trials if t.subject_id == target and t.side == "R"
                     and t.speed_ms == speed)
    return target, left, right, speed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fukuchi")
    ap.add_argument("--subject", default="RBDS010")
    ap.add_argument("--sessions", type=int, default=14)
    args = ap.parse_args()

    src, resolved = resolve_source(args.source)
    trials = src.load_trials()
    subject, left_t, right_t, speed = _pick_subject(trials, args.subject)

    model = AchillesLoadModel()
    left_res = model.compute(left_t)
    right_res = model.compute(right_t)

    print(f"\n=== Stage 4: product view ({resolved} data) ===")
    print(f"athlete: {subject} at {speed:g} m/s")
    print(f"per-stride peak Achilles force: R {right_res.peak_force_bw:.2f} BW, "
          f"L {left_res.peak_force_bw:.2f} BW")

    plan = SessionSimulator(n_sessions=args.sessions).generate()

    # asymmetry
    asym = AsymmetryAnalyzer(left_res, right_res).analyze(plan)
    print(f"left/right asymmetry: starts {asym.asi_pct[0]:+.1f}%, "
          f"ends {asym.asi_pct[-1]:+.1f}%, peak |ASI| {asym.peak_abs_asi:.1f}%")

    # accumulation (use mean of the two limbs as the athlete's per-stride seed)
    base = 0.5 * (left_res.loading_impulse_ns() + right_res.loading_impulse_ns())
    acc = AccumulationTimeline().compute(base, plan)
    flagged = np.where(acc.acwr > 1.5)[0]
    print(f"accumulation: cumulative load {acc.cumulative[-1]:.0f} (relative units); "
          f"ACWR peak {np.nanmax(acc.acwr):.2f}"
          + (f", watch-zone sessions {list(plan.sessions[flagged])}" if len(flagged) else ""))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    f3 = FIGURES_DIR / "fig3_stage4_asymmetry.png"
    f4 = FIGURES_DIR / "fig4_stage4_accumulation.png"
    plot_stage4_asymmetry(asym.sessions, asym.asi_pct, asym.left_load, asym.right_load, f3)
    plot_stage4_accumulation(acc.sessions, acc.per_session, acc.cumulative, acc.acwr, f4)
    print(f"saved: {f3}\nsaved: {f4}")


if __name__ == "__main__":
    main()
