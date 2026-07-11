"""Real-time streaming demo: stream a held-out athlete's strides through the
surrogate, print the live readout + measured latency, and write a standalone
HTML dashboard for the event.

Usage:
    python scripts/run_realtime_demo.py [--source fukuchi|synthetic]
                                        [--subject RBDS021] [--calib-k 2]
"""
from __future__ import annotations

import argparse

from achilles.config import FIGURES_DIR, REPO_ROOT
from achilles.data.factory import resolve_source
from achilles.product.realtime import build_streaming_demo
from achilles.viz.dashboard import render_realtime_dashboard
from achilles.viz.plots import plot_realtime_snapshot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="fukuchi")
    ap.add_argument("--subject", default=None, help="athlete id; default = hardest-to-model")
    ap.add_argument("--calib-k", type=int, default=2)
    args = ap.parse_args()

    src, resolved = resolve_source(args.source)
    trials = src.load_trials()
    results, s = build_streaming_demo(trials, demo_subject=args.subject, calib_k=args.calib_k)

    print(f"\n=== Real-time streaming demo ({resolved}, athlete {s.subject_id} held out) ===")
    print(f"{'stride':>6} {'side':>4} {'speed':>6} {'peak true':>10} {'peak shown':>11} "
          f"{'latency':>9} {'asym':>7}  status")
    for r in results:
        status = "calibrating" if r.is_calibration_stride else "calibrated"
        asi = "  n/a" if r.asymmetry_pct != r.asymmetry_pct else f"{r.asymmetry_pct:+5.1f}%"
        print(f"{r.index:>6} {r.side:>4} {r.speed_ms:>5.1f}m {r.peak_true_bw:>9.2f} "
              f"{r.peak_pred_bw:>10.2f} {r.latency_ms:>7.2f}ms {asi:>7}  {status}")

    print(f"\nLatency: {s.mean_latency_ms:.2f} ms/stride mean, {s.p95_latency_ms:.2f} ms p95 "
          f"(~{round(s.strides_per_sec):,} strides/s). A running stride lasts ~600-700 ms, so the "
          f"surrogate runs real-time with orders of magnitude of headroom.")
    print(f"Peak-load error on the held-out athlete: {s.peak_mape_uncal:.0f}% uncalibrated "
          f"-> {s.peak_mape_cal:.0f}% after {s.calib_k} onboarding strides. Final L/R asymmetry "
          f"{s.final_asymmetry_pct:+.1f}%.")

    out = render_realtime_dashboard(results, s, REPO_ROOT / "demo" / "realtime_demo.html")
    print(f"\nsaved dashboard: {out}")
    print("Open it in any browser (self-contained, no server).")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_realtime_snapshot(results, s, FIGURES_DIR / "fig14_realtime.png")
    print(f"saved figure: {fig}")


if __name__ == "__main__":
    main()
