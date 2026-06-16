"""Publication-style figures. Each function saves a PNG and returns its path.

Plotting is kept separate from computation: these functions receive already
computed result objects/arrays and only render.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from achilles.config import TENDON
from achilles.viz.style import (ACCENT, ACCENT2, GOOD, INK, MUTED, SPEED_COLORS,
                                WARN, apply_house_style)


def _ensemble(results, attr):
    """Stack a per-sample attribute across results -> (n_results, n_samples)."""
    return np.vstack([getattr(r, attr) for r in results])


# --- Stage 1 ---------------------------------------------------------------
def plot_stage1_achilles(results, out_path: Path, title_suffix: str = "") -> Path:
    """Achilles force (BW) and stress over the gait cycle, by running speed,
    with the tendon failure-stress reference lines."""
    apply_house_style()
    from achilles.biomech.tendon import ToeLinearTendon
    phase = results[0].trial.gait_phase
    speeds = sorted({r.trial.speed_ms for r in results})
    tendon = results[0].tendon

    fig, (ax_f, ax_s, ax_m) = plt.subplots(1, 3, figsize=(15, 4.6))

    for sp in speeds:
        rs = [r for r in results if r.trial.speed_ms == sp]
        c = SPEED_COLORS.get(sp, ACCENT)
        f = _ensemble(rs, "force_bw")
        s = _ensemble(rs, "stress_pa") / 1e6
        for ax, data, in ((ax_f, f), (ax_s, s)):
            mean = data.mean(0)
            sd = data.std(0)
            ax.plot(phase, mean, color=c, lw=2.2, label=f"{sp:g} m/s (n={len(rs)})")
            ax.fill_between(phase, mean - sd, mean + sd, color=c, alpha=0.15)

    ax_f.set_title("Tendon force across the stride")
    ax_f.set_xlabel("Gait cycle (%)")
    ax_f.set_ylabel("Tendon force (body weights)")
    ax_f.legend(title="Running speed")
    ax_f.set_xlim(0, 100)

    # failure / operating reference lines on the stress panel
    ult = TENDON.ultimate_stress_pa / 1e6
    op = TENDON.operating_stress_pa / 1e6
    ax_s.axhline(ult, color=WARN, ls="--", lw=1.5)
    ax_s.text(2, ult - 5, f"ultimate stress ~{ult:.0f} MPa", color=WARN, fontsize=9, va="top")
    ax_s.axhspan(op, ult, color=WARN, alpha=0.06)
    ax_s.text(2, op + 1, f"operating ceiling ~{op:.0f} MPa", color=MUTED, fontsize=9)
    ax_s.set_title("Tendon stress vs. failure margin")
    ax_s.set_xlabel("Gait cycle (%)")
    ax_s.set_ylabel("Tendon stress (MPa)")
    ax_s.set_xlim(0, 100)
    ax_s.set_ylim(0, ult * 1.1)

    # third panel: where running sits on the tendon's stress-strain curve
    mat = ToeLinearTendon(tendon)
    rupture_eps = float(mat.strain(np.array([tendon.ultimate_stress_pa]))[0]) * 100
    eps = np.linspace(0, rupture_eps / 100 * 1.05, 200)
    sigma = mat.stress(eps) / 1e6
    ax_m.plot(eps * 100, sigma, color=INK, lw=2.4, zorder=3, label="tendon material law")
    ax_m.axvspan(0, tendon.toe_strain * 100, color=ACCENT, alpha=0.10)
    ax_m.text(tendon.toe_strain * 50, ult * 0.40, "toe\nregion", color=ACCENT,
              fontsize=8, ha="center", va="center")
    # actual running operating points (peak per trial, to keep the cloud readable)
    peak_eps = np.array([r.peak_strain_pct for r in results])
    peak_sig = np.array([np.max(r.stress_pa) / 1e6 for r in results])
    ax_m.scatter(peak_eps, peak_sig, s=22, alpha=0.45, color=ACCENT2,
                 edgecolors="none", zorder=2, label="per-stride peak (running)")
    ax_m.scatter([rupture_eps], [ult], s=90, color=WARN, zorder=4, marker="X",
                 label=f"rupture (~{rupture_eps:.0f}%, ~{ult:.0f} MPa)")
    ax_m.set_title("Tendon constitutive curve (toe + linear)")
    ax_m.set_xlabel("Tendon strain (%)")
    ax_m.set_ylabel("Tendon stress (MPa)")
    ax_m.set_xlim(0, rupture_eps * 1.12)
    ax_m.set_ylim(0, ult * 1.1)
    ax_m.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"Stage 1 - analytical Achilles load from real running data{title_suffix}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --- Stage 2 ---------------------------------------------------------------
def plot_stage2_pinn(phase, true_curves, pred_curves, r2, rmse_bw,
                     out_path: Path, n_show: int = 4, subtitle: str = "") -> Path:
    """Predicted vs true Achilles force on held-out subjects: overlay + scatter."""
    apply_house_style()
    fig, (ax_o, ax_s) = plt.subplots(1, 2, figsize=(12, 4.6))

    idx = np.linspace(0, len(true_curves) - 1, min(n_show, len(true_curves))).astype(int)
    for k, i in enumerate(idx):
        lbl_t = "true (inverse dynamics)" if k == 0 else None
        lbl_p = "predicted (wearable surrogate)" if k == 0 else None
        ax_o.plot(phase, true_curves[i], color=INK, lw=1.8, alpha=0.8, label=lbl_t)
        ax_o.plot(phase, pred_curves[i], color=ACCENT2, lw=1.8, ls="--", label=lbl_p)
    ax_o.set_title("Held-out subjects: predicted vs. true")
    ax_o.set_xlabel("Gait cycle (%)")
    ax_o.set_ylabel("Achilles force (body weights)")
    ax_o.legend()
    ax_o.set_xlim(0, 100)

    t = np.concatenate(true_curves)
    p = np.concatenate(pred_curves)
    lim = [0, max(t.max(), p.max()) * 1.05]
    ax_s.scatter(t, p, s=6, alpha=0.25, color=ACCENT, edgecolors="none")
    ax_s.plot(lim, lim, color=INK, lw=1.2, ls=":")
    ax_s.set_xlim(lim); ax_s.set_ylim(lim)
    ax_s.set_title("Per-sample agreement")
    ax_s.set_xlabel("True force (BW)")
    ax_s.set_ylabel("Predicted force (BW)")
    ax_s.text(0.05, 0.92, f"$R^2$ = {r2:.3f}\nRMSE = {rmse_bw:.2f} BW",
              transform=ax_s.transAxes, fontsize=11, va="top",
              bbox=dict(boxstyle="round", fc="white", ec=MUTED))

    title = "Stage 2 - physics-guided surrogate from a wearable-style input"
    if subtitle:
        fig.suptitle(title + "\n" + subtitle, fontsize=13, fontweight="bold")
    else:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93 if subtitle else 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --- Stage 4: asymmetry ----------------------------------------------------
def plot_stage4_asymmetry(sessions, asi_pct, left_load, right_load,
                          out_path: Path, watch_threshold: float = 10.0) -> Path:
    apply_house_style()
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax_l.plot(sessions, right_load, "-o", color=ACCENT, lw=2, ms=4, label="right limb")
    ax_l.plot(sessions, left_load, "-o", color=ACCENT2, lw=2, ms=4, label="left limb")
    ax_l.set_title("Per-limb Achilles loading impulse")
    ax_l.set_xlabel("Session")
    ax_l.set_ylabel("Loading impulse (relative)")
    ax_l.legend()

    ax_a.axhspan(-watch_threshold, watch_threshold, color=GOOD, alpha=0.12)
    ax_a.axhline(0, color=MUTED, lw=1)
    ax_a.plot(sessions, asi_pct, "-o", color=INK, lw=2, ms=4)
    ax_a.axhline(watch_threshold, color=WARN, ls="--", lw=1.2)
    ax_a.axhline(-watch_threshold, color=WARN, ls="--", lw=1.2)
    lo = min(float(np.min(asi_pct)) - 4, -watch_threshold - 4)
    hi = max(float(np.max(asi_pct)) + 4, watch_threshold + 4)
    ax_a.set_ylim(lo, hi)
    ax_a.text(sessions[len(sessions) // 2], 0, f"symmetric band (±{watch_threshold:.0f}%)",
              color=GOOD, fontsize=9, ha="center", va="center")
    ax_a.text(sessions[0], lo + 1.5, "watch zone (left-dominant)", color=WARN, fontsize=9)
    ax_a.set_title("Left/right asymmetry index")
    ax_a.set_xlabel("Session")
    ax_a.set_ylabel("Asymmetry (%)   + right-dominant / - left-dominant")

    fig.suptitle("Stage 4 - left/right Achilles load asymmetry over sessions",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --- Stage 4: accumulation -------------------------------------------------
def plot_stage4_accumulation(sessions, per_session, cumulative, acwr,
                             out_path: Path, sweet_low=0.8, sweet_high=1.3,
                             danger=1.5) -> Path:
    apply_house_style()
    fig, (ax_c, ax_r) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax_c.bar(sessions, per_session, color=ACCENT, alpha=0.55, label="per-session load")
    ax_c2 = ax_c.twinx()
    ax_c2.plot(sessions, cumulative, "-o", color=INK, lw=2, ms=3, label="cumulative")
    ax_c2.set_ylabel("Cumulative loading exposure (relative)")
    ax_c2.grid(False)
    ax_c.set_title("Cumulative tendon loading exposure")
    ax_c.set_xlabel("Session")
    ax_c.set_ylabel("Per-session load (relative)")

    # Gabbett 2016: ~0.8-1.3 sweet spot, >1.5 elevated injury risk.
    ax_r.axhspan(sweet_low, sweet_high, color=GOOD, alpha=0.12)
    ax_r.plot(sessions, acwr, "-o", color=INK, lw=2, ms=4)
    ax_r.axhline(danger, color=WARN, ls="--", lw=1.3)
    ax_r.text(sessions[0], danger + 0.03, f"elevated-risk zone (>{danger})",
              color=WARN, fontsize=9)
    ax_r.text(sessions[len(sessions) // 2], (sweet_low + sweet_high) / 2,
              "sweet spot 0.8-1.3", color=GOOD, fontsize=9, ha="center", va="center")
    ax_r.set_title("Acute:chronic workload ratio (ACWR)")
    ax_r.set_xlabel("Session")
    ax_r.set_ylabel("Acute / chronic load")
    ax_r.set_ylim(0, max(2.0, float(np.nanmax(acwr)) * 1.1))

    fig.suptitle("Stage 4 - simulated load timeline: a fatigue-style risk indicator (not a prediction)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --- Stage 3 (bonus) -------------------------------------------------------
def plot_stage3_opensim(phase, analytical_bw, opensim_bw, out_path: Path,
                        subject_label: str = "") -> Path:
    apply_house_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(phase, analytical_bw, color=ACCENT, lw=2.4,
            label="analytical (moment / arm)")
    ax.plot(phase, opensim_bw, color=WARN, lw=2.4, ls="--",
            label="OpenSim triceps surae (gastroc + soleus)")
    ax.set_title(f"Stage 3 - OpenSim cross-check {subject_label}".strip())
    ax.set_xlabel("Gait cycle (%)")
    ax.set_ylabel("Achilles / triceps surae force (body weights)")
    ax.set_xlim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --- Walking vs running continuum ------------------------------------------
def plot_walking_vs_running(walk_results, run_results, out_path: Path) -> Path:
    """Peak Achilles force vs gait speed, from slow walking to running, showing
    the same pipeline spans the whole gait spectrum (incl. her walking cohort)."""
    apply_house_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))

    def _xy(results):
        x = np.array([r.trial.speed_ms for r in results])
        y = np.array([r.peak_force_bw for r in results])
        return x, y

    def _binned(x, y, edges):
        cx, cy, cs = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (x >= lo) & (x < hi)
            if mask.sum() >= 3:
                cx.append((lo + hi) / 2); cy.append(y[mask].mean()); cs.append(y[mask].std())
        return np.array(cx), np.array(cy), np.array(cs)

    wx, wy = _xy(walk_results)
    rx, ry = _xy(run_results)
    ax.scatter(wx, wy, s=10, alpha=0.20, color=ACCENT, edgecolors="none")
    ax.scatter(rx, ry, s=14, alpha=0.30, color=ACCENT2, edgecolors="none")

    bx, by, bs = _binned(wx, wy, np.arange(0.3, 2.4, 0.3))
    ax.errorbar(bx, by, yerr=bs, fmt="-o", color=ACCENT, lw=2.2, ms=5, capsize=3,
                label=f"walking (n={len(walk_results)}, her cohort's mode)")
    rbx, rby, rbs = _binned(rx, ry, np.array([2.25, 3.0, 4.0, 4.75]))
    ax.errorbar(rbx, rby, yerr=rbs, fmt="-s", color=ACCENT2, lw=2.2, ms=5, capsize=3,
                label=f"running (n={len(run_results)})")

    ax.axvspan(2.0, 2.5, color=MUTED, alpha=0.10)
    ax.text(2.25, ax.get_ylim()[1] * 0.12, "walk→run\ntransition", color=MUTED,
            fontsize=8, ha="center")
    ax.set_title("One pipeline across the gait spectrum: Achilles load vs. speed",
                 fontweight="bold")
    ax.set_xlabel("Gait speed (m/s)")
    ax.set_ylabel("Peak Achilles tendon force (body weights)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --- Moment-arm sensitivity ------------------------------------------------
def plot_moment_arm_sensitivity(sens, out_path: Path, default_cm: float = 5.2,
                                opensim_cm: float = 4.44) -> Path:
    """Peak Achilles force and stress vs. the assumed moment arm, with the
    literature range, our default, and the OpenSim value marked."""
    apply_house_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sens.arms_cm, sens.peak_force_bw_mean, "-o", color=ACCENT, lw=2.4,
            ms=4, label="peak Achilles force")
    ax.fill_between(sens.arms_cm,
                    sens.peak_force_bw_mean - sens.peak_force_bw_sd,
                    sens.peak_force_bw_mean + sens.peak_force_bw_sd,
                    color=ACCENT, alpha=0.12)
    ax.set_xlabel("Assumed Achilles moment arm (cm)")
    ax.set_ylabel("Peak Achilles force (body weights)", color=ACCENT)
    ax.tick_params(axis="y", labelcolor=ACCENT)

    ax2 = ax.twinx()
    ax2.plot(sens.arms_cm, sens.peak_stress_mpa_mean, "-s", color=ACCENT2, lw=2.0,
             ms=4, label="peak stress")
    ax2.set_ylabel("Peak tendon stress (MPa)", color=ACCENT2)
    ax2.tick_params(axis="y", labelcolor=ACCENT2)
    ax2.grid(False)

    ax.axvline(default_cm, color=INK, ls="--", lw=1.2)
    ax.text(default_cm, ax.get_ylim()[1] * 0.96, " our model (5.2 cm mean)",
            color=INK, fontsize=9)
    ax.axvline(opensim_cm, color=GOOD, ls=":", lw=1.4)
    ax.text(opensim_cm, ax.get_ylim()[1] * 0.86, "OpenSim\n(4.4 cm mean) ", color=GOOD,
            fontsize=9, ha="right")

    ax.set_title("Moment-arm sensitivity: the dominant assumption, quantified",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --- Pipeline diagram for the README --------------------------------------
def plot_pipeline_diagram(out_path: Path) -> Path:
    apply_house_style()
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.axis("off")
    steps = [
        ("Plantar load\n+ motion", "#EAF0F7"),
        ("Ground reaction\nforce (GRF)", "#EAF0F7"),
        ("Ankle plantarflexion\nmoment (inverse dyn.)", "#FDEBD7"),
        ("Achilles tendon\nforce  (M / r)", "#FDEBD7"),
        ("Stress / strain +\nrelative load index", "#E3F0DE"),
    ]
    n = len(steps)
    w, h, gap = 1.7, 1.2, 0.55
    x = 0.2
    for i, (label, fc) in enumerate(steps):
        ax.add_patch(plt.Rectangle((x, 0.4), w, h, fc=fc, ec=INK, lw=1.4,
                                   zorder=2, joinstyle="round"))
        ax.text(x + w / 2, 0.4 + h / 2, label, ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=3)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap, 1.0), xytext=(x + w, 1.0),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8))
        x += w + gap
    ax.set_xlim(0, x)
    ax.set_ylim(0, 2)
    ax.set_title("Pipeline: from a wearable plantar-load signal to internal tendon load",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
