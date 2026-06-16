"""Build a wearable-style input from a gait trial (the PINN's input side).

The surrogate must predict internal Achilles load from signals a *wearable*
could actually produce, not from the inverse-dynamics moment itself. So the
input channels are:

  1. vertical GRF (body weights)   - a calibrated insole approximates total
                                      plantar load.
  2. ankle angle (deg)             - an IMU gives sagittal ankle angle.
  3-6. four "insole zone" channels - Heel / Arch / Forefoot / Big-toe, mimicking
       Mirai's TENG insole layout (Kanabekova 2026; Issabek 2025).

We do not have a true pressure map, so the four zones are derived from the
vertical GRF by a documented centre-of-pressure progression over stance
(heel-strike -> forefoot -> toe-off). This is an explicit PROXY for the insole's
spatial channels, labelled as such everywhere.
"""
from __future__ import annotations

import numpy as np

from achilles.data.trial import GaitTrial

# Centre of each zone's activation along stance (0 = heel-strike, 1 = toe-off).
# Heel dominates early, big toe at push-off. REF for CoP progression: standard
# roll-over gait description (e.g. Cavanagh & Lafortune 1980).
_ZONE_CENTERS = {"heel": 0.12, "arch": 0.40, "forefoot": 0.70, "bigtoe": 0.90}
_ZONE_WIDTH = 0.18
ZONE_NAMES = ("heel", "arch", "forefoot", "bigtoe")
FEATURE_NAMES = ("vGRF_BW", "ankle_angle_deg") + tuple(f"zone_{z}" for z in ZONE_NAMES)
N_FEATURES = len(FEATURE_NAMES)


def _stance_position(vgrf_bw: np.ndarray) -> np.ndarray:
    """Map gait-cycle index to a 0..1 position *within stance* using the GRF.

    Stance is where vertical GRF is non-trivial; we normalise the index within
    that window so the zone bumps sit in the right place regardless of stance
    fraction.
    """
    n = len(vgrf_bw)
    contact = vgrf_bw > 0.05
    pos = np.zeros(n)
    if contact.any():
        idx = np.where(contact)[0]
        lo, hi = idx[0], idx[-1]
        span = max(hi - lo, 1)
        pos = np.clip((np.arange(n) - lo) / span, 0.0, 1.0)
    return pos


def build_features(trial: GaitTrial) -> np.ndarray:
    """Return a (N_FEATURES, n_samples) array of wearable-proxy channels."""
    if not trial.has_grf:
        raise ValueError(
            f"trial {trial.trial_id} has no vertical GRF; the surrogate needs a "
            "wearable GRF input. Walking trials are Stage-1 only."
        )
    vgrf_bw = trial.vgrf_bw
    angle = trial.ankle_angle_deg
    pos = _stance_position(vgrf_bw)

    channels = [vgrf_bw, angle]
    for z in ZONE_NAMES:
        bump = np.exp(-0.5 * ((pos - _ZONE_CENTERS[z]) / _ZONE_WIDTH) ** 2)
        # zone load = total plantar load gated by that zone's spatial window
        channels.append(vgrf_bw * bump)
    return np.vstack(channels).astype(np.float32)
