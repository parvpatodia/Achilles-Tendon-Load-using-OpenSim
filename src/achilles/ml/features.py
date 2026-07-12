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


def _derived_zone_shares(vgrf_bw: np.ndarray) -> np.ndarray:
    """The modelled CoP-progression proxy: a Gaussian bump per zone along stance.

    Returns (4, n) spatial shares (before gating by total load). This is the
    PROXY the surrogate uses when no real insole is available.
    """
    pos = _stance_position(vgrf_bw)
    bumps = [np.exp(-0.5 * ((pos - _ZONE_CENTERS[z]) / _ZONE_WIDTH) ** 2) for z in ZONE_NAMES]
    return np.vstack(bumps)


def build_features(trial: GaitTrial, use_measured: bool = True) -> np.ndarray:
    """Return a (N_FEATURES, n_samples) array of wearable input channels.

    Channels 1-2 (vertical GRF in BW, ankle angle) are identical either way.
    Channels 3-6 are the four insole zones, formed as ``vGRF_BW * spatial_share``
    so the two arms differ ONLY in where that share comes from:

      * measured (use_measured and trial.measured_zones present): the real
        16->4 Moticon pressure split, normalised to a per-frame spatial share.
      * derived (otherwise): the modelled heel->toe CoP bump.

    Holding vGRF, angle, target, and architecture fixed and swapping only the
    zone source is the controlled measured-vs-derived comparison (Stage C).
    """
    if not trial.has_grf:
        raise ValueError(
            f"trial {trial.trial_id} has no vertical GRF; the surrogate needs a "
            "wearable GRF input. Walking trials are Stage-1 only."
        )
    vgrf_bw = trial.vgrf_bw
    angle = trial.ankle_angle_deg

    if use_measured and trial.measured_zones is not None:
        # real plantar pressure -> per-frame spatial share across the 4 zones
        mz = trial.measured_zones.astype(np.float64)          # (4, n) N/cm^2
        total = mz.sum(axis=0, keepdims=True)
        shares = np.divide(mz, total, out=np.zeros_like(mz), where=total > 1e-6)
    else:
        shares = _derived_zone_shares(vgrf_bw)                # (4, n) modelled bump

    channels = [vgrf_bw, angle]
    for k in range(len(ZONE_NAMES)):
        # zone load = total plantar load gated by that zone's spatial share
        channels.append(vgrf_bw * shares[k])
    return np.vstack(channels).astype(np.float32)
