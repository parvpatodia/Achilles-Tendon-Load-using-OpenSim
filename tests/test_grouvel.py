"""Tests for the Stage-C measured-pressure infrastructure.

These exercise the pluggable measured-insole path (GaitTrial.measured_zones,
the build_features measured-vs-derived A/B, and GrouvelDataSource) on synthetic
CLEAN stance CSVs in a temp dir, so they are independent of OpenSim and of the
real dataset's signal quality.
"""
import numpy as np
import pandas as pd
import pytest

from achilles.data.grouvel import GrouvelDataSource
from achilles.data.trial import GaitTrial
from achilles.ml.features import ZONE_NAMES, build_features

N = 101


def _clean_trial(measured=True):
    phase = np.linspace(0, 100, N)
    angle = 10 * np.sin(np.pi * phase / 100)             # deg, dorsi/plantar sweep
    moment_per_kg = 1.2 * np.sin(np.pi * phase / 100)    # ~plantarflexion bump
    vgrf_per_kg = 11.0 * np.clip(np.sin(np.pi * phase / 100), 0, None)  # ~1.1 BW
    zones = None
    if measured:
        # a real-looking heel->toe roll-over across the 4 zones
        centers = np.array([0.15, 0.4, 0.7, 0.9])
        zones = np.vstack([np.exp(-0.5 * ((phase / 100 - c) / 0.15) ** 2) for c in centers])
    return GaitTrial(
        subject_id="P02", side="R", speed_ms=1.3, body_mass_kg=70.0,
        gait_phase=phase, ankle_angle_deg=angle,
        ankle_moment_nm_per_kg=moment_per_kg, vgrf_n_per_kg=vgrf_per_kg,
        source="grouvel", task="walk", measured_zones=zones,
    )


def test_measured_zones_shape_validation():
    with pytest.raises(ValueError):
        GaitTrial("P02", "R", 1.3, 70.0, np.zeros(N), np.zeros(N),
                  np.zeros(N), np.zeros(N), measured_zones=np.zeros((3, N)))  # 3 != 4 zones
    with pytest.raises(ValueError):
        GaitTrial("P02", "R", 1.3, 70.0, np.zeros(N), np.zeros(N),
                  np.zeros(N), np.zeros(N), measured_zones=np.zeros((4, N - 1)))  # bad length


def test_features_measured_vs_derived_differ_only_in_zones():
    t = _clean_trial(measured=True)
    x_meas = build_features(t, use_measured=True)
    x_deriv = build_features(t, use_measured=False)
    assert x_meas.shape == x_deriv.shape == (2 + len(ZONE_NAMES), N)
    # vGRF and angle channels are identical; only the 4 zone channels change.
    np.testing.assert_allclose(x_meas[:2], x_deriv[:2], rtol=1e-6)
    assert not np.allclose(x_meas[2:], x_deriv[2:])


def test_features_fall_back_to_derived_without_measured():
    t = _clean_trial(measured=False)
    assert t.measured_zones is None
    # with no measured zones, use_measured=True must equal the derived proxy
    np.testing.assert_allclose(
        build_features(t, use_measured=True), build_features(t, use_measured=False), rtol=1e-6)


def _write_stance(stances_dir, name, moment_per_kg_peak=1.2, mass=70.0):
    phase = np.linspace(0, 100, N)
    bump = np.clip(np.sin(np.pi * phase / 100), 0, None)
    centers = np.array([0.15, 0.4, 0.7, 0.9])
    zones = {f"zone_{z}": np.exp(-0.5 * ((phase / 100 - c) / 0.15) ** 2)
             for z, c in zip(ZONE_NAMES, centers)}
    df = pd.DataFrame({
        "phase": phase,
        "ankle_angle_deg": 10 * bump,
        "ankle_moment_nm": moment_per_kg_peak * mass * bump,   # Nm
        "vgrf_n": 11.0 * mass * bump,
        **zones,
        "insole_total_n": 11.0 * mass * bump,
    })
    df.to_csv(stances_dir / name, index=False)


def _make_source(tmp_path, mass=70.0):
    proc = tmp_path / "grouvel_processed"
    (proc / "stances").mkdir(parents=True)
    pd.DataFrame({"subject": ["P02"], "body_mass_kg": [mass], "height_mm": [1710],
                  "sex": ["F"], "age_y": [22], "has_insole": [1]}).to_csv(
        proc / "subjects.csv", index=False)
    return proc


def test_grouvel_source_reads_stance(tmp_path):
    proc = _make_source(tmp_path)
    _write_stance(proc / "stances", "P02_S01_Gait_01_R1.csv")
    trials = GrouvelDataSource(processed_dir=proc).load_trials()
    assert len(trials) == 1
    t = trials[0]
    assert t.subject_id == "P02" and t.side == "R" and t.source == "grouvel"
    assert t.task == "walk" and t.has_grf
    assert t.measured_zones is not None and t.measured_zones.shape == (4, N)
    assert 0.5 <= t.ankle_moment_nm.max() / 0.035 / t.body_weight_n <= 4.0


def test_grouvel_source_qc_drops_implausible(tmp_path):
    proc = _make_source(tmp_path)
    # peak moment ~50x plausible -> Achilles well over the physiological ceiling
    _write_stance(proc / "stances", "P02_S01_Gait_02_R1.csv", moment_per_kg_peak=60.0)
    assert GrouvelDataSource(processed_dir=proc).load_trials() == []
