"""Tests for the analytical Achilles load core (the MVP's scientific spine)."""
import numpy as np
import pytest

from achilles.biomech.achilles import AchillesLoadModel
from achilles.biomech.moment_arm import (AngleDependentMomentArm, ConstantMomentArm)
from achilles.config import MOMENT_ARM, TENDON
from achilles.data.trial import GaitTrial


def _toy_trial(peak_moment_nm_per_kg=2.5, mass=70.0):
    """A single half-sine plantarflexion-moment stance, neutral ankle angle."""
    phase = np.linspace(0, 100, 101)
    s = np.clip(phase / 40.0, 0, 1)  # stance 0-40%
    moment = peak_moment_nm_per_kg * np.sin(np.pi * s)
    moment[phase > 40] = 0.0
    vgrf = np.clip(20.0 * np.sin(np.pi * s), 0, None)
    vgrf[phase > 40] = 0.0
    return GaitTrial(
        subject_id="T1", side="R", speed_ms=3.5, body_mass_kg=mass,
        gait_phase=phase, ankle_angle_deg=np.zeros(101),
        ankle_moment_nm_per_kg=moment, vgrf_n_per_kg=vgrf, source="toy",
    )


def test_force_equals_moment_over_arm():
    """F = M / r must hold exactly for a constant moment arm."""
    trial = _toy_trial()
    r = 0.05
    res = AchillesLoadModel(moment_arm=ConstantMomentArm(r)).compute(trial)
    expected = np.clip(trial.ankle_moment_nm, 0, None) / r
    np.testing.assert_allclose(res.force_n, expected, rtol=1e-9)


def test_peak_force_in_literature_range():
    """A 2.5 Nm/kg peak moment should give a running-plausible peak force."""
    res = AchillesLoadModel(moment_arm=ConstantMomentArm(0.05)).compute(_toy_trial(2.5, 70))
    assert 3.0 <= res.peak_force_bw <= 7.5


def test_dorsiflexion_moment_does_not_load_tendon():
    """Negative (dorsiflexor) net moment must produce zero Achilles force."""
    trial = _toy_trial()
    trial = GaitTrial(
        subject_id="T", side="R", speed_ms=3.5, body_mass_kg=70,
        gait_phase=trial.gait_phase, ankle_angle_deg=trial.ankle_angle_deg,
        ankle_moment_nm_per_kg=-np.abs(trial.ankle_moment_nm_per_kg),
        vgrf_n_per_kg=trial.vgrf_n_per_kg, source="toy",
    )
    res = AchillesLoadModel().compute(trial)
    assert res.peak_force_n == pytest.approx(0.0, abs=1e-9)


def test_stress_consistent_and_linear_strain():
    from achilles.biomech.tendon import LinearTendon
    res = AchillesLoadModel(moment_arm=ConstantMomentArm(0.05),
                            material=LinearTendon()).compute(_toy_trial())
    np.testing.assert_allclose(res.stress_pa, res.force_n / TENDON.csa_m2, rtol=1e-9)
    np.testing.assert_allclose(res.strain, res.stress_pa / TENDON.linear_modulus_pa, rtol=1e-9)


def test_angle_dependent_moment_arm_clamped():
    ma = AngleDependentMomentArm()
    r = ma.moment_arm_m(np.array([-60.0, 0.0, 60.0]))
    assert np.all(r >= MOMENT_ARM.min_m) and np.all(r <= MOMENT_ARM.max_m)


def test_force_nonnegative_everywhere():
    res = AchillesLoadModel().compute(_toy_trial())
    assert np.all(res.force_n >= 0)
