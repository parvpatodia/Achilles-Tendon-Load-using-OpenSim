"""Tests for the tendon constitutive models (the materials-science core)."""
import numpy as np
import pytest

from achilles.biomech.tendon import LinearTendon, ToeLinearTendon
from achilles.config import TENDON


def test_linear_roundtrip():
    m = LinearTendon()
    s = np.array([0.0, 10e6, 50e6])
    np.testing.assert_allclose(m.stress(m.strain(s)), s, rtol=1e-9)


def test_toe_linear_inverse_consistency():
    """strain(stress(eps)) must recover eps across toe and linear regions."""
    m = ToeLinearTendon()
    eps = np.array([0.0, 0.005, 0.02, 0.04, 0.07])  # spans toe and linear
    np.testing.assert_allclose(m.strain(m.stress(eps)), eps, rtol=1e-6, atol=1e-9)


def test_toe_region_more_compliant_than_linear():
    """In the toe region the tendon must be MORE compliant: more strain per
    stress than a Hookean line from the origin would give."""
    toe = ToeLinearTendon()
    lin = LinearTendon()
    low_stress = 5e6  # below the toe->linear transition (sigma_t)
    assert toe.strain(np.array([low_stress]))[0] > lin.strain(np.array([low_stress]))[0]


def test_toe_linear_continuous_at_transition():
    """Stress is continuous at the toe->linear knee and equals E*eps_t/2."""
    m = ToeLinearTendon()
    e = m.eps_t
    h = 1e-7
    below = m.stress(np.array([e - h]))[0]
    above = m.stress(np.array([e + h]))[0]
    assert below == pytest.approx(above, rel=1e-3)  # continuous across the knee
    # transition stress matches the analytic value sigma_t = E*eps_t/2
    assert m.stress(np.array([e]))[0] == pytest.approx(TENDON.linear_modulus_pa * e / 2, rel=1e-6)
    # tangent slope is the linear modulus on both sides of the knee (C1-continuous)
    slope_toe = (m.stress(np.array([e]))[0] - m.stress(np.array([e - h]))[0]) / h
    assert slope_toe == pytest.approx(TENDON.linear_modulus_pa, rel=1e-2)


def test_failure_strain_reached_below_ultimate_stress_order():
    """Sanity: ultimate stress maps to a strain in the rupture range (~8%)."""
    m = ToeLinearTendon()
    eps_at_ultimate = m.strain(np.array([TENDON.ultimate_stress_pa]))[0]
    assert 0.06 <= eps_at_ultimate <= 0.12
