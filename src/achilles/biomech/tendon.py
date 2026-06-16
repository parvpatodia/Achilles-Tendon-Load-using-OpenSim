"""Tendon constitutive models: the stress <-> strain law (strategy pattern).

Tendon is not a linear spring. Its stress-strain curve has a compliant "toe"
region at low strain (collagen crimp straightening out) followed by a stiff
linear region, then yield and rupture near ~8-10% strain. Modelling that
non-linearity matters when the deliverable is *strain*, because a linear law
badly underestimates strain at the low loads that dominate a stride.

Two interchangeable strategies:
  LinearTendon       strain = stress / E              (transparent baseline)
  ToeLinearTendon    quadratic toe -> linear region   (materials-accurate)

REF: LaCroix et al. 2013 (J Appl Physiol) for the toe-then-linear form;
Wren et al. 2001 and Maganaris & Paul 2002 for moduli and failure stress.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from achilles.config import TENDON, TendonProperties


class TendonMaterialModel(ABC):
    @abstractmethod
    def strain(self, stress_pa: np.ndarray) -> np.ndarray:
        """Strain (-) for a given tensile stress (Pa). Inverse constitutive law."""
        raise NotImplementedError

    @abstractmethod
    def stress(self, strain: np.ndarray) -> np.ndarray:
        """Stress (Pa) for a given strain (-). Forward constitutive law."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class LinearTendon(TendonMaterialModel):
    """Hookean baseline: stress = E * strain."""

    def __init__(self, tendon: TendonProperties = TENDON):
        self.E = tendon.linear_modulus_pa

    def strain(self, stress_pa):
        return np.asarray(stress_pa, dtype=float) / self.E

    def stress(self, strain):
        return np.asarray(strain, dtype=float) * self.E

    @property
    def name(self) -> str:
        return f"linear (E={self.E/1e9:.1f} GPa)"


class ToeLinearTendon(TendonMaterialModel):
    """Quadratic toe region then a linear region, C1-continuous in slope.

    Forward law sigma(eps):
        toe   (eps <= eps_t):  sigma = (E / (2 eps_t)) * eps^2
        linear(eps >  eps_t):  sigma = sigma_t + E * (eps - eps_t)
    The toe stiffens from slope 0 at eps=0 to the linear modulus E at eps_t, so
    the tangent modulus is continuous. sigma_t = E * eps_t / 2 is the stress at
    the toe->linear transition.

    Inverse law eps(sigma) (what Stage 1 needs, stress -> strain):
        sigma <= sigma_t:  eps = sqrt(2 eps_t sigma / E)
        sigma >  sigma_t:  eps = eps_t + (sigma - sigma_t) / E
    """

    def __init__(self, tendon: TendonProperties = TENDON):
        self.E = tendon.linear_modulus_pa
        self.eps_t = tendon.toe_strain
        self.sigma_t = self.E * self.eps_t / 2.0  # transition stress

    def strain(self, stress_pa):
        s = np.asarray(stress_pa, dtype=float)
        s = np.clip(s, 0.0, None)
        toe = np.sqrt(2.0 * self.eps_t * s / self.E)
        lin = self.eps_t + (s - self.sigma_t) / self.E
        return np.where(s <= self.sigma_t, toe, lin)

    def stress(self, strain):
        e = np.asarray(strain, dtype=float)
        e = np.clip(e, 0.0, None)
        toe = (self.E / (2.0 * self.eps_t)) * e**2
        lin = self.sigma_t + self.E * (e - self.eps_t)
        return np.where(e <= self.eps_t, toe, lin)

    @property
    def name(self) -> str:
        return f"toe+linear (E={self.E/1e9:.1f} GPa, toe<{self.eps_t*100:.0f}%)"
