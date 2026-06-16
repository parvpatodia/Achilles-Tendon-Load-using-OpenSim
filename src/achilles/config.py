"""Central configuration: physical constants and pipeline parameters.

All domain constants live here (single source of truth). Grep this module
before adding any constant elsewhere.

Citations are inline so a reviewer can check every assumed number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Repo paths -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
FIGURES_DIR = REPO_ROOT / "figures"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

GRAVITY_M_S2 = 9.81  # standard gravity, used to convert mass-normalised GRF -> body weights


@dataclass(frozen=True)
class TendonProperties:
    """Material/geometry properties of the Achilles tendon.

    These are population averages with wide inter-subject spread. They are
    parameters, not facts about any individual, and the demo treats absolute
    stress as indicative only.
    """

    # Cross-sectional area. Adult Achilles CSA ranges ~50-90 mm^2.
    # REF: Magnusson & Kjaer 2003; Kongsgaard et al. 2005.
    csa_m2: float = 60e-6  # 60 mm^2

    # Linear-region (high-strain) elastic modulus ~1-1.5 GPa.
    # NOTE: real tendon has a non-linear "toe" region below ~2% strain; this
    # linear modulus over-stiffens the low-load regime. Used only for a
    # first-order strain estimate. REF: Wren et al. 2001; LaCroix et al. 2013.
    modulus_pa: float = 1.0e9  # 1 GPa

    # Ultimate tensile stress ~50-100 MPa. We use 100 MPa as the failure line.
    # REF: Wren et al. 2001 (ultimate stress ~~ 100 MPa).
    ultimate_stress_pa: float = 100e6  # 100 MPa

    # Typical operating-stress upper bound during running, for the reference band.
    operating_stress_pa: float = 70e6  # 70 MPa


@dataclass(frozen=True)
class MomentArmParams:
    """Parameters for the Achilles tendon moment arm about the ankle.

    The moment arm is the lever through which the tendon force produces the
    ankle plantarflexion moment: F_achilles = M_plantarflexion / r.
    """

    # Constant fallback. Adult Achilles moment arm ~4-6 cm.
    # REF: Rugg et al. 1990; Maganaris et al. 2000.
    constant_m: float = 0.05  # 5 cm

    # Angle-dependent quadratic r(theta) = r0 + c1*theta + c2*theta^2, theta in
    # degrees with dorsiflexion positive. Coefficients approximate the in-vivo
    # trend (moment arm largest near neutral/slight plantarflexion, smaller at
    # extreme dorsiflexion). REF: Maganaris et al. 2000; Rugg et al. 1990.
    r0_m: float = 0.052
    c1_m_per_deg: float = -2.0e-4
    c2_m_per_deg2: float = -3.0e-6
    # Physiological clamp so the estimate never leaves a plausible range.
    min_m: float = 0.035
    max_m: float = 0.060


@dataclass(frozen=True)
class GaitConventions:
    """Column conventions discovered in the Fukuchi 2017 processed files.

    The processed.txt files are time-normalised to 101 samples over the gait
    cycle (0-100%). Joint angles/moments are in the joint coordinate system,
    GRF in the lab frame, so the sagittal axis label differs between them.

    Determined empirically (magnitude + timing) in setup:
      - sagittal ankle angle   -> '<side>ankleAngZ<speed>'  (deg, + dorsiflexion)
      - sagittal ankle moment  -> '<side>ankleMomZ<speed>'  (Nm/kg, + plantarflexion)
      - vertical GRF           -> '<side>grfY<speed>'        (N/kg)
    Moments and GRF are normalised to body MASS.
    """

    n_samples: int = 101
    ankle_angle_axis: str = "Z"
    ankle_moment_axis: str = "Z"
    vgrf_axis: str = "Y"
    speed_map: dict = field(default_factory=lambda: {"25": 2.5, "35": 3.5, "45": 4.5})


# Convenient module-level singletons (immutable, safe to share).
TENDON = TendonProperties()
MOMENT_ARM = MomentArmParams()
CONVENTIONS = GaitConventions()
