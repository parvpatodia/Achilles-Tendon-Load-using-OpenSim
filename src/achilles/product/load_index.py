"""Stage 4 - the product view: relative, tissue-aware, longitudinal load.

These classes turn per-stride Achilles load into the things a coach or athlete
would actually watch over time:

  TissueLoadIndex    - per-session loading impulse normalised to a rolling
                       baseline (a relative index, not an absolute stress).
  AsymmetryAnalyzer  - left vs right Achilles load and its trend (echoes the
                       gait-asymmetry finding in Kanabekova 2026).
  AccumulationTimeline - simulated multi-session cumulative load + an
                       acute:chronic workload ratio (ACWR) with a watch band.

Everything here is explicitly RELATIVE and the multi-session timeline is
SIMULATED from one real recording per limb: it demonstrates the product logic
that continuous self-powered capture would drive, and is framed as risk
indication, not prediction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from achilles.biomech.achilles import AchillesLoadResult


# --- session simulation ----------------------------------------------------
@dataclass
class SessionPlan:
    sessions: np.ndarray   # 1..N
    volume: np.ndarray     # relative session volume (e.g. distance)
    intensity: np.ndarray  # relative session intensity (e.g. pace/effort)


class SessionSimulator:
    """Generate a realistic training block (volume build + a deliberate spike).

    The spike session pushes the acute:chronic ratio into the watch zone so the
    risk-indication band is demonstrably doing something. Deterministic (seeded).
    """

    def __init__(self, n_sessions: int = 14, seed: int = 1, spike_session: int = 11,
                 spike_mult: float = 2.8):
        self.n = n_sessions
        self.seed = seed
        self.spike_session = spike_session
        self.spike_mult = spike_mult

    def generate(self) -> SessionPlan:
        rng = np.random.default_rng(self.seed)
        sessions = np.arange(1, self.n + 1)
        # gradual volume build (~5% per session) with session-to-session noise
        volume = 1.0 + 0.05 * (sessions - 1) + rng.normal(0, 0.06, self.n)
        # a deliberate overload spike (and a smaller echo the next session)
        if 1 <= self.spike_session <= self.n:
            volume[self.spike_session - 1] *= self.spike_mult
            if self.spike_session < self.n:
                volume[self.spike_session] *= 1.25
        intensity = 1.0 + rng.normal(0, 0.08, self.n)
        return SessionPlan(sessions, np.clip(volume, 0.3, None),
                           np.clip(intensity, 0.5, None))


# --- per-limb session load -------------------------------------------------
def _session_loads(base_impulse: float, plan: SessionPlan) -> np.ndarray:
    """Per-session loading impulse = base stride impulse x volume x intensity."""
    return base_impulse * plan.volume * plan.intensity


@dataclass
class AsymmetryResult:
    sessions: np.ndarray
    left_load: np.ndarray
    right_load: np.ndarray
    asi_pct: np.ndarray  # +ve = right higher

    @property
    def peak_abs_asi(self) -> float:
        return float(np.max(np.abs(self.asi_pct)))


class AsymmetryAnalyzer:
    """Left/right Achilles load asymmetry over simulated sessions.

    Seeded from one real bilateral recording; an optional drift simulates an
    asymmetry developing over a block (the kind of trend continuous monitoring
    would catch).
    """

    def __init__(self, left: AchillesLoadResult, right: AchillesLoadResult, seed: int = 2):
        self.left_base = left.cyclic_load_index()
        self.right_base = right.cyclic_load_index()
        self.rng = np.random.default_rng(seed)

    def analyze(self, plan: SessionPlan, drift_pct_per_session: float = 1.5) -> AsymmetryResult:
        left = _session_loads(self.left_base, plan)
        right = _session_loads(self.right_base, plan)
        # independent per-limb biological noise so the asymmetry index is not a
        # perfectly smooth line (each limb varies session to session on its own).
        left = left * (1 + self.rng.normal(0, 0.03, len(left)))
        right = right * (1 + self.rng.normal(0, 0.03, len(right)))
        # Simulate overuse drift that AMPLIFIES the already-dominant limb (the
        # athlete progressively favours their stronger side). The asymmetry
        # therefore grows over the block and crosses the watch threshold,
        # demonstrating what longitudinal monitoring would surface.
        drift = 1.0 + (drift_pct_per_session / 100.0) * (plan.sessions - 1)
        if self.right_base >= self.left_base:
            right = right * drift
        else:
            left = left * drift
        mean = 0.5 * (left + right)
        asi = 100.0 * (right - left) / mean
        return AsymmetryResult(plan.sessions, left, right, asi)


@dataclass
class AccumulationResult:
    sessions: np.ndarray
    per_session: np.ndarray
    cumulative: np.ndarray
    relative_index: np.ndarray  # per-session load / rolling chronic baseline
    acwr: np.ndarray


class AccumulationTimeline:
    """Cumulative tissue load and an ACWR-style acute:chronic ratio.

    ACWR: acute = mean load over the last `acute_window` sessions; chronic =
    mean over the last `chronic_window`. Ratios ~0.8-1.3 are the commonly cited
    sweet spot; >1.5 is flagged. REF: Gabbett 2016 (acute:chronic workload).
    """

    def __init__(self, acute_window: int = 2, chronic_window: int = 8):
        self.acute_window = acute_window
        self.chronic_window = chronic_window

    @staticmethod
    def _trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
        out = np.empty_like(x, dtype=float)
        for i in range(len(x)):
            out[i] = x[max(0, i - w + 1): i + 1].mean()
        return out

    def compute(self, base_impulse: float, plan: SessionPlan) -> AccumulationResult:
        per = _session_loads(base_impulse, plan)
        cumulative = np.cumsum(per)
        acute = self._trailing_mean(per, self.acute_window)
        chronic = self._trailing_mean(per, self.chronic_window)
        acwr = acute / np.where(chronic == 0, np.nan, chronic)
        relative_index = per / chronic
        return AccumulationResult(plan.sessions, per, cumulative,
                                  relative_index, acwr)
