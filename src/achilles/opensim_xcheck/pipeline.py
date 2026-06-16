"""Stage 3 (bonus) - OpenSim cross-check of the Achilles moment arm.

The pip OpenSim wheel ships no example motion data, so instead of the full
IK->ID->SO workflow we use a validated lower-limb model (Gait2392) for what it
is uniquely good at here: its real triceps surae geometry. OpenSim's engine
computes the moment arm of medial gastrocnemius, lateral gastrocnemius and
soleus about the ankle across its range of motion. Their common insertion is
the Achilles tendon, so the mean of the three is an OpenSim-grounded estimate
of the Achilles moment arm.

That estimate is wrapped as a MomentArmModel, so it drops into the same
AchillesLoadModel as the analytical strategies. Feeding the *same* measured
ankle moment through OpenSim's moment arm vs. our analytical one is an
apples-to-apples cross-check of the single biggest modelling assumption.

If OpenSim or the model file is unavailable, this module fails only when
instantiated; the rest of the pipeline is unaffected.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from achilles.biomech.moment_arm import MomentArmModel

TRICEPS_SURAE = {"R": ("med_gas_r", "lat_gas_r", "soleus_r"),
                 "L": ("med_gas_l", "lat_gas_l", "soleus_l")}
ANKLE_COORD = {"R": "ankle_angle_r", "L": "ankle_angle_l"}


class OpenSimMomentArmModel(MomentArmModel):
    """Effective Achilles moment arm from an OpenSim model's triceps surae.

    Builds a moment-arm-vs-ankle-angle lookup once, then interpolates. Ankle
    angle convention is reconciled to ours (+ dorsiflexion); the lever is taken
    as the magnitude of OpenSim's signed moment arm.
    """

    def __init__(self, model_path: Path | str, side: str = "R",
                 angle_range_deg: tuple[float, float] = (-35.0, 30.0), n: int = 40):
        import opensim as osim  # guarded import (Stage 3 is optional)

        self.side = side
        self.model_path = str(model_path)
        model = osim.Model(self.model_path)
        state = model.initSystem()
        coord = model.getCoordinateSet().get(ANKLE_COORD[side])
        muscles = [model.getMuscles().get(n_) for n_ in TRICEPS_SURAE[side]]

        angles_deg = np.linspace(angle_range_deg[0], angle_range_deg[1], n)
        per_muscle = np.zeros((len(muscles), n))
        for j, a_deg in enumerate(angles_deg):
            coord.setValue(state, np.deg2rad(a_deg))
            model.realizePosition(state)
            for i, mu in enumerate(muscles):
                per_muscle[i, j] = abs(mu.computeMomentArm(state, coord))

        # effective Achilles moment arm = mean across the three heads (common tendon)
        self._angles_deg = angles_deg
        self._r_lookup = per_muscle.mean(axis=0)
        self.per_muscle = per_muscle
        self.muscle_names = TRICEPS_SURAE[side]

    def moment_arm_m(self, ankle_angle_deg: np.ndarray) -> np.ndarray:
        theta = np.asarray(ankle_angle_deg, dtype=float)
        return np.interp(theta, self._angles_deg, self._r_lookup)

    @property
    def name(self) -> str:
        return "OpenSim Gait2392 triceps surae"

    def summary(self) -> dict:
        return {
            "mean_moment_arm_cm": float(self._r_lookup.mean() * 100),
            "range_cm": (float(self._r_lookup.min() * 100),
                         float(self._r_lookup.max() * 100)),
        }
