"""Grouvel measured-pressure pipeline, stage 2: OpenSim scale -> IK -> ID ->
Achilles tendon force, from the .trc/.mot produced by grouvel_prep.py.

This computes the physics reference (Achilles load) for the Grouvel dataset the
same way the repo does for Fukuchi, but here we must run the musculoskeletal
pipeline ourselves because Grouvel ships raw motion capture only.

Method (validated on P02, walking):
  * Place a Grouvel-labelled MarkerSet on gait2392: anatomical markers at the
    OpenSim reference locations (opensim-models) and joint-centre markers
    (HJC/KJC/AJC) at the model's body origins, which ARE the joint centres
    (verified: femur 0.396 m, tibia 0.43 m). Thigh/shank wands are dropped
    (they were the largest IK residuals).
  * ScaleTool with measurement-based per-segment scaling from the joint-centre
    distances (femur=HJC->KJC, tibia=KJC->AJC) + pelvis/foot, subject mass.
  * IK on the walking trial (achieves ~1.9 cm marker RMS, research-grade).
  * ID with the force plates as ExternalLoads, each plate assigned to the foot
    whose heel is nearest its centre of pressure.
  * Achilles = plantarflexion moment / moment arm, using the repo's
    angle-dependent moment arm, reported in body weights.

Only stances where a foot is fully on a plate are valid (elsewhere there is no
GRF for that limb and the ID moment is meaningless); callers segment on that.
The plate->foot heuristic (nearest heel to COP over the window) cleanly isolates
a single-foot stance (validated: right foot, plate 1, 2.05 BW clean push-off),
but can misgroup when consecutive plates load the SAME foot; a per-stance /
gait-event based assignment is the refinement for harvesting every stance.

OpenSim-only process (never import ezc3d here; they segfault together).

Usage:
    python scripts/grouvel_opensim.py --prep <dir> --stem <P02_S01_Gait_01> \
        --static <P02_S01_Static_01> --mass 50 --height 1710 --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import opensim as osim

GRAVITY = 9.81

# Grouvel label -> (model body, location) at OpenSim reference locations; JC
# markers sit at body origins (= joint centres).
ANAT = {
    "RASI": ("pelvis", [0.02, 0.03, 0.128]),  "LASI": ("pelvis", [0.02, 0.03, -0.128]),
    "SACR": ("pelvis", [-0.16, 0.04, 0.0]),
    "RKNE": ("femur_r", [0.0, -0.404, 0.05]), "LKNE": ("femur_l", [0.0, -0.404, -0.05]),
    "RANK": ("tibia_r", [-0.005, -0.41, 0.053]), "LANK": ("tibia_l", [-0.005, -0.41, -0.053]),
    "RHEE": ("calcn_r", [-0.02, 0.02, 0.0]),  "LHEE": ("calcn_l", [-0.02, 0.02, 0.0]),
    "RTOE": ("calcn_r", [0.26, 0.005, 0.0]),  "LTOE": ("calcn_l", [0.26, 0.005, 0.0]),
    "RFMH": ("calcn_r", [0.19, 0.005, -0.04]), "LFMH": ("calcn_l", [0.19, 0.005, 0.04]),
}
JC = {"RHJC": ("femur_r", [0, 0, 0]), "LHJC": ("femur_l", [0, 0, 0]),
      "RKJC": ("tibia_r", [0, 0, 0]), "LKJC": ("tibia_l", [0, 0, 0]),
      "RAJC": ("talus_r", [0, 0, 0]), "LAJC": ("talus_l", [0, 0, 0])}


def _col(tab, label):
    c = tab.getDependentColumn(label)
    return np.array([c[i] for i in range(tab.getNumRows())])


def moment_arm_m(theta_deg):
    """Repo AngleDependentMomentArm (config.MomentArmParams), + dorsiflexion."""
    r = 0.052 - 2.0e-4 * theta_deg - 3.0e-6 * theta_deg ** 2
    return np.clip(r, 0.035, 0.060)


def build_scaled_model(model_path, static_trc, mass, height, out_dir):
    model = osim.Model(model_path)
    for nm, (body, loc) in {**ANAT, **JC}.items():
        model.addMarker(osim.Marker(nm, model.getBodySet().get(body), osim.Vec3(*loc)))
    model.finalizeConnections()
    model_markers = str(out_dir / "model_markers.osim")
    model.printToXML(model_markers)

    md = osim.MarkerData(static_trc)
    tr = osim.ArrayDouble(); tr.append(md.getStartFrameTime()); tr.append(md.getLastFrameTime())

    def meas(name, m1, m2, bodies):
        m = osim.Measurement(); m.setName(name)
        m.getMarkerPairSet().cloneAndAppend(osim.MarkerPair(m1, m2))
        for b in bodies:
            bs = osim.BodyScale(); bs.setName(b)
            ax = osim.ArrayStr(); [ax.append(a) for a in ("X", "Y", "Z")]
            bs.setAxisNames(ax); m.getBodyScaleSet().cloneAndAppend(bs)
        m.setApply(True); return m

    st = osim.ScaleTool()
    st.setName("subject"); st.setSubjectMass(mass); st.setSubjectHeight(height)
    gmm = st.getGenericModelMaker(); gmm.setModelFileName(model_markers); gmm.setMarkerSetFileName("Unassigned")
    msr = st.getModelScaler(); msr.setApply(True)
    order = osim.ArrayStr(); order.append("measurements"); msr.setScalingOrder(order)
    mset = osim.MeasurementSet()
    for m in (meas("pelvis", "RASI", "LASI", ["pelvis"]),
              meas("femur_r", "RHJC", "RKJC", ["femur_r"]), meas("femur_l", "LHJC", "LKJC", ["femur_l"]),
              meas("tibia_r", "RKJC", "RAJC", ["tibia_r"]), meas("tibia_l", "LKJC", "LAJC", ["tibia_l"]),
              meas("foot_r", "RHEE", "RTOE", ["talus_r", "calcn_r", "toes_r"]),
              meas("foot_l", "LHEE", "LTOE", ["talus_l", "calcn_l", "toes_l"])):
        mset.cloneAndAppend(m)
    msr.setMeasurementSet(mset)
    msr.setMarkerFileName(static_trc); msr.setTimeRange(tr); msr.setPreserveMassDist(True)
    scaled = str(out_dir / "scaled.osim")
    msr.setOutputModelFileName(scaled)
    st.getMarkerPlacer().setApply(False)  # JC markers are exact; scaling alone gives <2 cm RMS
    st.run()
    return scaled


def run_ik(scaled, gait_trc, out_dir):
    sm = osim.Model(scaled); sm.initSystem()
    ik = osim.InverseKinematicsTool()
    ik.setModel(sm)
    ik.setMarkerDataFileName(gait_trc)
    ik.set_report_errors(True); ik.setResultsDir(str(out_dir))
    ik_mot = str(out_dir / "ik.mot"); ik.set_output_motion_file(ik_mot)
    md = osim.MarkerData(gait_trc)
    ik.setStartTime(md.getStartFrameTime()); ik.setEndTime(md.getLastFrameTime())
    ik.run()
    return ik_mot


def assign_plates(grf_mot, gait_trc, n_plates=3, thresh_n=40.0):
    grf = osim.TimeSeriesTable(grf_mot); gt = np.array(grf.getIndependentColumn())
    trc = osim.TimeSeriesTableVec3(gait_trc); tt = np.array(trc.getIndependentColumn())

    def heel(name, t):
        v = trc.getDependentColumn(name)[int(np.argmin(np.abs(tt - t)))]
        return np.array([v.get(0), v.get(1), v.get(2)])

    out = {}
    for p in range(1, n_plates + 1):
        vy = _col(grf, f"ground_force_{p}_vy")
        if vy.max() < thresh_n:
            continue
        # Assign to the foot whose heel is closest to the COP AVERAGED over the
        # plate's whole active window. Single-frame checks (peak or onset) fail
        # during double support; over the window the swing foot is far, so the
        # mean cleanly identifies the stance foot.
        cop = np.column_stack([_col(grf, f"ground_force_{p}_p{a}") for a in "xyz"])
        idxs = np.where(vy > thresh_n)[0][::10]
        dR = np.mean([np.linalg.norm(cop[k] - heel("RHEE", gt[k])) for k in idxs])
        dL = np.mean([np.linalg.norm(cop[k] - heel("LHEE", gt[k])) for k in idxs])
        out[p] = "calcn_r" if dR < dL else "calcn_l"
    return out


def run_id(scaled, ik_mot, grf_mot, plate_foot, out_dir):
    ext = osim.ExternalLoads(); ext.setDataFileName(grf_mot)
    for p, foot in plate_foot.items():
        ef = osim.ExternalForce(); ef.setName(f"plate{p}")
        ef.set_applied_to_body(foot)
        ef.set_force_expressed_in_body("ground"); ef.set_point_expressed_in_body("ground")
        ef.set_force_identifier(f"ground_force_{p}_v")
        ef.set_point_identifier(f"ground_force_{p}_p")
        ef.set_torque_identifier(f"ground_torque_{p}_")
        ext.cloneAndAppend(ef)
    ext_xml = str(out_dir / "extloads.xml"); ext.printToXML(ext_xml)

    idt = osim.InverseDynamicsTool()
    idt.setModelFileName(scaled); idt.setCoordinatesFileName(ik_mot)
    idt.setLowpassCutoffFrequency(6.0); idt.setExternalLoadsFileName(ext_xml)
    ik = osim.TimeSeriesTable(ik_mot); it = np.array(ik.getIndependentColumn())
    idt.setStartTime(float(it[0])); idt.setEndTime(float(it[-1]))
    idt.setResultsDir(str(out_dir)); idt.setOutputGenForceFileName("id.sto")
    idt.run()
    return str(out_dir / "id.sto")


def achilles_bw(id_sto, ik_mot, mass, side="r"):
    idt = osim.TimeSeriesTable(id_sto); it = np.array(idt.getIndependentColumn())
    ik = osim.TimeSeriesTable(ik_mot); kt = np.array(ik.getIndependentColumn())
    M = _col(idt, [l for l in idt.getColumnLabels() if f"ankle_angle_{side}" in l][0])
    ang = np.interp(it, kt, _col(ik, f"ankle_angle_{side}"))
    pf = np.clip(-M, 0.0, None)                       # plantarflexor moment (stance)
    force_bw = pf / moment_arm_m(ang) / (mass * GRAVITY)
    return it, force_bw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/opensim/gait2392_thelen2003muscle.osim")
    ap.add_argument("--prep", required=True, help="dir with the grouvel_prep .trc/.mot")
    ap.add_argument("--stem", required=True, help="gait trial stem, e.g. P02_S01_Gait_01")
    ap.add_argument("--static", required=True, help="static trial stem, e.g. P02_S01_Static_01")
    ap.add_argument("--mass", type=float, required=True)
    ap.add_argument("--height", type=float, default=1700.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prep, out = Path(args.prep), Path(args.out); out.mkdir(parents=True, exist_ok=True)
    scaled = build_scaled_model(args.model, str(prep / f"{args.static}.trc"), args.mass, args.height, out)
    ik_mot = run_ik(scaled, str(prep / f"{args.stem}.trc"), out)
    grf_mot = str(prep / f"{args.stem}_grf.mot")
    pf = assign_plates(grf_mot, str(prep / f"{args.stem}.trc"))
    print("plate -> foot:", pf)
    id_sto = run_id(scaled, ik_mot, grf_mot, pf, out)
    for side in ("r", "l"):
        t, f = achilles_bw(id_sto, ik_mot, args.mass, side)
        print(f"ankle_{side}: peak Achilles {f.max():.2f} BW at t={t[int(f.argmax())]:.2f}s")


if __name__ == "__main__":
    main()
