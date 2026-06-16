# Plantar Load → Achilles Tendon Load

**Your insole measures load *under* the foot. This estimates the load *inside* the tendon — the quantity that actually drives injury.**

A feasibility demo extending Mirai Tech's self-powered insole work one step inward: from plantar load to internal Achilles tendon force, stress, and strain, on real running data, with the tendon treated as a material and every assumption checked.

---

## Why this is a continuation of your work

| Your published work | This demo |
|---|---|
| TENG insole measures **external** plantar load (Kanabekova 2026; Issabek 2025) | estimates the **internal** tendon load that load produces |
| ML = **classification** (flatfoot, 82% with Random Forest) and asymmetry **signatures** | ML = **regression** of the full internal-load waveform |
| stress/strain on the **sensor material** (TENG: ~linear to 10 V at 20 N) | stress/strain in the **tissue material** (tendon constitutive curve) |
| walking, rehab/trauma cohort | running (highest, best-characterised Achilles loads) — same pipeline applies to walking |

Your sensor is a material that turns mechanical stress into charge. The Achilles is a material that turns load into stress and strain. This connects the two: **from the stress on your sensor to the stress in the tissue.**

---

## The pipeline

![pipeline](figures/fig0_pipeline.png)

`plantar load + motion → ground reaction force → ankle moment (inverse dynamics) → tendon force = moment ÷ moment-arm → stress & strain → relative load index`

Data: **Fukuchi et al. 2017** treadmill-running dataset (PeerJ; 31 subjects, both legs, 2.5/3.5/4.5 m/s, 186 trials after quality control).

---

## The four results

### 1. Tendon load from real data — the numbers are right
![stage1](figures/fig1_stage1_achilles_load.png)

Force across each stride, the resulting stress against the tendon's failure limit, and **where that lands on the tendon's own stress–strain curve** (a real toe-then-linear material law, not a linear spring).

- Peak Achilles force **5.1 body-weights** on average (2.4–7.2), and it **rises with running speed** — both match the biomechanics literature (~4–7 BW).
- Peak stress averages **~60 MPa** and reaches **~90 MPa** in the highest-load runners, against a ~100 MPa failure stress — about **half the tendon's strength on an average stride (~1.7× safety factor)**, and close to the limit at the high end. That thin margin is *why* the Achilles is a top running injury.

### 2. A wearable signal can recover that internal load
![stage2](figures/fig2_stage2_pinn.png)

A small neural network (16k parameters) predicts the internal tendon-load curve from a **wearable-style input only** — vertical load + ankle angle + four "insole-zone" channels matching your Big-Toe / Forefoot / Arch / Heel layout. No lab equipment at inference.

- Tested with **5-fold cross-validation where every subject is held out once** (the honest test on people it never saw): **R² = 0.983 ± 0.004**, error 0.21 BW.
- It is *physics-guided*: the loss enforces tendon force ≥ 0, the moment relation `force × arm = moment`, and a bounded loading rate. Honestly, on this clean lab data those constraints don't change accuracy (we tested — the effect is within fold-to-fold noise); they are physical-validity guardrails that matter more on noisy real-insole data.

### 3. Cross-checked against a validated model (OpenSim)
![stage3](figures/fig5_stage3_opensim_xcheck.png)

The biggest assumption is the tendon's moment arm. A validated OpenSim musculoskeletal model (its real gastrocnemius + soleus geometry) gives a moment arm of **4.4–4.8 cm with the same angle-dependence I assumed (5.2 cm)**. The two force estimates agree in shape and sit within ~16% — and that gap is *entirely* the moment arm, which pinpoints the one thing worth measuring per athlete.

### 4. The product view: relative, per-athlete, over time
![asymmetry](figures/fig3_stage4_asymmetry.png)
![accumulation](figures/fig4_stage4_accumulation.png)

The output is a **relative, longitudinal load index**, not an absolute stress number:

- **Left/right asymmetry** over sessions (echoing your asymmetry finding) — here a developing imbalance crosses a ±10% watch line.
- **Cumulative loading + an acute:chronic workload ratio** (a fatigue-style indicator): an overload session pushes it past the 1.5 risk threshold, then it recovers.

This is **risk indication, not prediction** — exactly the continuous signal a self-powered insole is built to capture.

---

## Honest limitations (the important part)

- **Proxy data.** Public running data stands in for your insole; the four zone channels are *derived* from total load, not measured pressure.
- **Relative, not absolute.** Tendon area, stiffness, and moment arm are population averages; absolute stress is indicative. The product index is deliberately relative.
- **No internal ground truth.** Tendon load is not directly measured here (it rarely is, even in labs); OpenSim is a model-vs-model cross-check, not validation.
- **Healthy running, not your cohort.** All data is healthy treadmill running. Transfer to walking, rehab patients, and real TENG signals is unproven.

**What I'd validate next:** the moment arm per-athlete (ultrasound); the surrogate on your real insole + IMU signals; and the load index against actual training-load and symptom outcomes in your rehab cohort.

---

## Run it

```bash
conda env create -f environment.yml && conda activate mirai-demo
pip install -e .
python scripts/download_data.py     # ~5 MB
python scripts/run_all.py           # all stages + figures
pytest                              # 22 tests
```
No download → add `--source synthetic` (clearly-labelled parametric data). No OpenSim → Stage 3 skips; the rest is unaffected.

**Design:** every stage depends on a small `GaitTrial` / `GaitDataSource` interface, and the moment-arm and tendon-material laws are swappable strategies — so changing the data source (synthetic → Fukuchi → one day a Mirai insole) or any assumption is a one-line change whose effect on the result is auditable.

*References: Kanabekova et al., Sensors 2026, 26(10):3191 · Issabek et al., Adv. Mater. Technol. 2025, 10(6):2401282 · Fukuchi et al., PeerJ 2017 · OpenSim Gait2392 · tendon mechanics: Wren 2001, Maganaris & Paul 2002, LaCroix 2013 · ACWR: Gabbett 2016.*
