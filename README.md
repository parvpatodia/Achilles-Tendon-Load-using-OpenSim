# Plantar Load → Achilles Tendon Load

**A feasibility demo: taking a wearable plantar-load signal one step inward, to internal Achilles tendon load.**

This takes the kind of signal a self-powered insole produces (plantar load + motion) and estimates the *internal* load the Achilles tendon carries during running — tendon force, stress, strain, left/right asymmetry, and a longitudinal load index — on real running data, with an honest account of where it would and would not hold up.

It is built as a continuation of Mirai Tech's published work, one step further into the body:

- **Kanabekova et al. 2026** — *Sensors* 26(10):3191. Self-powered TENG insole, plantar pressure, gait asymmetry, rehabilitation cohort.
- **Issabek et al. 2025** — *Advanced Materials Technologies* 10(6):2401282. TENG insole + ML for flatfoot.

Those papers measure load *at the foot–ground interface*. This demo asks: can that same signal estimate the load *inside the tendon* — the quantity that actually governs injury? The answer here is a qualified yes, and the qualifications are the point.

---

## The pipeline

![pipeline](figures/fig0_pipeline.png)

```
plantar load + motion
  → ground reaction force (GRF)
  → ankle plantarflexion moment        (inverse dynamics)
  → Achilles tendon force   = moment / moment arm
  → tendon stress = force / CSA,  strain = stress / modulus
  → relative, per-athlete, tissue-aware load index (asymmetry + accumulation)
```

---

## What it produces (on real data)

Data: **Fukuchi et al. 2017** treadmill running dataset (PeerJ; figshare `10.6084/m9.figshare.4543435`) — 31 subjects with bilateral 3D kinematics and kinetics at 2.5 / 3.5 / 4.5 m/s (186 limb-trials after quality control).

### Stage 1 — Analytical Achilles load
![stage1](figures/fig1_stage1_achilles_load.png)

| quantity | result | literature |
|---|---|---|
| peak Achilles force | **5.1 BW** mean (2.4–7.2 across subjects/speeds) | ~4–7 BW running |
| peak tendon stress | **59 MPa** mean (up to 92) | ultimate ~100 MPa |
| safety factor to failure stress | **~1.7×** mean | low margin, as expected |
| speed trend (peak force) | 4.79 → 5.19 → 5.38 BW at 2.5 → 3.5 → 4.5 m/s | rises with speed ✓ |

The numbers land where the biomechanics literature says they should, and running loads the tendon to roughly half its ultimate stress — the thin safety margin that makes the Achilles a common running injury.

### Stage 2 — Physics-informed surrogate (held-out subjects)
![stage2](figures/fig2_stage2_pinn.png)

A small temporal CNN (16k parameters) predicts the Achilles force waveform from a **wearable-style input** — vertical GRF + ankle angle + four "insole-zone" channels mimicking Mirai's Big-Toe / Forefoot / Arch / Heel layout — and is scored on **held-out subjects it never saw in training** (subject-wise split, the honest generalisation test).

- **Held-out R² = 0.984, RMSE = 0.19 BW** (8 unseen subjects).
- Robust under stress: with only **3 training subjects + simulated sensor noise**, held-out R² still ≈ 0.91.
- Physics-loss terms (non-negativity, moment-consistency `F·r = M`, bounded loading rate) are documented and enforced. Honestly: on this clean lab data they do **not** measurably change held-out accuracy — the mapping is already learnable from the data term. Their role is as guardrails that keep the output physically valid; their accuracy benefit would show on messier real-insole signals, which I'd test next.

### Stage 3 (bonus) — OpenSim cross-check
![stage3](figures/fig5_stage3_opensim_xcheck.png)

The biggest assumption in Stage 1 is the Achilles moment arm. I cross-checked it against a **validated OpenSim musculoskeletal model** (Gait2392): OpenSim's engine computes the moment arm of medial/lateral gastrocnemius and soleus (the triceps surae, which share the Achilles tendon) across the ankle's range of motion.

- OpenSim moment arm: **4.4–4.8 cm**, decreasing with dorsiflexion — the **same angle dependence** my analytical model assumed (5.0–5.2 cm).
- Resulting Achilles force agrees in shape; magnitude within **~16%**, and that gap is *entirely* the moment-arm difference. This pinpoints the moment arm as the parameter most worth measuring per-athlete.

### Stage 4 — The product view
![asymmetry](figures/fig3_stage4_asymmetry.png)
![accumulation](figures/fig4_stage4_accumulation.png)

The output is deliberately a **relative, tissue-aware, longitudinal load index**, not an absolute stress number:

- **Asymmetry** — left vs right Achilles loading over sessions, echoing the asymmetry finding in Kanabekova 2026. The simulated block shows a developing left-dominant imbalance crossing the ±10% watch threshold.
- **Accumulation** — cumulative tissue load and an acute:chronic workload ratio (ACWR; Gabbett 2016) with a watch band. A deliberate overload session pushes ACWR past 1.5 into the elevated-risk zone, then it recovers.

This is framed as **risk indication, not prediction** — exactly the kind of continuous, longitudinal signal a self-powered insole is uniquely positioned to capture.

---

## What this is NOT (read this)

This is a **feasibility proof of the method**, not a validated product.

- **Proxy data.** Public treadmill running data stands in for Mirai's insole; the four "insole-zone" channels are *derived* from total GRF by a documented centre-of-pressure model, not measured pressure.
- **Analytical moment arm.** Achilles force uses an assumed moment arm (cross-checked against OpenSim, but not subject-specific MRI/ultrasound).
- **Relative, not absolute.** Tendon CSA and modulus are population averages; absolute stress/strain are indicative. The product index is explicitly relative and per-athlete.
- **No in-vivo ground truth.** Internal tendon load is not measured here (it rarely is, even in labs).
- **Generalisation is subject-wise only.** The surrogate is honest about unseen *people*, but all data is healthy treadmill running; transfer to overground, clinical, or Mirai-sensor data is unproven.

**What I'd validate next:** the moment arm per-athlete; the surrogate on real insole signals (where the physics priors should earn their place); and the load index against actual training-load / symptom outcomes.

---

## Run it

```bash
conda env create -f environment.yml      # python 3.11 + stack
conda activate mirai-demo
pip install -e .
python scripts/download_data.py          # ~5 MB from figshare
python scripts/run_all.py                # all stages + figures -> figures/
```

No data download? Every stage takes `--source synthetic` and runs on a clearly-labelled parametric gait model. No OpenSim? Stage 3 skips cleanly; Stages 1/2/4 are unaffected.

```bash
pytest                                   # 17 tests (biomech, data QC, product, loss)
```

---

## Repo structure

```
src/achilles/
  config.py            physical constants + citations (single source of truth)
  data/                GaitTrial value object + GaitDataSource interface
                       (FukuchiDataSource, SyntheticGaitSource, factory)
  biomech/             MomentArmModel strategies + AchillesLoadModel (Stage 1)
  ml/                  wearable features, subject-wise dataset, CNN, physics loss, trainer (Stage 2)
  product/             load index, asymmetry, ACWR accumulation (Stage 4)
  opensim_xcheck/      OpenSim moment-arm cross-check (Stage 3, optional)
  viz/                 publication-style figures
scripts/               download_data.py + run_stage{1..4}.py + run_all.py
tests/                 pytest suite
```

The design point: every stage depends on the small `GaitTrial` / `GaitDataSource` interface, so swapping the data source (synthetic → Fukuchi → one day a Mirai insole export) or the moment-arm model (constant → angle-dependent → OpenSim) is a one-line change, and its effect on the result is auditable.

---

*Built as a feasibility study extending Mirai Tech's TENG-insole work. Public running data; analytical and validated-model estimates; honest about every assumption.*
