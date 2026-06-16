# Plantar Load → Achilles Tendon Load

[![tests](https://github.com/parvpatodia/Achilles-Tendon-Load-using-OpenSim/actions/workflows/ci.yml/badge.svg)](https://github.com/parvpatodia/Achilles-Tendon-Load-using-OpenSim/actions/workflows/ci.yml)

**Your insole measures the load *under* the foot. This estimates the load *inside* the Achilles tendon, which is the quantity that actually drives injury.** One step further into the body, on real walking and running data, with every assumption checked.

---

## 1. The idea in one line

Mirai's self-powered insole measures how hard the foot pushes on the ground. This project takes that same kind of signal and works out how much force the Achilles tendon carries on each step, then turns that into a simple, per-athlete load score you can track over time. It is a proof that the method works, not a finished product, and it is honest about its limits.

## 2. How it continues your work

| Your published work | What this adds |
|---|---|
| The insole measures load **under the foot** (Kanabekova 2026; Issabek 2025) | estimates the load **inside the tendon** that this produces |
| Your models **classify** (flatfoot yes/no, 82%) and describe gait signatures | this **predicts a full load curve** (a number for every instant of the step) |
| You study stress and strain in the **sensor material** | this studies stress and strain in the **tissue** (the tendon as a material) |
| Tested on **walking** in rehab patients | runs on **walking too** (42 walkers) and on running, same method |

Your sensor is a material that turns pressure into an electrical signal. The Achilles is a material that turns load into stress and strain. This connects the two: **from the load your sensor sees, to the stress inside the tissue.**

## 3. The pipeline (five simple steps)

![pipeline](figures/fig0_pipeline.png)

1. Start from **plantar load + motion** (what a wearable gives).
2. Get the **ground reaction force** (the push from the ground).
3. Turn that into the **turning effort at the ankle** (the ankle moment).
4. Divide by the tendon's small lever (the **moment arm**, ~5 cm) to get **tendon force**.
5. Convert force to **stress and strain**, then to a **relative load score** tracked over sessions.

The one equation to remember: **tendon force = ankle effort ÷ lever length**. More effort, or a shorter lever, means more force on the tendon.

## 4. The data

Two open datasets from the same lab (BMClab), so the processing is consistent:

- **Running** (Fukuchi 2017, PeerJ): **31 runners**, both legs, three speeds (2.5 / 3.5 / 4.5 m/s) → **186 step-cycles** after quality checks.
- **Walking** (Fukuchi 2018, PeerJ): **42 adults** (younger and older), treadmill walking across 8 speeds (0.4–2.2 m/s) → **653 step-cycles**. This is the gait mode that matches your rehab/walking cohort.
- **Combined: 73 people, 839 step-cycles.**

Each recording is time-aligned to one step cycle (0–100%). From each we read three things: the **ground push**, the **ankle angle**, and the **ankle effort (moment)**, plus the person's mass and height. We dropped 8 running subjects whose files were corrupted (impossible values), rather than let bad data in.

## 5. What we found

### 5a. Tendon load from real data — the numbers match reality
![stage1](figures/fig1_stage1_achilles_load.png)

Three views: force across the step, the resulting stress against the tendon's breaking point, and **where running lands on the tendon's own stress–strain curve** (a real material curve with a soft "toe" region and a stiff region, not a simple spring).

- Peak Achilles force is **~5 body-weights** on average in running (up to ~7), and it **rises with speed** — both agree with the biomechanics literature.
- Peak stress averages ~60 MPa and reaches ~90 MPa, against a ~100 MPa breaking stress. **Running uses about half the tendon's strength on an average stride, and approaches the limit at the top end.** That thin margin is why the Achilles is a common running injury.

### 5b. The same method works across the whole gait spectrum (including your cohort)
![continuum](figures/fig7_walking_vs_running.png)

Running one pipeline on **both** datasets gives a clean, physically correct picture: Achilles load rises smoothly from **~2.7 body-weights in walking** (your cohort's gait mode) to **~5 in running**. The method is not tuned for runners; it tracks load correctly from a slow walk to a fast run. Walking values land right where the literature says they should (~2–3.5 BW), which is itself a check that the physics is right.

### 5c. A wearable signal can recover the internal load (the ML pipeline)
![stage2](figures/fig2_stage2_pinn.png)

This is the core machine-learning result, and the part most relevant to a device.

**The question:** can a cheap, wearable-style signal reproduce the internal tendon-load curve that the full physics computes? If yes, you don't need a lab; an insole plus a motion sensor is enough.

**What goes in:** six signals over one step, all things a wearable can produce — the vertical ground push, the ankle angle (from a motion sensor), and **four "insole-zone" channels** standing in for your Big-Toe / Forefoot / Arch / Heel layout. (Honest note: we don't have a real pressure map, so those four zones are derived from the total push using a documented heel-to-toe rollover. They show the model can take a four-zone insole signal.)

**What comes out:** the Achilles force curve over the whole step.

**The model:** a small **temporal convolutional network** (a network that slides a shared filter along the time signal), about **16,000 numbers** in total. To be precise about wording: this is a *physics-guided* network, not the classic "PINN" that solves a differential equation. The physics enters as **rules in the training objective**, not as an equation solver. A convolutional network fits because the relationship is local in time and the shared filter generalizes across people with very few parameters.

**The physics rules it must obey** (each is a real fact about tendons):
1. force can never be negative (a tendon pulls, it cannot push),
2. force × lever must equal the measured ankle effort (balance at the joint),
3. the force cannot jump instantly (loading is smooth).

**How we tested it (the honest part):** **5-fold cross-validation, holding out whole people.** We split the runners into 5 groups; each group is, in turn, kept completely out of training and used only for testing. So every person is tested on a model that never saw them. This is the real test for a device that will meet new athletes.

**Result:**

| Metric | Value |
|---|---|
| Accuracy on unseen people (R²) | **0.983 ± 0.002** across the 5 folds |
| Typical error | **0.21 body-weights** (against peaks near 5) |

In plain terms: from a cheap wearable-style signal, the model reproduces the internal load curve on people it never trained on, capturing about **98%** of the variation.

**An honest check we ran:** we retrained with the physics rules switched off. Accuracy was the same (within noise). So on this clean lab data the rules don't *improve* the score; their job is to keep the output physically valid (never negative, never jumpy), which matters more on messy real-world insole data. We report this straight rather than overclaim.

### 5d. We quantified the biggest assumption (the lever length)
![sensitivity](figures/fig6_moment_arm_sensitivity.png)

The whole estimate leans on one number: the tendon's lever (moment arm), and the literature spread is wide (4–6 cm). Instead of hiding behind one value, we **swept it across that range**: peak force changes by about **40%** from one end to the other. Two improvements address this directly:
- The lever is now **set per person from their height** (a bigger person has a longer lever), so each athlete gets their own value instead of one fixed number.
- We **cross-checked it against a validated model** (next section).

This is the honest message: the lever is the one thing most worth measuring per athlete (a quick ultrasound), and here is exactly how much it matters.

### 5e. Cross-checked against a validated musculoskeletal model (OpenSim)
![opensim](figures/fig5_stage3_opensim_xcheck.png)

We compared our lever assumption to **OpenSim**, a standard, validated model of the human musculoskeletal system, using the real geometry of the three calf muscles that share the Achilles tendon. OpenSim gives a lever of **4.4–4.8 cm with the same shape we assumed**. The two force estimates agree in shape and sit within about **16%** of each other, and that gap is entirely the lever difference. So our estimate agrees with a validated model, and the remaining uncertainty is pinned to one measurable number.

### 5f. The product view: a relative score, per athlete, over time
![asymmetry](figures/fig3_stage4_asymmetry.png)
![accumulation](figures/fig4_stage4_accumulation.png)

We deliberately output a **relative load score over time**, not an absolute stress number, because that is what is defensible and useful, and what continuous capture enables.

- **Left/right balance:** the load each leg carries over sessions. In the example, a growing imbalance crosses a 10% watch line — the kind of trend continuous monitoring catches early (and echoes your own asymmetry finding).
- **Build-up over time:** total tendon loading per session, plus a **recent-vs-usual workload ratio** (a fatigue-style indicator). A spike session pushes it past the risk threshold, then it settles. Framed as a **warning sign, not a prediction.**

## 6. The figures, at a glance

| File | Shows |
|---|---|
| `fig0_pipeline` | the five-step method |
| `fig1_stage1_achilles_load` | tendon force, stress vs. failure, and the material curve |
| `fig7_walking_vs_running` | load across the gait spectrum (walking → running) |
| `fig2_stage2_pinn` | the model's prediction vs. truth on unseen people |
| `fig6_moment_arm_sensitivity` | how much the lever assumption matters |
| `fig5_stage3_opensim_xcheck` | our estimate vs. a validated model |
| `fig3` / `fig4` | left/right balance and load build-up over sessions |

## 7. What this is NOT (the limits, stated plainly)

- **Stand-in data.** Public walking/running data stands in for your insole; the four insole zones are derived from the total push, not measured pressure.
- **Relative, not absolute.** Tendon thickness, stiffness, and lever are population averages, so the absolute stress is indicative; the product score is deliberately relative.
- **No internal ground truth.** Tendon load is not directly measured here (it rarely is, even in labs); the OpenSim check is model-vs-model, not proof.
- **A lower bound on force.** We use the *net* ankle effort, which already subtracts any opposing muscle pulling the other way (co-contraction). Real Achilles force is therefore a little higher than we report; the honest fix is muscle-activity (EMG) measurement.
- **Healthy gait.** All subjects are healthy. Transfer to patients and to real insole signals is unproven.

**What I'd validate next with your data:** fit the lever per athlete; calibrate the insole zones to real pressure; retrain the model on your insole + motion signals (where the physics rules should start to earn their place); and test the load score against real symptoms in your rehab cohort.

## 8. Run it

```bash
conda env create -f environment.yml && conda activate mirai-demo
pip install -e .
python scripts/download_data.py            # running data (~5 MB)
python scripts/download_data.py --walking   # add walking data (~586 MB, optional)
python scripts/run_all.py                   # every stage + figures
pytest                                       # 28 tests
```
No download → add `--source synthetic` to any stage (clearly-labelled synthetic data). No OpenSim → that one cross-check skips; everything else runs.

## 9. How the code is built

Every stage depends on one small shared interface (a "gait trial"), and the two big assumptions (the tendon's lever, and its stress–strain law) are **swappable pieces**. So changing the data source (synthetic → running → walking → one day a Mirai insole) or any assumption is a one-line change, and you can see exactly how it moves the result.

```
src/achilles/
  config.py        all physical constants + citations, in one place
  data/            gait-trial interface; running, walking, synthetic sources
  biomech/         the lever models, the tendon material law, the load model, sensitivity
  ml/              wearable features, the network, the physics loss, cross-validation
  product/         the load score, balance, and build-up over time
  opensim_xcheck/  the validated-model cross-check (optional)
  viz/             the figures
```

*References: Kanabekova et al., Sensors 2026, 26(10):3191 · Issabek et al., Adv. Mater. Technol. 2025, 10(6):2401282 · Fukuchi et al., PeerJ 2017 (running) and 2018 (walking) · OpenSim Gait2392 · tendon mechanics: Wren 2001, Maganaris & Paul 2002, LaCroix 2013 · workload ratio: Gabbett 2016.*
