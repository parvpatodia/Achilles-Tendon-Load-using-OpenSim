# Plantar Load to Achilles Tendon Load

[![tests](https://github.com/parvpatodia/Achilles-Tendon-Load-using-OpenSim/actions/workflows/ci.yml/badge.svg)](https://github.com/parvpatodia/Achilles-Tendon-Load-using-OpenSim/actions/workflows/ci.yml)

**Your insole measures the load *under* the foot. This estimates the load *inside* the Achilles tendon, the quantity that actually drives injury.** One step further into the body, on real walking and running data, with every assumption checked.

---

## 1. The idea in one line

Mirai's self-powered insole measures how hard the foot pushes on the ground. This project takes that same kind of signal, works out how much force the Achilles tendon carries on each step, and turns that into a simple per-athlete load score you can track over time.

> ### Read this first: what this is, and what it is not
> - **This is a feasibility study and a proposed direction**, built to understand the problem and explore one concrete, tractable path end to end. **It is not a validated product**, and it does not claim Mirai should pivot to it.
> - **There is no non-invasive way to measure true Achilles tendon force.** So this is **data reduction**, not measurement: it reproduces the standard biomechanics-lab estimate from a cheap wearable signal. High accuracy means *"the wearable signal carries the same information as the lab calculation,"* **not** *"we measure tendon force accurately."*
> - **The real way to validate this is against injury outcomes, not tendon force.** You never need to measure tendon force to prove a risk indicator; you show the metric tracks who gets hurt (see §7d).
> - The honest worth of this repo is showing **how the problem can be approached and modelled rigorously**, and being clear about every limit.

## 2. How it continues your work

| Your published work | What this adds |
|---|---|
| The insole measures load **under the foot** (Kanabekova 2026; Issabek 2025) | estimates the load **inside the tendon** that this produces |
| Your models **classify** (flatfoot yes/no, 82%) and describe gait signatures | this **predicts a full load curve** (a value for every instant of the step) |
| You study stress and strain in the **sensor material** | this studies stress and strain in the **tissue** (the tendon as a material) |
| Tested on **walking** in rehab patients | runs on **walking too** (42 walkers) and on running, same method |

Your sensor is a material that turns pressure into an electrical signal. The Achilles is a material that turns load into stress and strain. This connects the two: **from the load your sensor sees, to the stress inside the tissue.**

## 2b. Why the Achilles tendon (and not individual muscles)

A deliberate modelling choice, not convenience:

1. **It is what you can actually identify from a wearable.** A joint has more muscles than degrees of freedom, so the ankle moment does **not** uniquely determine individual muscle forces. This is the classic *muscle-redundancy problem*; recovering per-muscle forces needs EMG or optimisation assumptions (Crowninshield & Brand 1981; Erdemir et al. 2007). The **Achilles tendon force is the exception**: it is the common insertion of the calf muscles (gastrocnemius + soleus), so its load equals their *combined* output and follows directly from the ankle moment. From an insole plus IMU, the tendon is the internal load you can compute; individual muscle forces are not.
2. **Tendon load is also a muscle-load measure.** Muscle and tendon act in series, so the Achilles force we compute **is** the aggregate force the calf muscles produce. We report the part that is identifiable, and we are explicit that we estimate tendon stress/strain and aggregate force, **not** individual muscle forces or muscle-tissue strain (those need EMG).
3. **Tendon injury is common, mechanical, and fatigue-driven.** Achilles tendinopathy is the **most common overuse injury of the lower limb** (about 50% lifetime incidence in distance runners). It is a fatigue failure of the tendon *material* under repeated stress, so cumulative tendon stress is the mechanistically correct variable, and the tendon is a clean passive material with a well-characterised stress-strain law.
4. **The Achilles is the closest internal load to your sensor.** Plantar load, then ankle, then Achilles is the shortest, cleanest inference from a plantar-pressure insole.

**Honest scope, and how it maps to Mirai's pilot sports:**
- **Basketball:** highly relevant. Basketball is the leading cause of Achilles ruptures, and the mechanism (explosive push-off from a dorsiflexed foot) is exactly the loading this pipeline captures.
- **Soccer:** the single most common injury is the **hamstring (a muscle)**, which an insole cannot directly see; Achilles rupture is still notable (about 17% lifetime incidence in players). So the Achilles is the right **first** target for a foot sensor, not the whole injury picture.
- Muscle strains need muscle-level (EMG-driven) modelling this does not do. That is the natural next layer, and the code's swappable structure is built for it.

## 3. The pipeline (five simple steps)

![pipeline](figures/fig0_pipeline.png)

1. Start from **plantar load + motion** (what a wearable gives).
2. Get the **ground reaction force** (the push from the ground).
3. Turn that into the **turning effort at the ankle** (the ankle moment).
4. Divide by the tendon's small lever (the **moment arm**, about 5 cm) to get **tendon force**.
5. Convert force to **stress and strain**, then to a **relative load score** tracked over sessions.

The one equation to remember: **tendon force = ankle effort / lever length**. More effort, or a shorter lever, means more force on the tendon.

## 4. The data

Two open datasets from the same lab (BMClab), so the processing is consistent:

- **Running** (Fukuchi 2017, PeerJ): **31 runners**, both legs, three speeds (2.5 / 3.5 / 4.5 m/s), giving **186 step-cycles** after quality checks.
- **Walking** (Fukuchi 2018, PeerJ): **42 adults** (younger and older), treadmill walking across 8 speeds (0.4 to 2.2 m/s), giving **653 step-cycles**. This is the gait mode that matches your rehab/walking cohort.
- **Combined: 73 people, 839 step-cycles.**

Each recording is time-aligned to one step cycle (0 to 100%). From each we read three things: the **ground push**, the **ankle angle**, and the **ankle effort (moment)**, plus the person's mass and height. We dropped 8 running subjects whose files were corrupted (impossible values) rather than let bad data in.

## 5. What we found

### 5a. Tendon load from real data: the numbers match reality
![stage1](figures/fig1_stage1_achilles_load.png)

Three views: force across the step, the resulting stress against the tendon's breaking point, and **where running lands on the tendon's own stress-strain curve** (a real material curve with a soft "toe" region and a stiff region, not a simple spring).

- Peak Achilles force is **about 5 body-weights** on average in running (up to 7), and it **rises with speed**. Both agree with the biomechanics literature.
- Peak stress averages about 60 MPa and reaches about 90 MPa, against a roughly 100 MPa breaking stress. **Running uses about half the tendon's strength on an average stride, and approaches the limit at the top end.** That thin margin is why the Achilles is a common running injury.

### 5b. The same method works across the whole gait spectrum (including your cohort)
![continuum](figures/fig7_walking_vs_running.png)

Running one pipeline on **both** datasets gives a clean, physically correct picture: Achilles load rises smoothly from **about 2.7 body-weights in walking** (your cohort's gait mode) to **about 5 in running**. The method is not tuned for runners; it tracks load correctly from a slow walk to a fast run. Walking values land where the literature says they should (about 2 to 3.5 BW), itself a check that the physics is right.

### 5c. Can a wearable signal recover the internal load? (the ML pipeline, evaluated honestly)
![comparison](figures/fig8_model_comparison.png)

**The question:** can a cheap wearable-style signal reproduce the internal tendon-load curve that the full *lab* physics computes? If yes, you don't need a lab; an insole plus a motion sensor is enough.

**What goes in:** six signals over one step, the vertical ground push, the ankle angle (motion sensor), and **four "insole-zone" channels** standing in for the Big-Toe/Forefoot/Arch/Heel layout. (Honest note: no real pressure map yet, so the four zones are derived from the total push by a documented heel-to-toe rollover.) **What comes out:** the Achilles force curve over the step.

**How we tested it:** 5-fold cross-validation **holding out whole people** (everyone is scored by a model that never saw them), with **95% confidence intervals from a cluster bootstrap that resamples subjects** (steps within a person are correlated, so resampling subjects rather than steps is the honest choice). And we compared against **baselines**, because a high R² means nothing without them:

| Model | R² (full curve) | 95% CI | R² (loaded phase) | Peak-force error |
|---|---|---|---|---|
| mean curve (no-skill floor) | 0.91 | [0.89, 0.93] | 0.69 | 12.6% |
| linear: GRF only | 0.96 | [0.95, 0.97] | 0.86 | 8.4% |
| linear: GRF + ankle angle | 0.98 | [0.97, 0.99] | 0.92 | 8.2% |
| linear: all 6 inputs | 0.98 | [0.98, 0.99] | 0.94 | 7.5% |
| physics-guided CNN | 0.98 | [0.98, 0.99] | 0.94 | 6.4% |

*(R² is how much of the curve's variation the model captures: 1.0 is perfect, 0 is no better than guessing the average. "Loaded-phase R²" is the same score computed only while the foot is on the ground and the tendon is actually loaded. Peak error is how far off the single highest force is.)*

**Reading it the way a reviewer would:**
- The **mean curve alone scores R²=0.91**, because running curves all look similar, so judge skill *above this floor*, not above zero. The foot is in the air (Achilles force near 0) for more than half the step, and predicting those easy zeros pads the full-curve R². So we also report **loaded-phase R²**, the score on just the hard foot-on-ground part, and **peak-force error**, which is what actually matters.
- A **simple linear model reaches R²=0.98** (loaded 0.94, peak error 7.5%). The **CNN ties it** (loaded 0.94, peak 6.4%), and their confidence intervals overlap. So **the neural net does not beat a linear model here.**
- **Each input earns its place:** GRF alone gives loaded R²=0.86, adding ankle angle raises it to 0.92, the full set to 0.94. The model uses the signals, it is not trivially inverting one number.

**The honest verdict:** a **compact linear model is enough**, and it is well suited to **on-device inference** on an insole. The real result is that the wearable signal carries the internal-load curve, not that deep learning is required. We keep the physics-guided CNN because its constraints (force at least 0, force times lever equals the measured effort, bounded loading rate) **guarantee physically valid output** even where it does not raise accuracy, which matters on noisy real-world signals.

**Peak-force agreement (Bland-Altman):**

![peak](figures/fig9_peak_agreement.png)

Predicted versus reference peak force show **negligible bias (-0.02 BW)** and **95% limits of agreement of about 0.93 BW** (around 18% of a typical peak). We show this plainly: there is no systematic error, but per-stride peak prediction still has real spread. This is the clinical-validation view (do two methods *agree*), not just whether they correlate.

**"Isn't this circular, predicting your own formula?"** A fair challenge. The target is the Achilles force the **full inverse-dynamics pipeline** produces, which normally needs lab motion capture *and* force plates. The model reproduces it from a **reduced wearable signal** (vertical push plus ankle angle). That is a real data-reduction result (lab to wearable), and the residual is real: the omitted horizontal forces, centre of pressure, and limb accelerations carry the missing few percent. It is **not** measured tendon load (see the limits below).

### 5d. We quantified the biggest assumption (the lever length)
![sensitivity](figures/fig6_moment_arm_sensitivity.png)

The whole estimate leans on one number, the tendon's lever (moment arm), and the literature spread is wide (4 to 6 cm). Instead of hiding behind one value, we **swept it across that range**: peak force changes by about **40%** from one end to the other. Two improvements address this directly:
- The lever is now **set per person from their height** (a bigger person has a longer lever), so each athlete gets their own value instead of one fixed number.
- We **cross-checked it against a validated model** (next section).

The honest message: the lever is the one thing most worth measuring per athlete (a quick ultrasound), and here is exactly how much it matters.

### 5e. Cross-checked against a validated musculoskeletal model (OpenSim)
![opensim](figures/fig5_stage3_opensim_xcheck.png)

We compared our lever assumption to **OpenSim**, a standard validated model of the human musculoskeletal system, using the real geometry of the three calf muscles that share the Achilles tendon. OpenSim gives a lever of **4.4 to 4.8 cm with the same shape we assumed**. The two force estimates agree in shape and sit within about **16%** of each other, and that gap is entirely the lever difference. So our estimate agrees with a validated model, and the remaining uncertainty is pinned to one measurable number.

### 5f. The product view: a relative score, per athlete, over time
![asymmetry](figures/fig3_stage4_asymmetry.png)
![accumulation](figures/fig4_stage4_accumulation.png)

We deliberately output a **relative load score over time**, not an absolute stress number, because that is what is defensible and useful, and what continuous capture enables.

- **Left/right balance:** the load each leg carries over sessions. In the example, a growing imbalance crosses a 10% watch line, the kind of trend continuous monitoring catches early (and echoes your own asymmetry finding).
- **Build-up over time:** total tendon loading per session, plus a **recent-versus-usual workload ratio** (a fatigue-style indicator). A spike session pushes it past the watch line, then it settles. Framed as a **warning sign, not a prediction.** (Note: the acute:chronic workload ratio is intuitive but statistically contested in the sports-science literature; we use it as a transparent demonstration, not a validated predictor.)

### 5g. Robustness and uncertainty (the real-insole reality check)
![degradation](figures/fig10_input_degradation.png)

The R²=0.98 is on **pristine lab inputs**. The honest question is what happens on insole-grade signals, so we trained on clean data and tested on **progressively degraded inputs**:
- **Sensor noise:** holds to about 0.2x channel-SD noise (loaded R² 0.89), then falls (0.74 at 0.4x).
- **Temporal resolution:** stable to about 4x downsampling, collapses beyond about 8x.
- **ADC resolution:** holds up even at a **3-bit ADC (loaded R²=0.92)**, good news for a cheap low-power insole.

The realistic operating point lives on these curves, not at the clean-data left edge, and matched-noise retraining would recover part of the loss. This is the most important caveat, shown empirically rather than asserted.

![uncertainty](figures/fig11_uncertainty.png)

Every predicted force also carries a **confidence band**: a deep ensemble of networks gives the prediction, and **cross-fold conformal calibration** sets the band width from held-out residual quantiles (distribution-free, with no calibration-on-test leakage). We verify the bands are honest (a nominal X% band should contain X% of unseen truths), and they are: **90% gives 89%, 95% gives 94%** on held-out subjects (largest gap 1%). The 90% band is **about 0.19 BW** wide either side. That is the direct answer to "where are your error bars, and are they calibrated?"

## 6. The figures, at a glance

| File | Shows |
|---|---|
| `fig0_pipeline` | the five-step method |
| `fig1_stage1_achilles_load` | tendon force, stress versus failure, and the material curve |
| `fig7_walking_vs_running` | load across the gait spectrum (walking to running) |
| `fig8_model_comparison` | the model comparison versus baselines, with CIs (the ML headline) |
| `fig9_peak_agreement` | Bland-Altman agreement on peak force |
| `fig2_stage2_pinn` | example predicted versus true curves on unseen people |
| `fig6_moment_arm_sensitivity` | how much the lever assumption matters |
| `fig5_stage3_opensim_xcheck` | our estimate versus a validated model |
| `fig10_input_degradation` | accuracy under insole-grade input corruption |
| `fig11_uncertainty` | per-prediction confidence bands and calibration |
| `fig3` / `fig4` | left/right balance and load build-up over sessions |

## 7. What this is NOT (the limits, stated plainly)

- **Stand-in data.** Public walking/running data stands in for your insole; the four insole zones are derived from the total push, not measured pressure.
- **Relative, not absolute.** Tendon thickness, stiffness, and lever are population averages, so the absolute stress is indicative; the product score is deliberately relative.
- **No internal ground truth.** Tendon load is not directly measured here (it rarely is, even in labs); the OpenSim check is model versus model, not proof.
- **A modest lower bound on force.** We use the *net* ankle effort, so any opposing-muscle (tibialis anterior) co-contraction makes the true Achilles force slightly higher. For *peak* load this is small: at push-off the antagonist is largely quiet and multiarticular confounds at the ankle are around 2 to 7% (Honert & Zelik 2016), so the peak, the number that matters, is only a modest underestimate; co-contraction matters more in early stance. EMG-driven modelling is the honest fix.
- **Healthy gait.** All subjects are healthy. Transfer to patients and to real insole signals is unproven.
- **Sensor interface.** The pipeline's input is *calibrated plantar load*. A TENG insole outputs a voltage, so a voltage-to-load calibration (your reported near-linear response, about 10 V at 20 N) is the bridge this assumes; it is not modelled here.

**What I'd validate next with your data:** fit the lever per athlete; calibrate the insole zones to real pressure; retrain on your insole plus IMU signals (where the physics constraints should start to earn their place); and test the load score against real symptoms in your rehab cohort.

## 7b. Anticipated questions (the short answers)

- **"Why is R² so high, is it inflated?"** Partly: the force is near 0 during swing, so full-curve R² (0.98) flatters. We report the **loaded-phase R² (0.94)** and **peak error (7.5%)**, against a **no-skill floor of 0.91** (stereotyped curves). Judge skill above the floor.
- **"Isn't the target circular, you predict your own formula?"** The target needs full lab inverse dynamics (mocap plus force plates). The model reproduces it from a **reduced wearable signal**; that is a data-reduction result, and the missing few percent live in the horizontal forces, centre of pressure, and accelerations we drop. It is not measured tendon load.
- **"Is the neural net justified?"** No, a **linear model ties it** (overlapping CIs). We recommend the **compact linear model** for on-device use and say so plainly; the CNN is kept only for its physical-validity guarantees.
- **"Only 31 running subjects, is the estimate stable?"** We use subject-wise CV, **cluster-bootstrap CIs** (resampling subjects), and report the **per-subject R² spread**. Small sample size is a stated limit; the CIs quantify it.
- **"Why the physics constraints if they don't raise accuracy?"** They guarantee **physically valid output** (force at least 0, moment-consistent, bounded rate), insurance for noisy real-world signals, not an accuracy claim.
- **"How does this attach to your TENG insole?"** The input is calibrated plantar load plus ankle angle; your sensor's voltage-to-load linearity is the calibration, and an IMU gives the angle. The four zones map to your Big-Toe/Forefoot/Arch/Heel layout.

## 7c. Real-insole readiness (lab input to your hardware)

What each lab input becomes on Mirai's device, and the calibration it needs:

| Lab input here | On the Mirai insole | Calibration / processing needed |
|---|---|---|
| Vertical ground reaction force | sum of the TENG zone signals | per-zone voltage-to-force (your reported ~10 V at 20 N linearity), drift/temperature compensation, per-step baseline |
| Four insole zones | the four TENG regions directly | per-zone calibration; no spatial proxy needed (we only proxy it because public data lacks pressure maps) |
| Ankle angle | the paired IMU (you already use TENG plus IMU) | gyro/accel fusion, sagittal-plane extraction, drift correction |
| Step segmentation (0 to 100%) | insole heel-strike/toe-off timing or IMU | event detection on the live signal |
| Body mass | entered once per user | none |
| Moment arm (the dominant uncertainty) | one-time per-athlete ultrasound, else population value | imaging at onboarding (see the sensitivity analysis for why) |

The model we recommend is a **compact linear map**, so inference is a few multiply-adds per step. It runs on the insole's microcontroller, no cloud needed.

## 7d. How you'd validate this for real (the honest plan)

The obvious objection is "you can't measure true Achilles force, so how do you validate it?" Three answers, in order of importance:

1. **Validate the product against injury *outcomes*, not tendon force.** A risk indicator does not need a force ground truth; it needs to predict who gets hurt. This is exactly how a **credit score** or **blood pressure** is validated: not by measuring some "true" underlying quantity (which is unmeasurable), but by showing across a population that the number predicts the outcome (default, heart attack), with thresholds drawn from outcome data. Here: run the load index prospectively on a cohort (your rehab patients, your club pilots), then test whether it flags the athletes who go on to develop symptoms better than chance. That sidesteps the unmeasurable-force problem entirely and is the only validation that matters commercially. The catch to state plainly: this is a real multi-month cohort study, and injuries are relatively rare events, so it needs decent numbers and careful statistics.
2. **The moment-arm uncertainty mostly cancels in the metric we actually output.** The product is a *relative, per-athlete, over-time* index. The acute:chronic ratio divides one load by another for the *same person*, so their (roughly constant) moment arm cancels; left/right asymmetry compares two legs of the same person, so it cancels too. **The absolute body-weight number is uncertain; the relative trend you act on is stable.** Per-athlete ultrasound pins the absolute value when needed.
3. **A small calibration cohort to adapt to your hardware.** Recruit about 30 to 50 people of varied heights; for each, measure their own moment arm and tendon stiffness by ultrasound, and record them wearing the Mirai insole *while simultaneously* captured in a gait lab. Pretrain on public data, then fine-tune so the model reproduces the lab estimate **from the Mirai signal specifically** (domain adaptation). The lab estimate is the best available reference label, honestly still a model and not true force; the force stays modelled, and the *product* is judged on the outcomes in point 1.

## 8. Run it

```bash
conda env create -f environment.yml && conda activate mirai-demo
pip install -e .
python scripts/download_data.py            # running data (~5 MB)
python scripts/download_data.py --walking   # add walking data (~586 MB, optional)
python scripts/run_all.py                   # every stage + figures
pytest                                       # 42 tests
```
No download: add `--source synthetic` to any stage (clearly-labelled synthetic data). No OpenSim: that one cross-check skips, everything else runs.

## 9. How the code is built

Every stage depends on one small shared interface (a "gait trial"), and the two big assumptions (the tendon's lever, and its stress-strain law) are **swappable pieces**. So changing the data source (synthetic, running, walking, one day a Mirai insole) or any assumption is a one-line change, and you can see exactly how it moves the result.

```
src/achilles/
  config.py        all physical constants + citations, in one place
  data/            gait-trial interface; running, walking, synthetic sources
  biomech/         the lever models, the tendon material law, the load model, sensitivity
  ml/              wearable features, the network, the physics loss, baselines,
                   cross-validation, and the evaluation metrics (CIs, agreement)
  product/         the load score, balance, and build-up over time
  opensim_xcheck/  the validated-model cross-check (optional)
  viz/             the figures
```

*References: Kanabekova et al., Sensors 2026, 26(10):3191; Issabek et al., Adv. Mater. Technol. 2025, 10(6):2401282; Fukuchi et al., PeerJ 2017 (running) and 2018 (walking); OpenSim Gait2392; tendon mechanics: Wren 2001, Maganaris & Paul 2002, LaCroix 2013; workload ratio: Gabbett 2016.*
