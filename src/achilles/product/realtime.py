"""Real-time streaming inference: the Achilles surrogate as a live monitor.

The batch stages answer "is the model accurate?". This answers "can it run
live?". It replays an athlete's strides one at a time and, for each completed
stride, turns the wearable-style signal into an Achilles tendon-load curve, a
peak in body weights, and a running left/right asymmetry, timing the inference
so the real-time claim is measured, not asserted.

The recommended compact linear surrogate is a few multiply-adds per stride
(README 8, 10), so per-stride latency is well under a millisecond on a laptop:
real-time with orders of magnitude of headroom, since a running stride lasts
~600-700 ms. The first K strides calibrate to the athlete (subject-specific
calibration, once, at onboarding); every stride after is shown calibrated.

Honesty: this is a REPLAY of real recorded gait cycles, not a live sensor feed
(there is no Mirai device stream here). The model output is genuine per stride;
what is simulated is the arrival of strides over time. On the real device the
same engine consumes the live insole signal in place of the replayed one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from achilles.biomech.achilles import AchillesLoadModel
from achilles.data.trial import GaitTrial
from achilles.ml.baselines import RidgeSequenceModel, SequenceModel
from achilles.ml.calibration import fit_affine, fit_identity, fit_offset
from achilles.ml.cross_val import compare_models_kfold
from achilles.ml.dataset import AchillesSequenceDataset, build_samples
from achilles.ml.features import build_features

_FITTERS = {"identity": fit_identity, "offset": fit_offset, "affine": fit_affine}


@dataclass
class StrideResult:
    index: int
    side: str
    speed_ms: float
    phase: np.ndarray
    true_curve: np.ndarray          # reference Achilles force (BW)
    pred_curve: np.ndarray          # shown prediction (calibrated once active)
    raw_pred_curve: np.ndarray      # prediction before calibration
    peak_true_bw: float
    peak_pred_bw: float
    calibrated: bool
    is_calibration_stride: bool
    latency_ms: float
    asymmetry_pct: float            # running L/R peak asymmetry after this stride


@dataclass
class StreamSummary:
    subject_id: str
    n_strides: int
    calib_k: int
    calib_method: str
    mean_latency_ms: float
    p95_latency_ms: float
    strides_per_sec: float
    peak_mape_uncal: float          # post-warmup strides, raw prediction
    peak_mape_cal: float            # post-warmup strides, calibrated prediction
    final_asymmetry_pct: float


class StreamingAchilles:
    """Stateful per-stride Achilles estimator with on-the-fly calibration."""

    def __init__(self, model: SequenceModel, feat_mean, feat_std,
                 load_model: AchillesLoadModel | None = None,
                 calib_method: str = "affine"):
        self.model = model
        self.feat_mean = np.asarray(feat_mean, dtype=np.float32)
        self.feat_std = np.asarray(feat_std, dtype=np.float32)
        self.load_model = load_model or AchillesLoadModel()
        self._fit = _FITTERS[calib_method]
        self.calib_method = calib_method
        self._ab = (1.0, 0.0)
        self.calibrated = False
        self._cal_true: list[np.ndarray] = []
        self._cal_pred: list[np.ndarray] = []

    def infer(self, trial: GaitTrial) -> tuple[np.ndarray, float]:
        """Genuine per-stride inference (featurise -> standardise -> predict).
        Returns (raw prediction in BW, latency in ms)."""
        t0 = time.perf_counter()
        x = build_features(trial)
        xs = (x - self.feat_mean) / self.feat_std
        raw = self.model.predict(xs[None, ...])[0]
        raw = np.clip(raw, 0.0, None)             # tendon force >= 0
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return raw, latency_ms

    def observe_calibration(self, true_curve: np.ndarray, raw_pred: np.ndarray) -> None:
        self._cal_true.append(np.asarray(true_curve))
        self._cal_pred.append(np.asarray(raw_pred))

    def finalize_calibration(self) -> None:
        self._ab = self._fit(self._cal_true, self._cal_pred)
        self.calibrated = True

    def apply(self, raw_pred: np.ndarray) -> np.ndarray:
        a, b = self._ab
        return np.clip(a * raw_pred + b, 0.0, None)


def _asymmetry(side_peaks: dict[str, list[float]]) -> float:
    """Running L/R asymmetry from mean peak load per side (+ve = right higher)."""
    lft = np.mean(side_peaks["L"]) if side_peaks["L"] else np.nan
    rgt = np.mean(side_peaks["R"]) if side_peaks["R"] else np.nan
    if not (np.isfinite(lft) and np.isfinite(rgt)):
        return float("nan")
    mean = 0.5 * (lft + rgt)
    return float(100.0 * (rgt - lft) / mean) if mean > 0 else float("nan")


def run_stream(engine: StreamingAchilles, stream: list[GaitTrial], calib_k: int) -> list[StrideResult]:
    """Replay strides through the engine: calibrate on the first K, then stream."""
    results: list[StrideResult] = []
    side_peaks: dict[str, list[float]] = {"L": [], "R": []}
    for i, trial in enumerate(stream):
        raw, latency = engine.infer(trial)
        ref = engine.load_model.compute(trial).force_bw
        is_cal = i < calib_k
        if is_cal:
            engine.observe_calibration(ref, raw)
            pred = raw
            if i == calib_k - 1:
                engine.finalize_calibration()
        else:
            pred = engine.apply(raw)
        side_peaks[trial.side].append(float(np.max(pred)))
        results.append(StrideResult(
            index=i, side=trial.side, speed_ms=trial.speed_ms, phase=trial.gait_phase,
            true_curve=ref, pred_curve=pred, raw_pred_curve=raw,
            peak_true_bw=float(np.max(ref)), peak_pred_bw=float(np.max(pred)),
            calibrated=engine.calibrated and not is_cal, is_calibration_stride=is_cal,
            latency_ms=latency, asymmetry_pct=_asymmetry(side_peaks),
        ))
    return results


def _order_stream(demo_trials: list[GaitTrial]) -> list[GaitTrial]:
    # Alternate L/R across increasing speed, so it plays like a run and both
    # sides appear early (the asymmetry needs both limbs).
    return sorted(demo_trials, key=lambda t: (t.speed_ms, t.side))


def _summarize(subject: str, results: list[StrideResult], calib_k: int, method: str) -> StreamSummary:
    post = [r for r in results if not r.is_calibration_stride]
    lat = np.array([r.latency_ms for r in results])

    def _mape(peak_of) -> float:
        e = [abs(peak_of(r) - r.peak_true_bw) / r.peak_true_bw for r in post if r.peak_true_bw > 0]
        return float(np.mean(e) * 100) if e else float("nan")

    return StreamSummary(
        subject_id=subject, n_strides=len(results), calib_k=calib_k, calib_method=method,
        mean_latency_ms=float(lat.mean()), p95_latency_ms=float(np.percentile(lat, 95)),
        strides_per_sec=float(1000.0 / lat.mean()) if lat.mean() > 0 else float("inf"),
        peak_mape_uncal=_mape(lambda r: float(np.max(r.raw_pred_curve))),
        peak_mape_cal=_mape(lambda r: r.peak_pred_bw),
        final_asymmetry_pct=results[-1].asymmetry_pct if results else float("nan"),
    )


def _pick_hardest_subject(trials: list[GaitTrial], k_fold: int = 5, seed: int = 0) -> str:
    """The athlete the surrogate models worst (highest uncalibrated peak error),
    from one subject-wise k-fold pass, so the default demo is an honest hard case
    where calibration visibly earns its keep."""
    base = RidgeSequenceModel(channels=None)
    block = compare_models_kfold([t for t in trials if t.has_grf], [base],
                                 k=k_fold, seed=seed, include_cnn=False)[base.name]
    by: dict[str, list[tuple[float, float]]] = {}
    for s, t, p in zip(block["subject_ids"], block["true_curves"], block["pred_curves"]):
        by.setdefault(s, []).append((float(np.max(t)), float(np.max(p))))
    worst_s, worst_e = None, -1.0
    for s, peaks in by.items():
        e = float(np.mean([abs(pp - pt) / pt for pt, pp in peaks if pt > 0]) * 100)
        if e > worst_e:
            worst_e, worst_s = e, s
    return worst_s


def build_streaming_demo(
    trials: list[GaitTrial],
    demo_subject: str | None = None,
    calib_k: int = 2,
    calib_method: str = "affine",
) -> tuple[list[StrideResult], StreamSummary]:
    """Train the recommended linear surrogate on everyone except one athlete,
    then stream that athlete's strides live (calibrating on the first K).

    The demo athlete is held out of training, so this is an honest new-person
    stream, not a fit-and-replay.
    """
    subjects = sorted({t.subject_id for t in trials})
    if demo_subject is None:
        demo_subject = _pick_hardest_subject(trials)
    if demo_subject not in subjects:
        raise ValueError(f"subject {demo_subject!r} not in data (have {len(subjects)})")

    load = AchillesLoadModel()
    train = [t for t in trials if t.subject_id != demo_subject and t.has_grf]
    demo = [t for t in trials if t.subject_id == demo_subject and t.has_grf]
    if len(demo) <= calib_k:
        raise ValueError(f"subject {demo_subject} has {len(demo)} usable trials, need > calib_k={calib_k}")

    train_ds = AchillesSequenceDataset(build_samples(train, load))
    Xtr = np.stack([train_ds[i]["x"].numpy() for i in range(len(train_ds))])
    Ytr = np.stack([train_ds[i]["y"].numpy() for i in range(len(train_ds))])
    model = RidgeSequenceModel(channels=None).fit(Xtr, Ytr)

    engine = StreamingAchilles(model, train_ds.feat_mean, train_ds.feat_std,
                               load_model=load, calib_method=calib_method)
    stream = _order_stream(demo)
    results = run_stream(engine, stream, calib_k)
    return results, _summarize(demo_subject, results, calib_k, calib_method)
