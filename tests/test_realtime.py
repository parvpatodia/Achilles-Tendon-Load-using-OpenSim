"""Tests for the real-time streaming inference engine."""
import numpy as np

from achilles.data.synthetic import SyntheticGaitSource
from achilles.ml.baselines import RidgeSequenceModel
from achilles.ml.dataset import AchillesSequenceDataset, build_samples
from achilles.biomech.achilles import AchillesLoadModel
from achilles.ml.features import build_features
from achilles.product.realtime import StreamingAchilles, build_streaming_demo


def _trials(n=12):
    return SyntheticGaitSource(n_subjects=n, seed=0).load_trials()


def test_stream_runs_and_calibrates():
    results, summary = build_streaming_demo(_trials(), calib_k=2)
    assert summary.n_strides == len(results) > 2
    # first K strides are the calibration/onboarding strides, the rest are live
    assert [r.is_calibration_stride for r in results[:2]] == [True, True]
    assert all(not r.is_calibration_stride for r in results[2:])
    assert all(r.calibrated for r in results[2:])


def test_latency_is_measured_and_realtime():
    _, summary = build_streaming_demo(_trials(), calib_k=2)
    assert summary.mean_latency_ms > 0 and np.isfinite(summary.mean_latency_ms)
    # a running stride lasts ~0.6 s; inference must be far faster than that
    assert summary.mean_latency_ms < 100.0
    assert summary.strides_per_sec > 10.0


def test_engine_inference_matches_batch():
    # The streaming engine's per-stride output must equal a batch prediction on
    # the same standardised features (guards against a standardisation mismatch).
    trials = [t for t in _trials() if t.has_grf]
    load = AchillesLoadModel()
    ds = AchillesSequenceDataset(build_samples(trials, load))
    X = np.stack([ds[i]["x"].numpy() for i in range(len(ds))])
    Y = np.stack([ds[i]["y"].numpy() for i in range(len(ds))])
    model = RidgeSequenceModel(channels=None).fit(X, Y)
    engine = StreamingAchilles(model, ds.feat_mean, ds.feat_std, load_model=load)

    t = trials[0]
    raw, _ = engine.infer(t)
    xs = (build_features(t) - ds.feat_mean) / ds.feat_std
    batch = np.clip(model.predict(xs[None, ...])[0], 0.0, None)
    assert np.allclose(raw, batch, atol=1e-5)


def test_asymmetry_and_peaks_finite():
    results, summary = build_streaming_demo(_trials(), calib_k=2)
    assert np.isfinite(summary.final_asymmetry_pct)
    assert np.isfinite(summary.peak_mape_cal)
    for r in results:
        assert r.peak_pred_bw >= 0 and np.isfinite(r.peak_pred_bw)
        assert r.pred_curve.min() >= 0.0                     # tendon force never negative
