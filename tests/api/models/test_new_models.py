"""Unit tests for new model features: multivariate, params, NMS, backward compat.

These tests bypass Ray by constructing model instances with object.__new__
and injecting a mocked data_loader. This lets us test the model logic itself
without requiring a running Ray cluster.
"""

import pickle
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from toktagger.api.models.dtw_motif import DTWMotifModel, DTWMotifTrainParams
from toktagger.api.models.event_detection_utils import zscore
from toktagger.api.models.minirocket import MiniRocketModel, MiniRocketTrainParams
from toktagger.api.models.stumpy_motif import (
    StumpyMotifModel,
    StumpyMotifPredictParams,
    StumpyMotifTrainParams,
)
from toktagger.api.schemas.annotations import AnnotationBase, TimeRegion
from toktagger.api.schemas.data import MultiVariateTimeSeriesData, TimeSeriesData

pytestmark = pytest.mark.models_enabled

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_model_instance(cls):
    """Construct a Model subclass without the Ray-dependent __init__."""
    inst = object.__new__(cls)
    inst.id = "test_model_id"
    inst.project = MagicMock()
    inst.model = inst.define_model()
    inst.data_loader = MagicMock()
    inst._trained = False
    # Prevent log_progress from making real HTTP requests (API_URL may be set in CI)
    inst.log_progress = MagicMock()
    return inst


def make_mv_data(
    signal_names: list[str], n: int = 500, seed: int = 0
) -> MultiVariateTimeSeriesData:
    """Return MultiVariateTimeSeriesData with reproducible Gaussian signals."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 10, n).tolist()
    return MultiVariateTimeSeriesData(
        values={
            name: TimeSeriesData(time=time, values=rng.standard_normal(n).tolist())
            for name in signal_names
        }
    )


def make_annotation(t0: float, t1: float, label: str = "Event") -> TimeRegion:
    return TimeRegion(
        time_min=t0,
        time_max=t1,
        label=label,
        validated=True,
        uncertainty=0.0,
        created_by="manual",
    )


def make_sample() -> MagicMock:
    s = MagicMock()
    s.id = "sample_id"
    s.shot_id = 1
    return s


# ---------------------------------------------------------------------------
# DTW Motif
# ---------------------------------------------------------------------------


def _make_trained_dtw_motif(
    signal_names: list[str], window_size: int = 50
) -> DTWMotifModel:
    model = make_model_instance(DTWMotifModel)
    data = make_mv_data(signal_names, n=500)
    model.data_loader.get_sample.return_value = data
    sample = make_sample()
    ann = make_annotation(2.0, 3.0)
    params = DTWMotifTrainParams(
        signal_names=signal_names,
        threshold=5.0,
        window_size=window_size,
    )
    model.train([sample], [[ann]], params)
    model._trained = True
    return model


def test_dtw_motif_train_returns_score():
    model = make_model_instance(DTWMotifModel)
    data = make_mv_data(["Ip"], n=500)
    model.data_loader.get_sample.return_value = data
    sample = make_sample()
    ann = make_annotation(2.0, 3.0)
    params = DTWMotifTrainParams(signal_names=["Ip"], window_size=50)
    score = model.train([sample], [[ann]], params)
    assert isinstance(score, float)


def test_dtw_motif_window_size_param_stored():
    model = _make_trained_dtw_motif(["Ip"], window_size=42)
    assert model.model["window_size"] == 42


def test_dtw_motif_predict_returns_annotation_lists():
    model = _make_trained_dtw_motif(["Ip"])
    sample = make_sample()
    result = model.predict([sample])
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert all(isinstance(a, AnnotationBase) for a in result[0])


def test_dtw_motif_multivariate_train_predict():
    signal_names = ["Ip", "dalpha"]
    model = _make_trained_dtw_motif(signal_names)
    sample = make_sample()
    result = model.predict([sample])
    assert len(result) == 1
    assert isinstance(result[0], list)


def test_dtw_motif_backward_compat_load():
    model = make_model_instance(DTWMotifModel)
    old_state = {
        "signal_name": "Ip",
        "templates": [],
        "window_size": 100,
        "threshold": 3.0,
    }
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(old_state, f)
        path = f.name
    model.load(path)
    assert model.model["signal_names"] == ["Ip"]
    assert "signal_name" not in model.model or model.model.get("signal_names")


# ---------------------------------------------------------------------------
# STUMPY Motif
# ---------------------------------------------------------------------------


def _make_trained_stumpy_motif(
    signal_names: list[str], threshold: float = 5.0
) -> StumpyMotifModel:
    model = make_model_instance(StumpyMotifModel)
    data = make_mv_data(signal_names, n=500)
    model.data_loader.get_sample.return_value = data
    sample = make_sample()
    ann = make_annotation(2.0, 3.0)
    params = StumpyMotifTrainParams(signal_names=signal_names, threshold=threshold)
    model.train([sample], [[ann]], params)
    model._trained = True
    return model


def test_stumpy_motif_train_returns_score():
    model = make_model_instance(StumpyMotifModel)
    data = make_mv_data(["Ip"], n=500)
    model.data_loader.get_sample.return_value = data
    sample = make_sample()
    ann = make_annotation(2.0, 3.0)
    params = StumpyMotifTrainParams(signal_names=["Ip"], threshold=3.0)
    score = model.train([sample], [[ann]], params)
    assert score == 100.0


def test_stumpy_motif_predict_returns_annotation_lists():
    model = _make_trained_stumpy_motif(["Ip"])
    sample = make_sample()
    result = model.predict([sample])
    assert isinstance(result, list)
    assert len(result) == 1
    assert all(isinstance(a, AnnotationBase) for a in result[0])


def test_stumpy_motif_predict_threshold_controls_detection_extent():
    """A stricter (lower) threshold should detect a smaller extent of the
    signal than a more permissive (higher) threshold, for the same trained
    template — using controlled data with a known, separated match/mismatch
    rather than relying on merge behaviour to coincidentally produce equal
    counts."""
    window_size = 30
    n = 300
    time = np.linspace(0, 10, n).tolist()
    rng = np.random.default_rng(0)

    background = rng.standard_normal(n)
    template_raw = rng.standard_normal(window_size)

    # Inject an exact copy of the template at a known location so its MASS
    # distance from the trained template is ~0, while the surrounding
    # background (unrelated noise) has a much larger distance.
    inject_start = 150
    values = background.copy()
    values[inject_start : inject_start + window_size] = template_raw

    model = make_model_instance(StumpyMotifModel)
    model.model = {
        "templates": [(zscore(template_raw), "Event")],
        "window_size": window_size,
        "signal_names": ["Ip"],
        "threshold": 1.0,
    }
    model._trained = True
    data = MultiVariateTimeSeriesData(
        values={"Ip": TimeSeriesData(time=time, values=values.tolist())}
    )
    model.data_loader.get_sample.return_value = data

    sample = make_sample()

    strict_result = model.predict([sample], StumpyMotifPredictParams(threshold=1.0))
    permissive_result = model.predict(
        [sample], StumpyMotifPredictParams(threshold=50.0)
    )

    def total_span(regions):
        return sum(r.time_max - r.time_min for r in regions)

    assert len(strict_result[0]) >= 1
    assert total_span(permissive_result[0]) > total_span(strict_result[0])


def test_stumpy_motif_predict_uses_training_threshold_by_default():
    """predict() with no params should give the same result as passing the trained threshold explicitly."""
    training_threshold = 2.5
    model = _make_trained_stumpy_motif(["Ip"], threshold=training_threshold)
    sample = make_sample()
    default_result = model.predict([sample])
    explicit_result = model.predict(
        [sample], StumpyMotifPredictParams(threshold=training_threshold)
    )
    assert len(default_result[0]) == len(explicit_result[0])


def test_stumpy_motif_multivariate_train_predict_uses_second_channel():
    """Detection should depend on the second channel's content, not just
    the first. Ip carries independent noise at train vs. predict time (no
    self-matching artifact possible), while dalpha carries a distinctive
    pulse at the same location both times — so a bug that ignored the
    second channel would fail to localise the detection correctly."""
    n = 500
    time = np.linspace(0, 10, n).tolist()
    ann_start_idx = int(np.searchsorted(time, 2.0))
    ann_end_idx = int(np.searchsorted(time, 3.0))
    pulse = np.sin(np.linspace(0, 3 * np.pi, ann_end_idx - ann_start_idx))

    def make_data(ip_seed: int) -> MultiVariateTimeSeriesData:
        ip_values = np.random.default_rng(ip_seed).standard_normal(n)
        dalpha_values = 0.01 * np.random.default_rng(99).standard_normal(n)
        dalpha_values[ann_start_idx:ann_end_idx] = pulse
        return MultiVariateTimeSeriesData(
            values={
                "Ip": TimeSeriesData(time=time, values=ip_values.tolist()),
                "dalpha": TimeSeriesData(time=time, values=dalpha_values.tolist()),
            }
        )

    train_data = make_data(ip_seed=1)
    predict_data = make_data(ip_seed=2)

    model = make_model_instance(StumpyMotifModel)
    model.data_loader.get_sample.side_effect = [train_data, predict_data]
    sample = make_sample()
    ann = make_annotation(2.0, 3.0)
    # Averaging Ip's mismatched-noise distance (~9-10) with dalpha's near-0
    # match brings the combined distance to ~5; background elsewhere
    # averages ~9-10 on both channels, so 7.0 only catches the true match.
    params = StumpyMotifTrainParams(signal_names=["Ip", "dalpha"], threshold=7.0)
    model.train([sample], [[ann]], params)
    model._trained = True

    result = model.predict([sample])
    assert len(result) == 1
    assert len(result[0]) >= 1
    assert any(r.time_min <= 3.0 and r.time_max >= 2.0 for r in result[0])


def test_stumpy_motif_detect_handles_signal_length_equal_to_window_size():
    """A signal exactly as long as the window still yields one valid
    comparison and should not be rejected outright."""
    model = make_model_instance(StumpyMotifModel)
    window_size = 20
    rng = np.random.default_rng(0)
    pattern = rng.standard_normal(window_size)
    model.model = {
        "templates": [(zscore(pattern), "Event")],
        "window_size": window_size,
        "signal_names": ["Ip"],
        "threshold": 1.0,
    }
    time_array = np.linspace(0, 1, window_size)

    result = model._detect(pattern, time_array)
    assert len(result) == 1


def test_stumpy_motif_backward_compat_load():
    model = make_model_instance(StumpyMotifModel)
    old_state = {
        "signal_name": "Ip",
        "templates": [],
        "window_size": 100,
        "threshold": 3.0,
    }
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(old_state, f)
        path = f.name
    model.load(path)
    assert model.model["signal_names"] == ["Ip"]


# ---------------------------------------------------------------------------
# MiniRocket
# ---------------------------------------------------------------------------


def _make_trained_minirocket(signal_names: list[str]) -> MiniRocketModel:
    model = make_model_instance(MiniRocketModel)
    data = make_mv_data(signal_names, n=500)
    model.data_loader.get_sample.return_value = data
    sample = make_sample()
    ann = make_annotation(2.0, 3.0)
    params = MiniRocketTrainParams(
        signal_names=signal_names,
        n_background_per_shot=5,
        num_kernels=100,
    )
    model.train([sample], [[ann]], params)
    model._trained = True
    return model


def test_minirocket_train_predict_single_channel():
    model = _make_trained_minirocket(["Ip"])
    sample = make_sample()
    result = model.predict([sample])
    assert isinstance(result, list)
    assert len(result) == 1
    assert all(isinstance(a, AnnotationBase) for a in result[0])


def test_minirocket_nms_is_called_during_predict():
    model = _make_trained_minirocket(["Ip"])
    sample = make_sample()
    with patch(
        "toktagger.api.models.minirocket.non_max_suppression",
        wraps=lambda x: x,
    ) as mock_nms:
        model.predict([sample])
    mock_nms.assert_called_once()


def test_minirocket_multivariate_train_predict():
    model = _make_trained_minirocket(["Ip", "dalpha"])
    sample = make_sample()
    result = model.predict([sample])
    assert len(result) == 1
    assert isinstance(result[0], list)


def test_minirocket_backward_compat_load():
    model = make_model_instance(MiniRocketModel)
    old_state = {
        "signal_name": "Ip",
        "transformer": None,
        "classifier": None,
        "window_size": 100,
        "pos_label": "Event",
    }
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(old_state, f)
        path = f.name
    model.load(path)
    assert model.model["signal_names"] == ["Ip"]


# ---------------------------------------------------------------------------
# Shapelet Transform
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sktime():
    return pytest.importorskip("sktime")


def test_shapelet_train_predict(sktime):
    from toktagger.api.models.shapelet import (
        ShapeletTrainParams,
        ShapeletTransformModel,
    )

    model = make_model_instance(ShapeletTransformModel)
    data = make_mv_data(["Ip"], n=300)
    model.data_loader.get_sample.return_value = data
    sample = make_sample()
    ann = make_annotation(2.0, 3.0)
    params = ShapeletTrainParams(
        signal_names=["Ip"],
        n_background_per_shot=5,
        max_shapelets=2,
        n_shapelet_samples=20,
        batch_size=10,
    )
    score = model.train([sample], [[ann]], params)
    assert isinstance(score, float)

    model._trained = True
    result = model.predict([sample])
    assert isinstance(result, list)
    assert len(result) == 1
    assert all(isinstance(a, AnnotationBase) for a in result[0])
