import pytest

pytest.importorskip("ray")
pytest.importorskip("ultralytics")

import cv2
import numpy
import torch
from types import SimpleNamespace
from ultralytics.engine.results import Boxes

from toktagger.api.models.ultralytics_detection import video_detection
from toktagger.api.schemas.annotations import VideoBoundingBox
from toktagger.api.schemas.data import ImageData, ImageParams
from toktagger.api.schemas.samples import Sample, ShotData


class FakeDataLoader:
    def __init__(self, frame_image: ImageData):
        self.frame_image = frame_image
        self.calls = []

    def get_sample(self, sample, params):
        self.calls.append((sample, params))
        return self.frame_image


class FakePredictionModel:
    def __init__(self, results=None):
        self.calls = []
        self.results = [SimpleNamespace(boxes=None)] if results is None else results

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


def make_sample() -> Sample:
    return Sample(
        shot_id=30421,
        data=ShotData(protocol="uda", signal_names=["rba"]),
        _id="sample",
        project_id="project",
        validated_annotations=False,
    )


def make_model(data_loader, prediction_model):
    model = object.__new__(video_detection.YoloVideoDetectionModel)
    model._prediction_model = prediction_model
    model.data_loader = data_loader
    model.get_device = lambda: SimpleNamespace(type="cpu")
    return model


def test_iter_sample_frames_stops_at_end_of_video():
    sample = make_sample()
    frames = [
        ImageData(frame=3, values=[1]),
        ImageData(frame=4, values=[2]),
    ]
    calls = []

    def get_sample(iterated_sample, params):
        calls.append((iterated_sample, params))
        if params.frame is None:
            return frames[0]
        if params.frame == 4:
            return frames[1]
        raise FileNotFoundError

    data_loader = SimpleNamespace(get_sample=get_sample)

    assert list(video_detection.iter_sample_frames(data_loader, sample)) == frames
    assert [params.frame for _, params in calls] == [None, 4, 5]
    assert all(params.return_raw for _, params in calls)


def test_build_video_frame_manifest_includes_negative_frames_and_boxes(monkeypatch):
    sample = make_sample()
    frames = [
        ImageData(frame=0, values=[1, 2]),
        ImageData(frame=1, values=[3, 4]),
    ]
    annotations = [
        [
            VideoBoundingBox(
                label="alpha",
                created_by="manual",
                frame=1,
                track_id="track-alpha",
                x_min=10,
                y_min=20,
                width=30,
                height=40,
            ),
            VideoBoundingBox(
                label="beta",
                created_by="manual",
                frame=1,
                track_id="track-beta",
                x_min=2,
                y_min=3,
                width=4,
                height=5,
            ),
        ]
    ]
    monkeypatch.setattr(
        video_detection,
        "iter_sample_frames",
        lambda data_loader, iterated_sample: iter(frames),
    )

    manifest = video_detection.build_video_frame_manifest(
        samples=[sample],
        annotations=annotations,
        class_map={"alpha": 7, "beta": 3},
        data_loader=object(),
    )

    assert len(manifest) == 2
    assert manifest[0].frame == 0
    assert manifest[0].image == bytes([1, 2])
    assert manifest[0].boxes == []
    assert manifest[0].classes == []
    assert manifest[1].frame == 1
    assert manifest[1].image == bytes([3, 4])
    assert manifest[1].boxes == [
        (10.0, 20.0, 40.0, 60.0),
        (2.0, 3.0, 6.0, 8.0),
    ]
    assert manifest[1].classes == [7, 3]


def test_decode_frame_image_returns_bgr_array():
    expected = numpy.array(
        [
            [[1, 2, 3], [10, 20, 30]],
            [[40, 50, 60], [70, 80, 90]],
        ],
        dtype=numpy.uint8,
    )
    encoded_success, encoded = cv2.imencode(".png", expected)
    assert encoded_success

    decoded = video_detection.decode_frame_image(
        ImageData(frame=0, values=list(encoded.tobytes()))
    )

    assert decoded.shape == (2, 2, 3)
    assert decoded.dtype == numpy.uint8
    assert numpy.array_equal(decoded, expected)


def test_predicts_only_requested_video_frame(monkeypatch):
    sample = make_sample()
    frame_image = ImageData(frame=7, values=[0])
    data_loader = FakeDataLoader(frame_image)
    prediction_model = FakePredictionModel()
    model = make_model(data_loader, prediction_model)
    decoded_frames = []

    def decode_frame(frame):
        decoded_frames.append(frame)
        return "decoded"

    monkeypatch.setattr(video_detection, "decode_frame_image", decode_frame)

    predictions = model.predict(
        [sample],
        video_detection.YoloPredictParams(this_frame_only=True),
        ImageParams(name="image", frame=7, return_raw=False),
    )

    assert predictions == [[]]
    assert len(data_loader.calls) == 1
    loaded_sample, loaded_params = data_loader.calls[0]
    assert loaded_sample is sample
    assert loaded_params.frame == 7
    assert loaded_params.return_raw is True
    assert decoded_frames == [frame_image]
    assert len(prediction_model.calls) == 1


def test_this_frame_only_is_ignored_without_data_params(monkeypatch):
    sample = make_sample()
    data_loader = object()
    prediction_model = FakePredictionModel()
    model = make_model(data_loader, prediction_model)
    frames = [ImageData(frame=0, values=[0]), ImageData(frame=1, values=[0])]
    iterator_calls = []
    decoded_frames = []

    def iter_frames(loader, iterated_sample):
        iterator_calls.append((loader, iterated_sample))
        return iter(frames)

    def decode_frame(frame):
        decoded_frames.append(frame)
        return "decoded"

    monkeypatch.setattr(video_detection, "iter_sample_frames", iter_frames)
    monkeypatch.setattr(video_detection, "decode_frame_image", decode_frame)

    predictions = model.predict(
        [sample],
        video_detection.YoloPredictParams(this_frame_only=True),
        data_params=None,
    )

    assert predictions == [[]]
    assert iterator_calls == [(data_loader, sample)]
    assert decoded_frames == frames
    assert len(prediction_model.calls) == 2


def test_predict_converts_yolo_boxes_to_video_annotations(monkeypatch):
    sample = make_sample()
    frame_image = ImageData(frame=5, values=[0])
    boxes = Boxes(
        torch.tensor(
            [
                [10.2, 20.1, 30.8, 50.9, 0.8, 0],
                [3.1, 4.2, 15.2, 24.3, 0.35, 1],
            ]
        ),
        orig_shape=(100, 100),
    )
    prediction_model = FakePredictionModel(
        [SimpleNamespace(boxes=boxes, names={0: "alpha", 1: "beta"})]
    )
    data_loader = object()
    model = make_model(data_loader, prediction_model)
    monkeypatch.setattr(
        video_detection,
        "iter_sample_frames",
        lambda loader, iterated_sample: iter((frame_image,)),
    )
    monkeypatch.setattr(
        video_detection,
        "decode_frame_image",
        lambda frame: numpy.zeros((100, 100, 3), dtype=numpy.uint8),
    )

    predictions = model.predict(
        [sample],
        video_detection.YoloPredictParams(),
    )

    assert len(predictions) == 1
    assert all(
        isinstance(annotation, VideoBoundingBox) for annotation in predictions[0]
    )
    assert [annotation.label for annotation in predictions[0]] == ["alpha", "beta"]
    assert [
        (annotation.x_min, annotation.y_min, annotation.width, annotation.height)
        for annotation in predictions[0]
    ] == [(10, 20, 21, 31), (3, 4, 12, 20)]
    assert [annotation.frame for annotation in predictions[0]] == [5, 5]
    assert [annotation.uncertainty for annotation in predictions[0]] == pytest.approx(
        [0.2, 0.65]
    )
    assert [annotation.created_by for annotation in predictions[0]] == [
        "yolo_ufo",
        "yolo_ufo",
    ]
    assert [annotation.validated for annotation in predictions[0]] == [
        False,
        False,
    ]
    assert [annotation.track_id for annotation in predictions[0]] == [
        "pred-30421-5-0",
        "pred-30421-5-1",
    ]


@pytest.mark.parametrize(
    ("available_weights", "expected_weight"),
    [
        (("best.pt", "last.pt"), "best.pt"),
        (("last.pt",), "last.pt"),
    ],
)
def test_wrapped_load_restores_best_weights(
    tmp_path,
    monkeypatch,
    available_weights,
    expected_weight,
):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    for filename in available_weights:
        (weights_dir / filename).touch()

    sentinel_model = object()
    loaded_paths = []
    monkeypatch.setattr(
        video_detection,
        "YOLO",
        lambda path: loaded_paths.append(path) or sentinel_model,
    )

    model = object.__new__(video_detection.YoloVideoDetectionModel)
    model._trained = False
    model.wrapped_load(tmp_path)

    expected_path = weights_dir / expected_weight
    assert loaded_paths == [str(expected_path)]
    assert model._trained_weights_path == expected_path
    assert model._prediction_model is sentinel_model
    assert model._trained is True
