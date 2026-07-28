from __future__ import annotations # store type hints as strings

import logging
from collections.abc import Iterator
from typing import Literal

import cv2
import numpy as np
import pydantic
from ultralytics import YOLO

from toktagger.api.core.data_loaders import (
    DataLoader as TokTaggerDataLoader,
)
from toktagger.api.models.base import ModelRegistry
from toktagger.api.schemas.annotations import (
    Annotation,
    AnnotationBase,
    VideoBoundingBox,
)
from toktagger.api.schemas.data import ImageData, ImageParams
from toktagger.api.schemas.samples import Sample

from .base import BaseUltralyticsDetection, DetectionRecord

logger = logging.getLogger(__name__)

YoloModelName = Literal[
    "yolov8n.pt",
    "yolo11n.pt",
    "yolo26n.pt",
    "yolo26x.pt",
]


class YoloTrainParams(pydantic.BaseModel):
    """Parameters exposed by the model-training form."""

    learning_rate: float = pydantic.Field(
        default=0.003,
        gt=0,
        description="Initial learning rate.",
    )
    epochs: int = pydantic.Field(
        default=2,
        gt=0,
        description="Number of training epochs.",
    )
    yolo_size: YoloModelName = pydantic.Field(
        default="yolo26n.pt",
        description="Pretrained YOLO checkpoint to fine-tune.",
    )


class YoloPredictParams(pydantic.BaseModel):
    """Parameters exposed by the prediction form."""

    confidence_threshold: float = pydantic.Field(
        default=0.2,
        ge=0,
        le=1,
        description="Minimum confidence required for a detection.",
    )
    iou_threshold: float = pydantic.Field(
        default=0.2,
        ge=0,
        le=1,
        description="Intersection-over-union threshold.",
    )
    max_det: int = pydantic.Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of detections per frame.",
    )


def iter_sample_frames(
    data_loader: TokTaggerDataLoader,
    sample: Sample,
) -> Iterator[ImageData]:
    """Yield every contiguous frame in a TokTagger video sample.

    TokTagger's image loader returns the first available frame when ``frame`` is
    ``None``. Version one then requests successive frame numbers until the
    loader reports that the next frame does not exist.
    """
    try:
        frame_image = data_loader.get_sample(
            sample,
            ImageParams(
                name="image",
                frame=None,
                return_raw=True,
            ),
        )
    except FileNotFoundError:
        logger.warning(
            "No frames found for shot %s.",
            sample.shot_id,
        )
        return

    while True:
        yield frame_image

        try:
            frame_image = data_loader.get_sample(
                sample,
                ImageParams(
                    name="image",
                    frame=frame_image.frame + 1,
                    return_raw=True,
                ),
            )
        except FileNotFoundError:
            break


def build_video_frame_manifest(
    samples: list[Sample],
    annotations: list[list[Annotation]],
    class_map: dict[str, int],
    data_loader: TokTaggerDataLoader,
) -> list[DetectionRecord]:
    """
    Convert validated video samples into frame-level training records.
    Hash table has been used to speed up things.
    """
    if len(samples) != len(annotations):
        raise ValueError(
            "Samples and annotations must have the same length."
        )

    frame_manifest: list[DetectionRecord] = []

    for sample, sample_annotations in zip(
        samples,
        annotations,
        strict=True,
    ):
        # A frame can contain several objects. Grouping annotations once gives
        # constant-time lookup while every video frame is loaded.
        annotations_by_frame: dict[int, list[Annotation]] = {}

        for annotation in sample_annotations:
            if (
                getattr(annotation, "type", None)
                != "video_bounding_box"
            ):
                continue

            frame = int(annotation.frame)
            annotations_by_frame.setdefault(
                frame,
                [],
            ).append(annotation)

        sample_record_count = 0

        for frame_image in iter_sample_frames(
            data_loader,
            sample,
        ):
            if isinstance(frame_image.values, str):
                raise TypeError(
                    "Expected raw image bytes but received a base64 string."
                )

            frame = int(frame_image.frame)
            frame_annotations = annotations_by_frame.get(
                frame,
                [],
            )

            boxes: list[list[float]] = []
            classes: list[int] = []
            labels: list[str] = []
            track_ids: list[str | None] = []

            for annotation in frame_annotations:
                if annotation.label not in class_map:
                    logger.warning(
                        "Skipping unknown label %s in shot %s frame %s.",
                        annotation.label,
                        sample.shot_id,
                        frame,
                    )
                    continue

                x1 = float(annotation.x_min)
                y1 = float(annotation.y_min)
                x2 = x1 + float(annotation.width)
                y2 = y1 + float(annotation.height)

                boxes.append([x1, y1, x2, y2])
                classes.append(class_map[annotation.label])
                labels.append(annotation.label)
                track_ids.append(
                    getattr(annotation, "track_id", None)
                )

            frame_manifest.append(
                {
                    "shot_id": int(sample.shot_id),
                    "frame": frame,
                    # ImageData stores raw encoded bytes as a JSON-compatible
                    # list of integers. Convert it back into bytes here.
                    "image": bytes(frame_image.values),
                    "boxes": boxes,
                    "classes": classes,
                    "labels": labels,
                    "track_ids": track_ids,
                }
            )
            sample_record_count += 1

        logger.info(
            "Added %s frames from shot %s to the manifest.",
            sample_record_count,
            sample.shot_id,
        )

    logger.info(
        "Created a video manifest containing %s frames.",
        len(frame_manifest),
    )

    return frame_manifest


def decode_frame_image(frame_image: ImageData) -> np.ndarray:
    """Decode raw TokTagger image bytes for Ultralytics prediction."""
    if isinstance(frame_image.values, str):
        raise TypeError(
            "Expected raw image bytes but received a base64 string."
        )

    encoded_image = np.frombuffer(
        bytes(frame_image.values),
        dtype=np.uint8,
    )
    image = cv2.imdecode(
        encoded_image,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Could not decode frame {frame_image.frame}."
        )

    return image


@ModelRegistry.register(
    "yolo_ufo",
    ["video"],
    YoloTrainParams,
    YoloPredictParams,
)
class YoloVideoDetectionModel(BaseUltralyticsDetection):
    """YOLO video bounding-box detector for TokTagger."""

    model_name = "yolo26n.pt"
    class_map = {"UFO": 0}

    def build_manifest(
        self,
        samples: list[Sample],
        annotations: list[list[Annotation]],
    ) -> list[DetectionRecord]:
        """Build the in-memory video training manifest."""
        return build_video_frame_manifest(
            samples=samples,
            annotations=annotations,
            class_map=self.class_map,
            data_loader=self.data_loader,
        )

    def predict(
        self,
        samples: list[Sample],
        params: YoloPredictParams | None = None,
        data_params=None,
    ) -> list[list[AnnotationBase]]:
        """Predict bounding boxes for every frame in each sample."""
        del data_params

        if params is None:
            params = YoloPredictParams()

        weights_path = self.get_prediction_weights_path()
        model = YOLO(weights_path)

        all_predictions: list[list[AnnotationBase]] = []

        for sample in samples:
            sample_predictions: list[AnnotationBase] = []

            for frame_image in iter_sample_frames(
                self.data_loader,
                sample,
            ):
                image = decode_frame_image(frame_image)

                results = model.predict(
                    source=image,
                    conf=params.confidence_threshold,
                    iou=params.iou_threshold,
                    max_det=params.max_det,
                    device=self.get_device().type,
                    verbose=False,
                )

                if not results:
                    continue

                result = results[0]

                if result.boxes is None or len(result.boxes) == 0:
                    continue

                coordinates = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()

                for detection_index, (
                    box,
                    confidence,
                    class_id,
                ) in enumerate(
                    zip(
                        coordinates,
                        confidences,
                        class_ids,
                        strict=True,
                    )
                ):
                    x1, y1, x2, y2 = box

                    width = max(
                        0,
                        int(round(x2 - x1)),
                    )
                    height = max(
                        0,
                        int(round(y2 - y1)),
                    )

                    if width == 0 or height == 0:
                        continue

                    label = self.class_names.get(
                        int(class_id),
                        "UFO",
                    )

                    sample_predictions.append(
                        VideoBoundingBox(
                            label=label,
                            created_by=self.type or "yolo_ufo",
                            validated=False,
                            uncertainty=max(
                                0,
                                min(
                                    1,
                                    1 - float(confidence),
                                ),
                            ),
                            frame=int(frame_image.frame),
                            track_id=(
                                f"pred-{sample.shot_id}-"
                                f"{frame_image.frame}-"
                                f"{detection_index}"
                            ),
                            x_min=int(round(x1)),
                            y_min=int(round(y1)),
                            width=width,
                            height=height,
                        )
                    )

            all_predictions.append(sample_predictions)

        return all_predictions


@ModelRegistry.register(
    "yolo_ufo_p2",
    ["video"],
    YoloTrainParams,
    YoloPredictParams,
)
class YoloVideoDetectionP2Model(YoloVideoDetectionModel):
    """YOLO26 P2 detector for smaller objects."""

    model_name = "yolo26-p2.yaml"
    imgsz = 1024
    batch = 2

    def define_model(self) -> str:
        """Use the P2 architecture distributed with Ultralytics."""
        return self.model_name

    def get_training_model(
        self,
        params: pydantic.BaseModel | None,
    ) -> str:
        """Always train the P2 architecture instead of a pretrained variant."""
        del params
        return self.model_name


# The top-level model package currently imports this historical name. Keeping
# the alias allows both YOLO registrations above to run without modifying files
# outside ultralytics_detection during this first integration pass.
# this is important else the project will throw errors saying model DebugVideoGetSampleModel not found
# we will delete the project in a future implementation.
DebugVideoGetSampleModel = YoloVideoDetectionModel

"""
This version:
* Includes every frame from each validated sample.
* Uses a frame-number dictionary for annotation lookup.
* Stores encoded image bytes in memory.
* Uses the TokTagger data loader for training and prediction.
* Supports negative frames with empty bounding-box lists.
* Removes all print() statements.
* Keeps both ordinary YOLO and P2 registrations.
* Preserves the existing top-level import through the compatibility alias.
"""