from __future__ import annotations  # store type hints as strings

import logging
from collections.abc import Iterator
from pathlib import Path

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
from toktagger.api.schemas.data import DataParamTypes, ImageData, ImageParams
from toktagger.api.schemas.samples import Sample
from toktagger.api.models.ultralytics_detection.base import (
    BaseUltralyticsDetection,
    DetectionRecord,
    YoloP2TrainParams,
    YoloTrainParams,
)
from toktagger.api.models.ultralytics_detection.utils import (
    check_pretrained_model_availability,
    resolve_weights_path,
)

logger = logging.getLogger(__name__)


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
    this_frame_only: bool = pydantic.Field(
        default=False,
        description="Predict only the current frame for individual-sample predictions; ignored for multi-sample predictions.",
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
        raise ValueError("Samples and annotations must have the same length.")

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
            if getattr(annotation, "type", None) != "video_bounding_box":
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

            boxes: list[tuple[float, float, float, float]] = []
            classes: list[int] = []

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

                boxes.append((x1, y1, x2, y2))
                classes.append(class_map[annotation.label])

            frame_manifest.append(
                DetectionRecord(
                    shot_id=int(sample.shot_id),
                    frame=frame,
                    image=bytes(frame_image.values),
                    boxes=boxes,
                    classes=classes,
                )
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
        raise TypeError("Expected raw image bytes but received a base64 string.")

    encoded_image = np.frombuffer(
        bytes(frame_image.values),
        dtype=np.uint8,
    )
    image = cv2.imdecode(
        encoded_image,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(f"Could not decode frame {frame_image.frame}.")

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
    model_family = "yolo"

    imgsz = 640
    batch = 5

    def build_manifest(
        self,
        samples: list[Sample],
        annotations: list[list[Annotation]],
    ) -> list[DetectionRecord]:
        """Build the in-memory video training manifest."""
        labels = sorted(
            {
                annotation.label
                for sample_annotations in annotations
                for annotation in sample_annotations
                if getattr(annotation, "type", None) == "video_bounding_box"
            }
        )

        if not labels:
            raise ValueError(
                "No video bounding-box labels were found in the training annotations."
            )

        # Use a stable ordering so every annotation label maps to the same class ID
        # throughout manifest construction and checkpoint creation.
        self.class_map = {label: class_id for class_id, label in enumerate(labels)}

        return build_video_frame_manifest(
            samples=samples,
            annotations=annotations,
            class_map=self.class_map,
            data_loader=self.data_loader,
        )

    def load(
        self,
        results_dir: Path,
        weights_filename: str | None = None,
    ) -> None:
        """Load YOLO weights into this Ray actor."""
        weights_path = resolve_weights_path(
            results_dir=results_dir,
            weights_filename=weights_filename,
        )

        self._prediction_model = YOLO(str(weights_path))
        self._trained_weights_path = weights_path

    def predict(
        self,
        samples: list[Sample],
        params: YoloPredictParams,
        data_params: DataParamTypes | None = None,
    ) -> list[list[AnnotationBase]]:
        """Predict bounding boxes for the requested video frames."""

        if (
            params.this_frame_only
            and data_params is not None
            and not isinstance(data_params, ImageParams)
        ):
            raise TypeError("this_frame_only requires image data parameters.")

        # if load() was called self._prediction_model should exist
        # else we borrow self._trained_weights_path from self.train()
        if not hasattr(self, "_prediction_model"):
            self._prediction_model = YOLO(str(self._trained_weights_path))

        # in either case, model is not reloaded for every single prediction request
        model = self._prediction_model

        all_predictions: list[list[AnnotationBase]] = []

        # Keep the OpenCV-decoded image in BGR order. Ultralytics expects NumPy
        # prediction sources in BGR and converts them to RGB internally.
        # https://github.com/ultralytics/ultralytics/blob/9ea768a302d8865b1a16c9ef81a441d0e1714ad1/ultralytics/engine/predictor.py#L173
        for sample in samples:
            sample_predictions: list[AnnotationBase] = []

            if params.this_frame_only and data_params is not None:
                frame_image = self.data_loader.get_sample(
                    sample,
                    ImageParams(
                        name="image",
                        frame=data_params.frame,
                        return_raw=True,
                    ),
                )
                if not isinstance(frame_image, ImageData):
                    raise TypeError("Expected the data loader to return ImageData.")
                frame_images = iter((frame_image,))
            else:
                frame_images = iter_sample_frames(
                    self.data_loader,
                    sample,
                )

            for frame_image in frame_images:
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

                if not result.boxes:
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

                    # Class names are stored in the Ultralytics checkpoint and
                    # can be restored when the trained model is loaded.
                    # >>> model = YOLO("path/to/best.pt")
                    # >>> model.names
                    # {0: 'UFO'}
                    # this gets transferred to results = model.predict()
                    # when doing prediction
                    label = result.names[int(class_id)]

                    sample_predictions.append(
                        VideoBoundingBox(
                            label=label,
                            created_by=self.type,
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
    YoloP2TrainParams,
    YoloPredictParams,
)
class YoloVideoDetectionP2Model(YoloVideoDetectionModel):
    """YOLO26 P2 detector for smaller objects.
    Builds the P2 architecture and transfer the compatible pretrained weights.
    New P2-specific layers are randomly initialised.
    YAML file: ultralytics/cfg/models/26/yolo26-p2.yaml
    """

    model_name = "yolo26-p2.yaml"

    # Higher resolution preserves small objects but requires a smaller batch.
    imgsz = 1024
    batch = 2

    def define_model(self) -> str:
        """Use the P2 architecture distributed with Ultralytics."""
        return self.model_name

    def get_training_model(
        self,
        params: YoloTrainParams,
    ) -> str:
        """Build the P2 architecture matching the selected checkpoint scale."""
        checkpoint_stem = Path(params.yolo_size).stem
        return f"{checkpoint_stem}-p2.yaml"

    def get_pretrained_weights(
        self,
        params: YoloTrainParams,
    ) -> str:
        """Resolve weights used to initialise compatible P2 layers."""
        return str(check_pretrained_model_availability(params.yolo_size))
