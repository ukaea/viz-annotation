from __future__ import annotations  # store type hints as strings

import logging
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pydantic
import torch
from torch.utils.data import DataLoader, Dataset
from ultralytics.data.augment import LetterBox
from ultralytics.models.yolo.detect import DetectionTrainer

# Callable to pass functions in another function
from collections.abc import Callable
from ultralytics.utils import RANK

from toktagger.api.models.base import Model
from toktagger.api.schemas.annotations import Annotation, AnnotationBase
from toktagger.api.schemas.samples import Sample

from .utils import (
    check_pretrained_model_availability,
    get_toktagger_cache_dir,
    get_torch_device,
)

logger = logging.getLogger(__name__)

DetectionRecord = dict[str, Any]


class UltralyticsDetectionDataset(Dataset):
    """In-memory image dataset for Ultralytics detection training.

    Each record contains encoded image bytes and the bounding boxes associated
    with that frame. Images remain PNG/JPEG encoded in the manifest and are
    decoded only when requested by the PyTorch data loader.
    This can be improved later by loading the images lazily.
    """

    def __init__(
        self,
        records: list[DetectionRecord],
        imgsz: int = 640,
    ) -> None:
        self.records = records
        self.imgsz = imgsz
        self.letterbox = LetterBox(
            new_shape=(imgsz, imgsz),
            auto=False,
            scaleup=True,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]

        encoded_image = np.frombuffer(record["image"], dtype=np.uint8)
        image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(
                f"Could not decode shot {record['shot_id']} frame {record['frame']}"
            )

        # Ultralytics trains its pretrained models using RGB channel ordering.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]

        letterboxed_image = self.letterbox(image=image)
        if not isinstance(letterboxed_image, np.ndarray):
            letterboxed_image = letterboxed_image["img"]

        gain = min(
            self.imgsz / original_height,
            self.imgsz / original_width,
        )
        resized_width = round(original_width * gain)
        resized_height = round(original_height * gain)
        padding_width = (self.imgsz - resized_width) / 2
        padding_height = (self.imgsz - resized_height) / 2

        normalized_boxes = []

        for x1, y1, x2, y2 in record["boxes"]:
            x1 = x1 * gain + padding_width
            x2 = x2 * gain + padding_width
            y1 = y1 * gain + padding_height
            y2 = y2 * gain + padding_height

            center_x = ((x1 + x2) / 2) / self.imgsz
            center_y = ((y1 + y2) / 2) / self.imgsz
            width = (x2 - x1) / self.imgsz
            height = (y2 - y1) / self.imgsz

            normalized_boxes.append([center_x, center_y, width, height])

        if normalized_boxes:
            bboxes = torch.tensor(
                normalized_boxes,
                dtype=torch.float32,
            )
        else:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)

        if record["classes"]:
            classes = torch.tensor(
                record["classes"],
                dtype=torch.float32,
            ).view(-1, 1)
        else:
            classes = torch.zeros((0, 1), dtype=torch.float32)

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(letterboxed_image.transpose(2, 0, 1))
        )

        return {
            "img": image_tensor,
            "cls": classes,
            "bboxes": bboxes,
            "im_file": (f"shot-{record['shot_id']}/frame-{record['frame']}"),
            "ori_shape": (original_height, original_width),
            "resized_shape": (self.imgsz, self.imgsz),
            "ratio_pad": (
                (gain, gain),
                (padding_width, padding_height),
            ),
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Combine image records into the batch format Ultralytics expects."""
        images = []
        classes = []
        bboxes = []
        batch_indices = []
        image_files = []
        original_shapes = []
        resized_shapes = []
        ratio_pads = []

        for batch_index, sample in enumerate(batch):
            images.append(sample["img"])
            classes.append(sample["cls"])
            bboxes.append(sample["bboxes"])

            batch_indices.append(
                torch.full(
                    (len(sample["bboxes"]),),
                    batch_index,
                    dtype=torch.int64,
                )
            )

            image_files.append(sample["im_file"])
            original_shapes.append(sample["ori_shape"])
            resized_shapes.append(sample["resized_shape"])
            ratio_pads.append(sample["ratio_pad"])

        return {
            "img": torch.stack(images, dim=0),
            "cls": torch.cat(classes, dim=0),
            "bboxes": torch.cat(bboxes, dim=0),
            "batch_idx": torch.cat(batch_indices, dim=0),
            "im_file": image_files,
            "ori_shape": original_shapes,
            "resized_shape": resized_shapes,
            "ratio_pad": ratio_pads,
        }


class ToktaggerDetectionTrainer(DetectionTrainer):
    """Ultralytics trainer that consumes in-memory TokTagger datasets."""

    def __init__(
        self,
        *args,
        train_dataset: UltralyticsDetectionDataset,
        val_dataset: UltralyticsDetectionDataset | None = None,
        class_names: dict[int, str] | None = None,
        progress_callback: Callable[..., None] | None = None,
        **kwargs,
    ) -> None:
        # These must be assigned before DetectionTrainer.__init__ calls
        # the overridden get_dataset() method.
        self._tok_train_dataset = train_dataset
        self._tok_val_dataset = val_dataset
        self._tok_class_names = class_names or {0: "object"}
        self._tok_progress_callback = progress_callback

        super().__init__(*args, **kwargs)

        # https://docs.ultralytics.com/usage/callbacks
        # Executes after the end of the epoch.
        self.add_callback(
            "on_fit_epoch_end",
            self._log_progress,
        )

    def _log_progress(self, trainer) -> None:
        """
        Send epoch-level training progress back to TokTagger.
        For now it sends a basic progress log:
        epoch 1/50  → 2%
        epoch 2/50  → 4%
        ...
        epoch 50/50 → 100%
        It is worth including the metrics in future.
        """

        # https://github.com/ultralytics/ultralytics/blob/c3576e753264563eddeb1a3df0ce9565c3eb6b4c/ultralytics/engine/trainer.py#L142
        # RANK is the process rank used for PyTorch DistributedDataParallel (DDP)
        # Run this block on single process mode RANK = -1
        # or on main/leader DDP process RANK 0
        # to avoid duplication of the print message.
        if RANK not in {-1, 0} or self._tok_progress_callback is None:
            return

        self._tok_progress_callback(
            progress=int((trainer.epoch + 1) / trainer.epochs * 100),
            score=None,
        )

    def get_dataset(self) -> dict[str, Any]:
        """
        Provide metadata without requiring an Ultralytics YAML file.
        This avoids maintaining the dataset in a specific format which requires split and annotations stored in txt files.
        """
        return {
            "nc": len(self._tok_class_names),
            "names": self._tok_class_names,
            "channels": 3,
            "train": "",
            "val": "",
        }

    def get_dataloader(
        self,
        dataset_path=None,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ) -> DataLoader:
        """Return a data loader for the injected TokTagger dataset."""
        del dataset_path, rank

        if mode == "train":
            dataset = self._tok_train_dataset
        else:
            dataset = self._tok_val_dataset

        # Version one does not create a validation dataset.
        # Ultralytics simply doesn't need it for training.
        # if we want to avail things like patience and validation loss
        # we might consider splitting the validated dataset into training and val.

        if dataset is None:
            dataset = self._tok_train_dataset

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=self.args.workers,
            collate_fn=dataset.collate_fn,
        )

    def final_eval(self) -> None:
        """Skip final validation until a validation dataset is implemented."""
        return None


class BaseUltralyticsDetection(Model):
    """Base class for TokTagger models backed by Ultralytics detection."""

    model_name: str = "yolo26n.pt"
    class_map: dict[str, int] = {"object": 0}

    imgsz: int = 640
    batch: int = 5
    workers: int = 0

    @property
    def class_names(self) -> dict[int, str]:
        """Return the class-ID-to-label mapping expected by Ultralytics."""
        return {class_id: label for label, class_id in self.class_map.items()}

    def get_device(self) -> torch.device:
        """Return the device assigned to this model actor."""
        if self.gpu_available() and torch.cuda.is_available():
            return torch.device("cuda")

        return get_torch_device()

    def define_model(self) -> str:
        """Resolve the default pretrained checkpoint."""
        model_path = check_pretrained_model_availability(self.model_name)
        return str(model_path)

    def build_manifest(
        self,
        samples: list[Sample],
        annotations: list[list[Annotation]],
    ) -> list[DetectionRecord]:
        """Build task-specific image records."""
        raise NotImplementedError

    def get_training_model(
        self,
        params: pydantic.BaseModel | None,
    ) -> str:
        """Resolve the model selected through the training form."""
        selected_model = getattr(
            params,
            "yolo_size",
            self.model_name,
        )
        return str(check_pretrained_model_availability(selected_model))

    def make_train_dataset(
        self,
        records: list[DetectionRecord],
    ) -> UltralyticsDetectionDataset:
        """Create the in-memory Ultralytics training dataset."""
        return UltralyticsDetectionDataset(
            records,
            imgsz=self.imgsz,
        )

    def make_overrides(
        self,
        model_path: str,
        epochs: int,
        learning_rate: float,
        has_validation_data: bool,
    ) -> dict[str, Any]:
        """Build the configuration passed to Ultralytics."""
        training_output_root = (
            get_toktagger_cache_dir() / "yolo_model" / "training_outputs"
        )
        training_output_root.mkdir(parents=True, exist_ok=True)

        return {
            "model": model_path,
            "epochs": epochs,
            "lr0": learning_rate,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "workers": self.workers,
            "device": self.get_device().type,
            "project": str(training_output_root),
            "name": self.id,
            "exist_ok": True,
            "save": True,
            "plots": False,
            "val": has_validation_data,
            "close_mosaic": 0,
        }

    def train(
        self,
        samples: list[Sample],
        annotations: list[list[Annotation]],
        params: pydantic.BaseModel | None = None,
    ) -> float:
        """Train an Ultralytics detector using TokTagger data."""
        self.log_progress(
            training_status="started",
            progress=0,
        )

        epochs = int(getattr(params, "epochs", 50))
        learning_rate = float(getattr(params, "learning_rate", 1e-3))
        model_path = self.get_training_model(params)

        train_records = self.build_manifest(
            samples,
            annotations,
        )

        if not train_records:
            raise ValueError("No training records were created.")

        logger.info(
            "Created %s training records.",
            len(train_records),
        )

        train_dataset = self.make_train_dataset(train_records)
        validation_dataset = None

        overrides = self.make_overrides(
            model_path=model_path,
            epochs=epochs,
            learning_rate=learning_rate,
            has_validation_data=validation_dataset is not None,
        )

        logger.info(
            "Training %s on %s.",
            model_path,
            overrides["device"],
        )

        trainer = ToktaggerDetectionTrainer(
            overrides=overrides,
            train_dataset=train_dataset,
            val_dataset=validation_dataset,
            class_names=self.class_names,
            progress_callback=self.log_progress,
        )
        trainer.train()

        best_weights = Path(trainer.best)
        last_weights = Path(trainer.last)

        if best_weights.is_file():
            self._trained_weights_path = best_weights
        elif last_weights.is_file():
            self._trained_weights_path = last_weights
        else:
            raise FileNotFoundError(
                "Ultralytics training completed without producing weights."
            )

        self.log_progress(
            training_status="completed",
            progress=100,
        )

        # Validation is intentionally postponed, so no meaningful score is
        # available yet.
        return 0.0

    def get_prediction_weights_path(self) -> str:
        """Return trained weights when available."""
        trained_weights = getattr(
            self,
            "_trained_weights_path",
            None,
        )

        if trained_weights is not None:
            trained_weights = Path(trained_weights)
            if trained_weights.is_file():
                return str(trained_weights)

        return str(self.model)

    def save(self, file_stem: str) -> None:
        """Copy trained weights into TokTagger's model storage."""
        source = Path(self.get_prediction_weights_path())

        if not source.is_file():
            raise FileNotFoundError(f"Could not find trained weights at {source}")

        destination = Path(file_stem).with_suffix(".pt")
        destination.parent.mkdir(parents=True, exist_ok=True)

        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

        self.model = str(destination)
        self._trained_weights_path = destination

    def load(self, file_path: str) -> None:
        """Load a previously saved Ultralytics checkpoint."""
        weights_path = Path(file_path)

        if not weights_path.is_file():
            raise FileNotFoundError(f"Could not find model weights at {weights_path}")

        self.model = str(weights_path)
        self._trained_weights_path = weights_path

    def predict(
        self,
        samples: list[Sample],
        params: pydantic.BaseModel | None = None,
        data_params=None,
    ) -> list[list[AnnotationBase]]:
        """Require concrete models to convert detections into annotations."""
        raise NotImplementedError
