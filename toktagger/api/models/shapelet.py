import pathlib
import pickle
import logging
import random

import numpy as np
import pydantic
from sklearn.metrics import balanced_accuracy_score

from toktagger.api.models.base import Model, ModelRegistry
from toktagger.api.models.event_detection_utils import (
    compute_window_size,
    load_aligned_signals,
    merge_detections,
    non_max_suppression,
    select_training_label,
    zscore,
)
from toktagger.api.schemas.annotations import Annotation, AnnotationBase
from toktagger.api.schemas.data import DataParams
from toktagger.api.schemas.samples import Sample

logger = logging.getLogger("ray")


class ShapeletTrainParams(pydantic.BaseModel):
    class_label: str = pydantic.Field(
        description=(
            "Annotation label to train the binary classifier on (event vs. "
            "background). Must match one of the project's configured "
            "time-region annotation labels."
        ),
    )
    signal_names: list[str] = pydantic.Field(
        min_length=1,
        description=(
            "Signal channels to use. Provide one for univariate, "
            "or multiple for multivariate shapelet learning (e.g. ['Ip', 'dalpha'])."
        ),
    )
    n_background_per_shot: int = pydantic.Field(
        default=10,
        gt=0,
        description="Number of background (negative) windows sampled per training shot",
    )
    max_shapelets: int = pydantic.Field(
        default=10,
        gt=0,
        description="Maximum number of shapelets to extract per class",
    )
    n_shapelet_samples: int = pydantic.Field(
        default=100,
        gt=0,
        description="Number of candidate shapelet samples to evaluate",
    )
    batch_size: int = pydantic.Field(
        default=100,
        gt=0,
        description="Batch size for shapelet fitting",
    )


class ShapeletPredictParams(pydantic.BaseModel):
    step_size: int = pydantic.Field(
        default=10,
        gt=0,
        description="Sliding window stride in samples (larger = faster but coarser)",
    )


@ModelRegistry.register(
    "shapelet_transform", ["time-series"], ShapeletTrainParams, ShapeletPredictParams
)
class ShapeletTransformModel(Model):
    """Sliding-window binary event classifier using Shapelet Transform features."""

    def define_model(self) -> dict:
        return {
            "classifier": None,
            "window_size": None,
            "signal_names": [],
            "pos_label": "Event",
        }

    def train(
        self,
        samples: list[Sample],
        annotations: list[list[Annotation]],
        params: ShapeletTrainParams,
    ) -> float:
        from sktime.classification.shapelet_based import ShapeletTransformClassifier

        self.log_progress(training_status="started", progress=0)

        paired = [(s, a) for s, a in zip(samples, annotations) if a]
        if not paired:
            raise ValueError("No annotated samples found for training.")

        ann_time_pairs: list[tuple] = []
        sample_data: list[tuple] = []

        for sample, anns in paired:
            data = self.data_loader.get_sample(sample, DataParams())
            missing = [n for n in params.signal_names if data.values.get(n) is None]
            if missing:
                logger.warning(f"Signals {missing} not found in sample {sample.id}.")
                continue

            ta, va = load_aligned_signals(
                [
                    (data.values[n].time, data.values[n].values)
                    for n in params.signal_names
                ]
            )
            if va.ndim == 1:
                va = va[np.newaxis, :]  # (n_channels, n_samples)

            for ann in anns:
                ann_time_pairs.append((ann, ta))
            sample_data.append((ta, va, anns))

        if not ann_time_pairs:
            raise ValueError(
                f"Signals {params.signal_names} not found in any annotated sample."
            )

        window_size = compute_window_size(ann_time_pairs)
        logger.info(f"ShapeletTransform: inferred window_size={window_size}")

        pos_label = select_training_label(sample_data, params.class_label)

        windows: list[np.ndarray] = []
        labels: list[int] = []

        for ta, va_nd, anns in sample_data:
            n_channels, n_samples = va_nd.shape
            signal_zs = np.array([zscore(va_nd[ch]) for ch in range(n_channels)])

            ann_ranges: list[tuple[int, int]] = []
            for ann in anns:
                if not hasattr(ann, "time_min"):
                    continue
                start_idx = int(np.searchsorted(ta, ann.time_min))
                end_idx = int(np.searchsorted(ta, ann.time_max))
                ann_ranges.append((start_idx, end_idx))

                if ann.label != pos_label:
                    continue

                mid = (start_idx + end_idx) // 2
                half = window_size // 2
                w_start = max(0, mid - half)
                w_end = w_start + window_size
                if w_end > n_samples:
                    w_start = max(0, n_samples - window_size)
                    w_end = n_samples
                if w_end - w_start == window_size:
                    windows.append(signal_zs[:, w_start:w_end])
                    labels.append(1)

            rng = random.Random(42)
            attempts = 0
            neg_added = 0
            max_attempts = params.n_background_per_shot * 20
            while neg_added < params.n_background_per_shot and attempts < max_attempts:
                attempts += 1
                pos = rng.randint(0, max(0, n_samples - window_size))
                overlaps = any(
                    pos < end and (pos + window_size) > start
                    for start, end in ann_ranges
                )
                if not overlaps and pos + window_size <= n_samples:
                    windows.append(signal_zs[:, pos : pos + window_size])
                    labels.append(0)
                    neg_added += 1

        if not windows:
            raise ValueError("Could not extract training windows.")

        X = np.array(windows, dtype=np.float32)  # (n_windows, n_channels, window_size)
        y = np.array(labels)

        self.log_progress(progress=20)

        classifier = ShapeletTransformClassifier(
            max_shapelets=params.max_shapelets,
            n_shapelet_samples=params.n_shapelet_samples,
            batch_size=params.batch_size,
        )
        classifier.fit(X, y)

        self.log_progress(progress=90)

        y_pred = classifier.predict(X)
        score = float(balanced_accuracy_score(y, y_pred)) * 100

        self.model = {
            "classifier": classifier,
            "window_size": window_size,
            "signal_names": params.signal_names,
            "pos_label": pos_label,
        }

        self.log_progress(training_status="completed", progress=100, score=score)
        return score

    def predict(
        self,
        samples: list[Sample],
        params: ShapeletPredictParams = ShapeletPredictParams(),
        data_params: DataParams | None = None,
    ) -> list[list[AnnotationBase]]:
        step_size = params.step_size if params else ShapeletPredictParams().step_size
        signal_names: list[str] = self.model["signal_names"]
        results: list[list[AnnotationBase]] = []

        for sample in samples:
            data = self.data_loader.get_sample(sample, data_params or DataParams())
            missing = [n for n in signal_names if data.values.get(n) is None]
            if missing:
                logger.warning(f"Signals {missing} not found in sample {sample.id}.")
                results.append([])
                continue

            time_array, signal_vals = load_aligned_signals(
                [(data.values[n].time, data.values[n].values) for n in signal_names]
            )
            if signal_vals.ndim == 1:
                signal_vals = signal_vals[np.newaxis, :]  # (n_channels, n_samples)

            detections = non_max_suppression(
                self._classify_windows(signal_vals, time_array, step_size)
            )
            results.append(detections)

        return results

    def _classify_windows(
        self,
        signal_vals: np.ndarray,
        time_array: np.ndarray,
        step_size: int,
    ) -> list[AnnotationBase]:
        window_size: int = self.model["window_size"]
        classifier = self.model["classifier"]
        pos_label: str = self.model["pos_label"]

        n_channels, n_samples = signal_vals.shape
        if n_samples < window_size:
            return []

        positions = list(range(0, n_samples - window_size + 1, step_size))
        if not positions:
            return []

        signal_zs = np.array([zscore(signal_vals[ch]) for ch in range(n_channels)])
        windows = np.array(
            [signal_zs[:, p : p + window_size] for p in positions], dtype=np.float32
        )  # (n_windows, n_channels, window_size)

        preds = classifier.predict(windows)
        positive_positions = [positions[i] for i, pred in enumerate(preds) if pred == 1]
        return merge_detections(
            positive_positions, window_size, time_array, pos_label, self.type
        )

    def save(self, results_dir: pathlib.Path) -> None:
        with open(results_dir.joinpath("weights.model"), "wb") as f:
            pickle.dump(self.model, f)

    def load(
        self, results_dir: pathlib.Path, weights_filename: str | None = None
    ) -> None:
        weights_path = results_dir.joinpath(weights_filename or "weights.model")
        with open(weights_path, "rb") as f:
            self.model = pickle.load(f)
