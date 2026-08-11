import pathlib
import pickle
import logging
import random

import numpy as np
import pydantic
from sklearn.linear_model import RidgeClassifierCV
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


class MiniRocketTrainParams(pydantic.BaseModel):
    signal_names: list[str] = pydantic.Field(
        min_length=1,
        description=(
            "Signal channels to classify. Provide one for single-channel mode, "
            "or multiple for multivariate (e.g. ['Ip', 'dalpha'])."
        ),
    )
    n_background_per_shot: int = pydantic.Field(
        default=10,
        gt=0,
        description="Number of background (negative) windows sampled per training shot",
    )
    num_kernels: int = pydantic.Field(
        default=10000,
        gt=0,
        description="Number of MiniRocket convolutional kernels",
    )
    event_label: str | None = pydantic.Field(
        default=None,
        description=(
            "Annotation label to train the binary classifier on. Required if "
            "annotations use more than one distinct label, since this model "
            "only supports a single event label vs. background."
        ),
    )


class MiniRocketPredictParams(pydantic.BaseModel):
    step_size: int = pydantic.Field(
        default=1,
        gt=0,
        description="Sliding window stride in samples (larger = faster but coarser)",
    )


@ModelRegistry.register(
    "minirocket", ["time-series"], MiniRocketTrainParams, MiniRocketPredictParams
)
class MiniRocketModel(Model):
    """Sliding-window event classifier using MiniRocket features + Ridge regression.

    Training extracts positive windows (centred on annotation midpoints) and
    random negative (background) windows, applies MiniRocket feature
    transformation, and fits a RidgeClassifierCV. Supports both single-channel
    and multivariate inputs. Inference slides the fitted model across the target
    signal and groups consecutive positive predictions into TimeRegion
    annotations.
    """

    def define_model(self) -> dict:
        return {
            "transformer": None,
            "classifier": None,
            "window_size": None,
            "signal_names": [],
            "pos_label": "Event",
        }

    def train(
        self,
        samples: list[Sample],
        annotations: list[list[Annotation]],
        params: MiniRocketTrainParams,
    ) -> float:
        from sktime.transformations.panel.rocket import (
            MiniRocket,
            MiniRocketMultivariate,
        )

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

            for ann in anns:
                ann_time_pairs.append((ann, ta))
            sample_data.append((ta, va, anns))

        if not ann_time_pairs:
            raise ValueError(
                f"Signals {params.signal_names} not found in any annotated sample."
            )

        window_size = compute_window_size(ann_time_pairs)
        logger.info(f"MiniRocket: inferred window_size={window_size}")

        multivariate = len(params.signal_names) > 1

        pos_label = select_training_label(sample_data, params.event_label)

        windows: list[np.ndarray] = []
        labels: list[int] = []

        for ta, va, anns in sample_data:
            if multivariate:
                n_channels, n_samples = va.shape
                signal_zs = np.array([zscore(va[ch]) for ch in range(n_channels)])
            else:
                n_samples = len(va)
                signal_zs = zscore(va)

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
                    if multivariate:
                        windows.append(signal_zs[:, w_start:w_end])
                    else:
                        windows.append(signal_zs[w_start:w_end])
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
                    if multivariate:
                        windows.append(signal_zs[:, pos : pos + window_size])
                    else:
                        windows.append(signal_zs[pos : pos + window_size])
                    labels.append(0)
                    neg_added += 1

        if not windows:
            raise ValueError("Could not extract training windows.")

        if multivariate:
            X = np.array(
                windows, dtype=np.float32
            )  # (n_windows, n_channels, window_size)
        else:
            X = np.array(windows, dtype=np.float32).reshape(
                len(windows), 1, window_size
            )

        y = np.array(labels)

        self.log_progress(progress=20)

        transformer_class = MiniRocketMultivariate if multivariate else MiniRocket
        transformer = transformer_class(num_kernels=params.num_kernels)
        transformer.fit(X)
        X_features = transformer.transform(X)

        self.log_progress(progress=60)

        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        classifier.fit(X_features, y)

        self.log_progress(progress=90)

        y_pred = classifier.predict(X_features)
        score = float(balanced_accuracy_score(y, y_pred)) * 100

        self.model = {
            "transformer": transformer,
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
        params: MiniRocketPredictParams | None = None,
        data_params: DataParams | None = None,
    ) -> list[list[AnnotationBase]]:
        step_size = params.step_size if params else 1
        signal_names: list[str] = self.model["signal_names"]
        multivariate = len(signal_names) > 1
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

            if multivariate:
                signal_zs = np.array(
                    [zscore(signal_vals[ch]) for ch in range(signal_vals.shape[0])]
                )
            else:
                signal_zs = zscore(signal_vals)

            detections = non_max_suppression(
                self._classify_windows(signal_zs, time_array, step_size)
            )
            results.append(detections)

        return results

    def _classify_windows(
        self,
        signal_zs: np.ndarray,
        time_array: np.ndarray,
        step_size: int,
    ) -> list[AnnotationBase]:
        window_size: int = self.model["window_size"]
        transformer = self.model["transformer"]
        classifier = self.model["classifier"]
        pos_label: str = self.model["pos_label"]
        multivariate = signal_zs.ndim == 2

        n = signal_zs.shape[-1] if multivariate else len(signal_zs)
        if n < window_size:
            return []

        positions = list(range(0, n - window_size + 1, step_size))
        if not positions:
            return []

        if multivariate:
            windows = np.array(
                [signal_zs[:, p : p + window_size] for p in positions], dtype=np.float32
            )  # (n_windows, n_channels, window_size)
        else:
            windows = np.array(
                [signal_zs[p : p + window_size] for p in positions], dtype=np.float32
            ).reshape(len(positions), 1, window_size)

        X_features = transformer.transform(windows)
        preds = classifier.predict(X_features)

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
        # Backward compat: old models stored signal_name (singular)
        if "signal_name" in self.model and "signal_names" not in self.model:
            self.model["signal_names"] = [self.model["signal_name"]]
