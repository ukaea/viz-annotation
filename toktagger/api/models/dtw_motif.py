import pickle
import logging

import numpy as np
import pydantic

from toktagger.api.models.base import Model, ModelRegistry
from toktagger.api.models.event_detection_utils import (
    extract_segment,
    load_aligned_signals,
    merge_detections,
    non_max_suppression,
    zscore,
)
from toktagger.api.schemas.annotations import Annotation, AnnotationBase
from toktagger.api.schemas.data import DataParams
from toktagger.api.schemas.samples import Sample

logger = logging.getLogger("ray")


class DTWMotifTrainParams(pydantic.BaseModel):
    signal_names: list[str] = pydantic.Field(
        min_length=1,
        description=(
            "Signal channels to use. Provide one for single-channel mode, "
            "or multiple for multivariate DTW (e.g. ['Ip', 'dalpha'])."
        ),
    )
    threshold: float = pydantic.Field(
        default=5.0,
        gt=0,
        description=(
            "Maximum z-normalised DTW distance for a detection. "
            "For z-normalised windows of length L, typical values are 2–20. "
            "Lower values require closer shape matches."
        ),
    )
    window_size: int = pydantic.Field(
        default=100,
        gt=0,
        description="Window size in samples.",
    )


class DTWMotifPredictParams(pydantic.BaseModel):
    step_size: int = pydantic.Field(
        default=10,
        gt=0,
        description=(
            "Sliding window stride in samples. "
            "Increase for faster inference at the cost of position precision."
        ),
    )


@ModelRegistry.register(
    "dtw_motif", ["time-series"], DTWMotifTrainParams, DTWMotifPredictParams
)
class DTWMotifModel(Model):
    """Template-matching model using z-normalised Dynamic Time Warping.

    Training extracts subsequences around each labelled TimeRegion annotation
    and stores them as z-normalised templates. Inference slides a window across
    the target signal, z-normalises each window locally, and reports positions
    whose DTW distance to any template falls below the trained threshold.
    Supports both single-channel and multivariate inputs.
    """

    def define_model(self) -> dict:
        return {
            "templates": [],
            "window_size": None,
            "signal_names": [],
            "threshold": 5.0,
        }

    def train(
        self,
        samples: list[Sample],
        annotations: list[list[Annotation]],
        params: DTWMotifTrainParams,
    ) -> float:
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

        window_size = params.window_size
        logger.info(f"DTWMotif: using window_size={window_size}")

        multivariate = len(params.signal_names) > 1
        templates: list[tuple[np.ndarray, str]] = []
        for ta, va, anns in sample_data:
            for ann in anns:
                if not (hasattr(ann, "time_min") and hasattr(ann, "time_max")):
                    continue
                if multivariate:
                    segs = []
                    for ch in range(va.shape[0]):
                        seg = extract_segment(
                            ta, va[ch], ann.time_min, ann.time_max, window_size
                        )
                        if seg is None:
                            segs = None
                            break
                        segs.append(seg)
                    if segs is not None:
                        templates.append((np.column_stack(segs), ann.label))
                else:
                    seg = extract_segment(
                        ta, va, ann.time_min, ann.time_max, window_size
                    )
                    if seg is not None:
                        templates.append((seg, ann.label))

        if not templates:
            raise ValueError("Could not extract any valid templates from annotations.")

        self.model = {
            "templates": templates,
            "window_size": window_size,
            "signal_names": params.signal_names,
            "threshold": params.threshold,
        }

        score = 100.0
        self.log_progress(training_status="completed", progress=100, score=score)
        return score

    def predict(
        self,
        samples: list[Sample],
        params: DTWMotifPredictParams = DTWMotifPredictParams(),
        data_params: DataParams = DataParams(),
    ) -> list[list[AnnotationBase]]:
        step_size = params.step_size
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

            detections = non_max_suppression(
                self._detect(signal_vals, time_array, step_size)
            )
            results.append(detections)

        return results

    def _detect(
        self,
        signal_vals: np.ndarray,
        time_array: np.ndarray,
        step_size: int,
    ) -> list[AnnotationBase]:
        from dtaidistance import dtw as dtw_lib

        templates: list[tuple[np.ndarray, str]] = self.model["templates"]
        window_size: int = self.model["window_size"]
        threshold: float = self.model["threshold"]
        multivariate = signal_vals.ndim == 2

        if multivariate:
            from dtaidistance import dtw_ndim

            n = signal_vals.shape[1]
        else:
            n = len(signal_vals)

        if n < window_size:
            return []

        label_positions: dict[str, list[int]] = {}

        for i in range(0, n - window_size + 1, step_size):
            if multivariate:
                window_z = np.column_stack(
                    [
                        zscore(signal_vals[ch, i : i + window_size])
                        for ch in range(signal_vals.shape[0])
                    ]
                )
                best_dist = np.inf
                best_label = ""
                for template, label in templates:
                    dist = float(dtw_ndim.distance_fast(template, window_z))
                    if dist < best_dist:
                        best_dist = dist
                        best_label = label
            else:
                window_z = zscore(signal_vals[i : i + window_size])
                best_dist = np.inf
                best_label = ""
                for template, label in templates:
                    dist = float(dtw_lib.distance_fast(template, window_z))
                    if dist < best_dist:
                        best_dist = dist
                        best_label = label

            if best_dist < threshold and best_label:
                label_positions.setdefault(best_label, []).append(i)

        results: list[AnnotationBase] = []
        for lbl, positions in label_positions.items():
            results.extend(
                merge_detections(positions, window_size, time_array, lbl, self.type)
            )
        return results

    def save(self, file_stem: str) -> None:
        with open(f"{file_stem}.pkl", "wb") as f:
            pickle.dump(self.model, f)

    def load(self, file_path: str) -> None:
        with open(file_path, "rb") as f:
            self.model = pickle.load(f)
        # Backward compat: old models stored signal_name (singular)
        if "signal_name" in self.model and "signal_names" not in self.model:
            self.model["signal_names"] = [self.model["signal_name"]]
