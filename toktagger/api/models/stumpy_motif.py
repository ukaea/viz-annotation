import pickle
import logging

import numpy as np
import pydantic

from toktagger.api.models.base import Model, ModelRegistry
from toktagger.api.models.event_detection_utils import (
    compute_window_size,
    extract_segment,
    load_aligned_signals,
    merge_detections,
    non_max_suppression,
)
from toktagger.api.schemas.annotations import Annotation, AnnotationBase
from toktagger.api.schemas.data import DataParams
from toktagger.api.schemas.samples import Sample

logger = logging.getLogger("ray")


class StumpyMotifTrainParams(pydantic.BaseModel):
    signal_names: list[str] = pydantic.Field(
        min_length=1,
        description=(
            "Signal channels to use. Provide one for single-channel mode, "
            "or multiple for multivariate STUMPY (e.g. ['Ip', 'dalpha'])."
        ),
    )
    threshold: float = pydantic.Field(
        default=3.0,
        gt=0,
        description=(
            "Maximum z-normalised Euclidean distance (MASS) for a detection. "
            "Typical values are 1–5; lower values require closer matches."
        ),
    )


class StumpyMotifPredictParams(pydantic.BaseModel):
    threshold: float | None = pydantic.Field(
        default=None,
        gt=0,
        description=(
            "Maximum z-normalised Euclidean distance (MASS) for a detection. "
            "Typical values are 1–5; lower values require closer matches. "
            "Defaults to the threshold saved during training."
        ),
    )


@ModelRegistry.register(
    "stumpy_motif", ["time-series"], StumpyMotifTrainParams, StumpyMotifPredictParams
)
class StumpyMotifModel(Model):
    """Template-matching model using STUMPY's FFT-based MASS distance profile.

    Training extracts z-normalised subsequences from labelled TimeRegion
    annotations as templates. Inference computes the nearest-neighbour distance
    profile (via MASS) from each template against the target signal and reports
    positions whose distance falls below the trained threshold. Supports both
    single-channel and multivariate inputs (per-channel MASS averaged).
    """

    def define_model(self) -> dict:
        return {
            "templates": [],
            "window_size": None,
            "signal_names": [],
            "threshold": 3.0,
        }

    def train(
        self,
        samples: list[Sample],
        annotations: list[list[Annotation]],
        params: StumpyMotifTrainParams,
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

        window_size = compute_window_size(ann_time_pairs)
        logger.info(f"StumpyMotif: inferred window_size={window_size}")

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
                        # (window_size, n_channels) — per-channel MASS uses template[:, ch]
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
        params: StumpyMotifPredictParams | None = None,
        data_params: DataParams | None = None,
    ) -> list[list[AnnotationBase]]:
        signal_names: list[str] = self.model["signal_names"]
        threshold_override = params.threshold if params else None
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
                self._detect(signal_vals, time_array, threshold_override)
            )
            results.append(detections)

        return results

    def _detect(
        self,
        signal_vals: np.ndarray,
        time_array: np.ndarray,
        threshold_override: float | None = None,
    ) -> list[AnnotationBase]:
        import stumpy

        templates: list[tuple[np.ndarray, str]] = self.model["templates"]
        window_size: int = self.model["window_size"]
        threshold: float = (
            threshold_override
            if threshold_override is not None
            else self.model["threshold"]
        )
        multivariate = signal_vals.ndim == 2

        n = signal_vals.shape[-1] if multivariate else len(signal_vals)
        if n < window_size:
            return []

        combined_dist = np.full(n, np.inf)
        combined_label = [""] * n

        for template, label in templates:
            if multivariate:
                # Average per-channel MASS distance profiles
                n_channels = signal_vals.shape[0]
                chan_profiles = []
                for ch in range(n_channels):
                    dp = stumpy.mass(
                        template[:, ch].astype(np.float64),
                        signal_vals[ch].astype(np.float64),
                    )
                    chan_profiles.append(dp)
                dp = np.mean(np.array(chan_profiles), axis=0)
            else:
                dp = stumpy.mass(
                    template.astype(np.float64), signal_vals.astype(np.float64)
                )

            length = min(len(dp), n)
            improved = dp[:length] < combined_dist[:length]
            combined_dist[:length] = np.where(
                improved, dp[:length], combined_dist[:length]
            )
            for idx in np.where(improved)[0]:
                combined_label[idx] = label

        hit_positions = [
            int(i) for i in np.where(combined_dist < threshold)[0] if combined_label[i]
        ]

        if not hit_positions:
            return []

        label_positions: dict[str, list[int]] = {}
        for pos in hit_positions:
            lbl = combined_label[pos]
            label_positions.setdefault(lbl, []).append(pos)

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
