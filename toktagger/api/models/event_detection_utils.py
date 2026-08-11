import numpy as np
from scipy.interpolate import interp1d

from toktagger.api.schemas.annotations import TimeRegion


def compute_window_size(ann_time_pairs: list[tuple]) -> int:
    """Return median annotation duration converted to sample count.

    Parameters
    ----------
    ann_time_pairs : list of (Annotation, np.ndarray) pairs
        Each pair is an annotation and the time array of its parent signal.
    """
    durations = []
    for ann, time_array in ann_time_pairs:
        if not (hasattr(ann, "time_min") and hasattr(ann, "time_max")):
            continue
        if len(time_array) < 2:
            continue
        dt = float(np.median(np.diff(time_array)))
        if dt <= 0:
            continue
        n_samples = int(round((ann.time_max - ann.time_min) / dt))
        if n_samples > 1:
            durations.append(n_samples)

    if not durations:
        raise ValueError(
            "No valid TimeRegion annotations found to infer window size. "
            "Ensure the project has TimeRegion annotations before training."
        )
    return max(2, int(np.median(durations)))


def extract_segment(
    time_array: np.ndarray,
    values: np.ndarray,
    time_min: float,
    time_max: float,
    target_length: int,
) -> np.ndarray | None:
    """Extract a signal segment by time range, resample and z-score normalise.

    Returns None if the segment is too short to resample.
    """
    mask = (time_array >= time_min) & (time_array <= time_max)
    seg_t = time_array[mask]
    seg_v = values[mask]
    if len(seg_v) < 2:
        return None
    interp = interp1d(seg_t, seg_v, bounds_error=False, fill_value="extrapolate")
    new_t = np.linspace(seg_t[0], seg_t[-1], target_length)
    return zscore(interp(new_t))


def zscore(arr: np.ndarray) -> np.ndarray:
    std = np.std(arr)
    return (arr - np.mean(arr)) / (std + 1e-8)


def load_aligned_signals(
    signal_data: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Align one or more signals onto a common time grid.

    Parameters
    ----------
    signal_data : list of (time_array, values) pairs, one per channel.

    Returns
    -------
    (time_array, values) where values is 1D if a single channel was given,
    otherwise a 2D array of shape (n_channels, n_samples). When multiple
    channels are given, every channel is linearly resampled onto the time
    grid of the densest (most-sampled) input signal, so channels recorded
    at different sampling rates can still be combined.
    """
    if len(signal_data) == 1:
        ta, va = signal_data[0]
        return np.asarray(ta, dtype=float), np.asarray(va, dtype=float)

    ref_idx = max(range(len(signal_data)), key=lambda i: len(signal_data[i][0]))
    ref_time = np.asarray(signal_data[ref_idx][0], dtype=float)

    aligned = []
    for ta, va in signal_data:
        ta = np.asarray(ta, dtype=float)
        va = np.asarray(va, dtype=float)
        if ta.shape == ref_time.shape and np.array_equal(ta, ref_time):
            aligned.append(va)
        else:
            aligned.append(np.interp(ref_time, ta, va))
    return ref_time, np.array(aligned)


def select_training_label(
    sample_data: list[tuple[np.ndarray, np.ndarray, list]],
    event_label: str | None,
) -> str:
    """Determine which annotation label to train a binary classifier on.

    Models using this helper train a single binary classifier (event vs.
    background), so exactly one annotation label must be selected. If
    `event_label` is given it is returned as-is (callers are expected to
    filter annotations to that label). Otherwise, the label is inferred
    only if all annotations share the same label; if annotations use more
    than one distinct label, a ValueError is raised rather than silently
    collapsing them into a single positive class.
    """
    all_labels = {
        ann.label
        for _, _, anns in sample_data
        for ann in anns
        if hasattr(ann, "time_min")
    }
    if event_label is not None:
        return event_label
    if len(all_labels) > 1:
        raise ValueError(
            f"Multiple annotation labels found {sorted(all_labels)}, but this "
            "model trains a single binary classifier. Set `event_label` to "
            "choose which label to train on."
        )
    if not all_labels:
        return "Event"
    return next(iter(all_labels))


def non_max_suppression(
    detections: list[TimeRegion], iou_threshold: float = 0.5
) -> list[TimeRegion]:
    """Remove overlapping TimeRegion detections with the same label using greedy NMS.

    merge_detections handles consecutive window positions; this handles the case
    where separate position clusters produce time regions that still overlap heavily.
    Detections are sorted by duration (shorter = more precise) before suppression.
    """
    if not detections:
        return []

    by_label: dict[str, list[TimeRegion]] = {}
    for det in detections:
        by_label.setdefault(det.label, []).append(det)

    result: list[TimeRegion] = []
    for group in by_label.values():
        group = sorted(group, key=lambda a: a.time_max - a.time_min)
        kept: list[TimeRegion] = []
        for ann in group:
            for other in kept:
                inter = max(
                    0.0,
                    min(ann.time_max, other.time_max)
                    - max(ann.time_min, other.time_min),
                )
                union = (
                    (ann.time_max - ann.time_min)
                    + (other.time_max - other.time_min)
                    - inter
                )
                if union > 0 and inter / union > iou_threshold:
                    break
            else:
                kept.append(ann)
        result.extend(kept)
    return result


def merge_detections(
    positions: list[int],
    window_size: int,
    time_array: np.ndarray,
    label: str,
    model_type: str,
) -> list[TimeRegion]:
    """Merge adjacent/overlapping window start indices into TimeRegion annotations."""
    if not positions:
        return []

    positions = sorted(positions)
    regions: list[tuple[int, int]] = []
    start = end = positions[0]

    for pos in positions[1:]:
        if pos <= end + window_size:
            end = max(end, pos)
        else:
            regions.append((start, end))
            start = end = pos
    regions.append((start, end))

    anns = []
    n = len(time_array)
    for s, e in regions:
        t_min = float(time_array[s])
        t_max = float(time_array[min(e + window_size - 1, n - 1)])
        if t_max <= t_min:
            t_max = t_min + 1e-6
        anns.append(
            TimeRegion(
                label=label,
                time_min=t_min,
                time_max=t_max,
                created_by=model_type,
                validated=False,
            )
        )
    return anns
