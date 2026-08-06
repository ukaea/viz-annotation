import numpy as np
import ruptures as rpt
import hmmlearn.hmm as hmm
from typing import Iterable, List, Tuple
from abc import ABC, abstractmethod
from scipy.signal import find_peaks, peak_widths, stft
from scipy.ndimage import uniform_filter1d, gaussian_filter, uniform_filter
from scipy.interpolate import interp1d
from skimage import measure, morphology
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from toktagger.api.schemas.data import (
    MultiVariateTimeSeriesData,
    Profile2DData,
    MultiProfile2DData,
)
from toktagger.api.schemas.annotators import (
    AnnotatorTypes,
    ChangePointDetectionParams,
    PeakDetectionParams,
    JumpDetectionParams,
    OutlierDetectionParams,
)
from shapely.geometry import Polygon

from toktagger.api.schemas.data import TimeSeriesData
from toktagger.api.schemas.annotations import TimeRegion
from toktagger.api.schemas.annotations import (
    # Aliased to avoid clashing with shapely's Polygon, used below for geometry.
    Polygon as PolygonAnnotation,
)
from toktagger.api.schemas.annotators import (
    Profile2DThresholdParams,
    AnnotatorParamTypes,
)
from toktagger.api.schemas.projects import Task


def binary_runs_to_tuples(arr: np.ndarray) -> list[tuple[int, int]]:
    """
    Convert a 1D binary array into a list of (start, end) index tuples for each contiguous run of 1s.
    Parameters
    ----------
    arr : np.ndarray
        A 1D numpy array of binary values (0s and 1s).

    Returns
    -------
    list of tuple of int
        A list of tuples, where each tuple (start, end) represents the start (inclusive)

    Raises
    ------
    ValueError
        If the input array is not 1-dimensional.

    """
    arr = np.asarray(arr, dtype=bool)

    if arr.ndim != 1:
        raise ValueError("Input must be a 1D array or list.")

    padded = np.pad(arr.astype(int), (1, 1), mode="constant")
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if ends[-1] == len(arr):
        ends[-1] = ends[-1] - 1

    return list(zip(starts, ends))


def extract_segments(arr: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Convert a 1D array into a list of (start_index, end_index, label) tuples representing contiguous segments of the same labels.
    This function identifies contiguous segments of the same value in a 1D array and returns their start and end indices along with the label.

    Parameters
    ----------
    arr : np.ndarray or list
        Input 1D array or list of labels.

    Returns
    -------
    list of tuple of int
        List of (start, end, label) for each segment, where `start` and `end` are
        the indices of the segment (inclusive), and `label` is the value in that segment.

    Raises
    ------
    ValueError
        If the input is not a 1D array or list.

    Examples
    --------
    >>> extract_segments([1, 1, 2, 2, 2, 3])
    [(0, 1, 1), (2, 4, 2), (5, 5, 3)]
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("Input must be a 1D array or list.")

    change_indices = np.where(np.diff(arr) != 0)[0] + 1
    segment_starts = np.concatenate(([0], change_indices))
    segment_ends = np.concatenate((change_indices, [len(arr) - 1]))

    return [
        (start, end, arr[start]) for start, end in zip(segment_starts, segment_ends)
    ]


def downsample_time_series(
    time: np.ndarray, signal: np.ndarray, num_points=500
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample a time series to a specified number of points using linear interpolation.

    Parameters
    ----------
    time : np.ndarray
        1D array of time values.
    signal : np.ndarray
        1D array of signal values corresponding to the time array.
    num_points : int, optional
        Number of points to downsample to. Default is 500.

    Returns
    -------
    tuple of np.ndarray
        Tuple containing the downsampled time and signal arrays.

    Raises
    ------
    ValueError
        If the input arrays are not 1-dimensional.
    """
    signal = np.asarray(signal)
    time = np.asarray(time)

    if signal.ndim != 1 or time.ndim != 1:
        raise ValueError("Input must be a 1D array or list.")

    if len(time) <= num_points:
        return time, signal

    time_coarse = np.linspace(time.min(), time.max(), num_points)
    interpolator = interp1d(time, signal, kind="linear")
    signal = interpolator(time_coarse)
    return time_coarse, signal


def _coords_to_flat_list(coords: Iterable[Tuple[float, float]]) -> List[float]:
    """
    Convert an iterable of (x,y) coordinates to COCO flattened list [x1,y1,x2,y2,...].
    Drops the closing coordinate if it's equal to the first (Shapely exterior rings often close).
    """
    coords = list(coords)
    if len(coords) == 0:
        return []
    # If last is same as first, drop it
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    flat = []
    for x, y in coords:
        # keep floats (COCO accepts floats)
        flat.append(float(x))
        flat.append(float(y))
    return flat


def smooth_binary_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Smooth a binary mask (0/1) using morphological open+close.
    """
    # Ensure boolean mask
    mask = mask.astype(bool)

    selem = morphology.disk(radius)

    opened = morphology.opening(mask, selem)  # remove small noise
    closed = morphology.closing(opened, selem)  # fill small holes

    return closed.astype(np.uint8)


def compute_stft(
    data: TimeSeriesData, nfft: int = 256, nperseg: int = 256, noverlap: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.array(data.time)
    values = np.array(data.values)

    if len(time) < 2:
        raise ValueError(
            "At least two time samples are required to compute a sample rate."
        )

    time_step = np.abs(np.median(np.diff(time)))
    if time_step == 0:
        raise ValueError(
            "Cannot compute a sample rate: the median time step between samples is zero."
        )

    sample_rate = 1 / time_step

    freq, ts, Zxx = stft(
        values,
        fs=int(sample_rate),
        nfft=nfft,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    freq /= 1000
    ts += time[0]

    return freq, ts, np.abs(Zxx)


class DataAnnotator(ABC):
    @abstractmethod
    def __init__(self, params: AnnotatorParamTypes):
        pass

    @abstractmethod
    def predict(self, data: MultiVariateTimeSeriesData) -> list[TimeRegion]:
        pass


class PeakDetectionAnnotator(DataAnnotator):
    """
    PeakDetectionAnnotator for detecting peaks in a multivariate time series signal.

    This class applies normalization, detrending, and peak finding algorithms to identify
    regions in the specified signal where peaks occur, based on configurable parameters.

    Parameters
    ----------
    params : PeakDetectionParams
        Configuration parameters for peak detection, including signal name, prominence,
        distance, and optional time bounds.

    Methods
    -------
    predict(data: MultiVariateTimeSeriesData) -> list[TimeRegion]
        Detects peaks in the specified signal and returns a list of TimeRegion objects
        representing the regions around each detected peak.

    Attributes
    ----------
    params : PeakDetectionParams
        Configuration parameters for peak detection.

    Examples
    --------
    >>> annotator = PeakDetectionAnnotator(params)
    >>> regions = annotator.predict(data)
    """

    def __init__(self, params: PeakDetectionParams):
        self.params = params

    def predict(self, data: MultiVariateTimeSeriesData) -> list[TimeRegion]:
        time = data.values[self.params.signal_name].time
        time = np.array(time)
        signal = data.values[self.params.signal_name].values
        signal = np.array(signal)

        signal = signal.reshape(-1, 1)
        scaler = StandardScaler()
        signal = scaler.fit_transform(signal)
        signal = signal.flatten()

        tmin, tmax = self.params.time_min, self.params.time_max
        tmin = time.min() if tmin is None else tmin
        tmax = time.max() if tmax is None else tmax

        trend = uniform_filter1d(signal, 1000)
        dalpha_detrend = signal - trend

        peak_idx, params = find_peaks(
            dalpha_detrend,
            prominence=self.params.prominence,
            width=[1, 150],
            distance=self.params.distance,
        )

        widths, _, _, _ = peak_widths(dalpha_detrend, peak_idx, rel_height=0.9)

        dt = np.abs(time[1] - time[0])
        regions = []
        for w, idx in zip(widths, peak_idx):
            width = w * dt
            peak_time = time[idx]
            if peak_time >= tmin and peak_time <= tmax:
                region = TimeRegion(
                    label="Unknown",
                    time_min=max(float(peak_time - width), np.min(time)),
                    time_max=min(float(peak_time + width), np.max(time)),
                    created_by=AnnotatorTypes.PEAK_DETECTION,
                )
                regions.append(region)

        return regions


class OutlierDetectionAnnotator(DataAnnotator):
    """
    Annotator for detecting outliers in multivariate time series data using specified methods.

    Parameters
    ----------
    params : OutlierDetectionParams
        Configuration parameters for outlier detection, including method, signal name, and threshold.

    Methods
    -------
    predict(data: MultiVariateTimeSeriesData) -> list[TimeRegion]
        Detects outliers in the specified signal of the input time series data and returns regions marked as outliers.

    isoforest_outliers(time: np.ndarray, values: np.ndarray) -> list[TimeRegion]
        Identifies outlier regions using the Isolation Forest algorithm.

    mad_outliers(time: np.ndarray, data: np.ndarray) -> list[TimeRegion]
        Identifies outlier regions using the Median Absolute Deviation (MAD) method.

    Raises
    ------
    ValueError
        If an unknown outlier detection method is specified in `params`.
    """

    def __init__(self, params: OutlierDetectionParams):
        self.params = params

    def predict(self, data: MultiVariateTimeSeriesData) -> list[TimeRegion]:
        time = data.values[self.params.signal_name].time
        time = np.array(time)
        values = data.values[self.params.signal_name].values
        values = np.array(values)

        if self.params.method == "mad":
            bounds = self.mad_outliers(time, values)
        elif self.params.method == "isoforest":
            bounds = self.isoforest_outliers(time, values)
        else:
            raise ValueError(f"Unknown outlier detection method: {self.params.method}")

        return bounds

    def isoforest_outliers(
        self, time: np.ndarray, values: np.ndarray
    ) -> list[TimeRegion]:
        values = values.reshape(-1, 1)

        if self.params.contamination is None or self.params.contamination <= 0:
            return []

        scaler = StandardScaler()
        np_scaled = scaler.fit_transform(values)

        model = IsolationForest(contamination=self.params.contamination)
        model.fit(np_scaled)

        outliers = model.predict(np_scaled) == -1
        bounds = binary_runs_to_tuples(outliers)
        bounds = [
            TimeRegion(
                time_min=time[imin],
                time_max=time[imax],
                label="Unknown",
                created_by=AnnotatorTypes.OUTLIER_DETECTION,
            )
            for imin, imax in bounds
        ]
        return bounds

    def mad_outliers(self, time: np.ndarray, data: np.ndarray) -> list[TimeRegion]:
        median = np.median(data)
        abs_deviation = np.abs(data - median)
        mad = np.median(abs_deviation)
        mad = np.clip(mad, a_min=1e-10, a_max=None)

        MAD_CONSTANT = 0.6745  # Constant to convert MAD to standard Z-score
        modified_z_scores = MAD_CONSTANT * (data - median) / mad
        outliers = np.abs(modified_z_scores) > self.params.threshold

        if not np.any(outliers):
            return []

        bounds = binary_runs_to_tuples(outliers)
        bounds = [
            TimeRegion(
                time_min=time[imin],
                time_max=time[imax],
                label="Unknown",
                created_by=AnnotatorTypes.OUTLIER_DETECTION,
            )
            for imin, imax in bounds
        ]
        return bounds


class ChangePointDetectionAnnotator(DataAnnotator):
    """
    Annotator for detecting change points in multivariate time series data.

    This class provides methods to detect change points using either the PELT algorithm
    or Hidden Markov Models (HMM). It supports downsampling for performance and
    standardizes input signals before applying the selected change point detection method.

    Parameters
    ----------
    params : ChangePointDetectionParams
        Configuration parameters for change point detection, including the signal name,
        detection method ('pelt' or 'hmm'), number of downsample points, penalty for PELT,
        and number of HMM components.

    Methods
    -------
    predict(data: MultiVariateTimeSeriesData) -> list[TimeRegion]
        Detects change points in the provided time series data and returns a list of
        annotated time regions.

    pelt_changepoint(signal, time)
        Applies the PELT algorithm to detect change points in the signal.

    hmm_changepoint(signal, time)
        Applies a Hidden Markov Model to detect change points in the signal.
    """

    def __init__(self, params: ChangePointDetectionParams):
        self.params = params

    def predict(self, data: MultiVariateTimeSeriesData) -> list[TimeRegion]:
        time = data.values[self.params.signal_name].time
        time = np.array(time)

        signal = data.values[self.params.signal_name].values
        signal = np.array(signal)
        signal = np.nan_to_num(signal, 0)

        # Downsample the time series to for performance
        time, signal = downsample_time_series(
            time, signal, num_points=self.params.num_points
        )

        if self.params.method == "pelt":
            bounds = self.pelt_changepoint(signal, time)
        elif self.params.method == "hmm":
            bounds = self.hmm_changepoint(signal, time)
        else:
            raise ValueError(
                f"Unknown change point detection method: {self.params.method}"
            )

        return bounds

    def pelt_changepoint(self, signal, time):
        time = time.reshape(-1, 1)
        signal = signal.reshape(-1, 1)

        scaler = StandardScaler()
        signal = scaler.fit_transform(signal)

        intercept = np.ones(len(time))

        inputs = np.column_stack((signal, time, intercept))

        algo = rpt.Pelt(model="linear")
        algo.fit(inputs)
        result = algo.predict(pen=self.params.penalty)

        result = np.concatenate([[0], result])
        result[-1] -= 1
        time = time.flatten()
        bounds = [
            TimeRegion(
                time_min=time[imin],
                time_max=time[imax],
                label="Unknown",
                created_by=AnnotatorTypes.CHANGE_POINT_DETECTION,
            )
            for imin, imax in zip(result, result[1:])
        ]
        return bounds

    def hmm_changepoint(self, signal, time):
        time = time.reshape(-1, 1)
        signal = signal.reshape(-1, 1)

        scaler = StandardScaler()
        signal = scaler.fit_transform(signal)

        best_score = -1e-10
        best_model = None
        num_models = 10

        # try a range of fits to find the best model for the data
        for seed in range(num_models):
            m = hmm.GaussianHMM(
                n_components=self.params.num_components, n_iter=200, random_state=seed
            )
            m.fit(signal)
            score = m.score(signal)
            if score > best_score:
                best_score = score
                best_model = m

        # use best model for hidden states
        hidden_states = best_model.predict(signal)
        time = time.flatten()
        bounds = extract_segments(hidden_states + 1)
        bounds = [(time[imin], time[imax]) for (imin, imax, _) in bounds]
        bounds = [
            TimeRegion(
                time_min=tmin,
                time_max=tmax,
                label="Unknown",
                created_by=AnnotatorTypes.CHANGE_POINT_DETECTION,
            )
            for (tmin, tmax) in bounds
        ]
        return bounds


class JumpDetectionAnnotator(DataAnnotator):
    """
    JumpDetectionAnnotator for detecting sharp jumps in a multivariate time series signal.

    This annotator processes a specified signal from the input time series data,
    applies smoothing and normalization, and identifies regions where the signal
    exhibits sharp changes (jumps) based on the gradient exceeding a threshold.
    Detected jumps are returned as time regions.

    Parameters
    ----------
    params : JumpDetectionParams
        Parameters controlling the detection process, including signal name, smoothing factor,
        number of downsample points, detection threshold, and minimum distance between detections.

    Methods
    -------
    predict(data: MultiVariateTimeSeriesData) -> list[TimeRegion]
        Detects and returns a list of time regions where jumps are detected in the specified signal.

    Attributes
    ----------
    params : JumpDetectionParams
        Configuration parameters for jump detection.

    Examples
    --------
    >>> annotator = JumpDetectionAnnotator(params)
    >>> regions = annotator.predict(data)
    """

    def __init__(self, params: JumpDetectionParams):
        self.params = params

    def predict(self, data: MultiVariateTimeSeriesData) -> list[TimeRegion]:
        time = data.values[self.params.signal_name].time
        time = np.array(time)

        signal = data.values[self.params.signal_name].values
        signal = np.array(signal)
        signal = np.nan_to_num(signal, 0)

        # Downsample the time series to for performance
        time, signal = downsample_time_series(
            time, signal, num_points=self.params.num_points
        )

        # Smooth the signal to reduce noise
        signal = gaussian_filter(signal, self.params.smoothing)
        signal_grad = np.absolute(np.gradient(signal))

        signal_grad = signal_grad.reshape(-1, 1)
        scaler = StandardScaler()
        signal_grad = scaler.fit_transform(signal_grad)
        signal_grad = signal_grad.flatten()

        # Detect sharp drops (e.g., drops > 3 * std of normal fluctuations)
        threshold = self.params.threshold * signal_grad.std()
        peak_idx = np.where(signal_grad > threshold)[0]

        # Filter detections which are too close
        peak_idx = peak_idx[np.diff(peak_idx, prepend=0) > self.params.min_distance]

        bounds = []
        for i in peak_idx:
            wsize = 10
            window = signal[i - wsize : i + wsize]
            twindow = time[i - wsize : i + wsize]
            tmin = twindow[0]
            tmax = twindow[-1]
            tmin = twindow[np.argmax(window)]
            tmax = twindow[np.argmin(window)]
            bounds.append(
                TimeRegion(
                    time_min=tmin,
                    time_max=tmax,
                    label="Unknown",
                    created_by=AnnotatorTypes.JUMP_DETECTION,
                )
            )

        return bounds


class Profile2DThresholdAnnotator:
    """
    Profile2DThresholdAnnotator for generating a 2D profile mask based on thresholding.
    This annotator applies a percentile-based threshold to the 2D profile values, and generates a list of polygon annotations
    representing the regions where the profile exceeds the threshold.

    Parameters
    ----------
    params : Profile2DThresholdParams
        Configuration parameters for 2D profile thresholding, including signal name and percentile.

    Methods
    -------
    predict(data: MultiVariateTimeSeriesData) -> list[PolygonAnnotation]
        Computes the 2D profile mask based on the specified thresholding parameters.

    Examples
    --------
    >>> annotator = Profile2DThresholdAnnotator(params)
    >>> mask = annotator.predict(data)
    """

    def __init__(self, params: Profile2DThresholdParams):
        self.params = params

    def predict(
        self, data: MultiProfile2DData | MultiVariateTimeSeriesData
    ) -> list[PolygonAnnotation]:
        signal = data.values[self.params.signal_name]

        if isinstance(signal, TimeSeriesData):
            dim_1, time, values = compute_stft(signal)
        elif isinstance(signal, Profile2DData):
            dim_1 = np.array(signal.dim_1)
            time = np.array(signal.time)
            values = np.array(signal.values).T

        values = np.nan_to_num(values, 1e-6).clip(1e-6)

        if self.params.line_filter_width > 0:
            values = values - uniform_filter(
                values, size=(self.params.line_filter_width, 1)
            )
        values = gaussian_filter(values, sigma=self.params.sigma)

        mask = np.where(values > np.percentile(values, self.params.percentile), 1, 0)

        if self.params.dim_1_min is not None:
            mask[dim_1 < self.params.dim_1_min] = 0

        if self.params.dim_1_max is not None:
            mask[dim_1 > self.params.dim_1_max] = 0

        labeled_regions = measure.label(mask, connectivity=2)

        if self.params.min_size > 0:
            labeled_regions = morphology.remove_small_objects(
                labeled_regions, min_size=self.params.min_size
            )

        # Convert each labeled region to polygons
        polygons: list[Polygon] = []
        for region_label in range(1, labeled_regions.max() + 1):
            # Create a binary mask for the current label
            region_mask = labeled_regions == region_label

            # Find contours at the 0.5 level
            contours = measure.find_contours(region_mask, 0.5)

            for contour in contours:
                if (
                    self.params.max_polygon_points
                    and len(contour) > self.params.max_polygon_points
                ):
                    indices = np.round(
                        np.linspace(0, len(contour) - 1, self.params.max_polygon_points)
                    ).astype(int)
                    contour = contour[indices]
                # Contour coordinates are (row, col), convert to (x, y)
                contour_coords = [
                    (int(np.round(c[1])), int(np.round(c[0]))) for c in contour
                ]
                contour_coords = [(time[x], dim_1[y]) for x, y in contour_coords]
                poly = Polygon(contour_coords)
                # if poly.is_valid and poly.area > 0:
                polygons.append(poly)

        annotations = [
            PolygonAnnotation(
                segmentation=[_coords_to_flat_list(polygon.exterior.coords)],
                label="Unknown",
                created_by=AnnotatorTypes.PROFILE_2D_THRESHOLD,
                signal_name=self.params.signal_name,
            )
            for polygon in polygons
        ]
        return annotations


ANNOTATORS = {
    AnnotatorTypes.PEAK_DETECTION: PeakDetectionAnnotator,
    AnnotatorTypes.OUTLIER_DETECTION: OutlierDetectionAnnotator,
    AnnotatorTypes.CHANGE_POINT_DETECTION: ChangePointDetectionAnnotator,
    AnnotatorTypes.JUMP_DETECTION: JumpDetectionAnnotator,
    AnnotatorTypes.PROFILE_2D_THRESHOLD: Profile2DThresholdAnnotator,
}

# Currently only allowing these annotators to task mapping
# Might want user to be able to specify a choice when making the project down the line?
ANNOTATORS_PER_TASK = {
    Task.TIME_SERIES: [
        AnnotatorTypes.PEAK_DETECTION,
        AnnotatorTypes.OUTLIER_DETECTION,
        AnnotatorTypes.CHANGE_POINT_DETECTION,
        AnnotatorTypes.JUMP_DETECTION,
    ],
    Task.PROFILE_2D: [AnnotatorTypes.PROFILE_2D_THRESHOLD],
    Task.VIDEO: [],
}
