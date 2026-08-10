from enum import Enum
from typing import Literal

from pydantic import BaseModel


class AnnotatorTypes(str, Enum):
    PEAK_DETECTION = "peak_detection"
    OUTLIER_DETECTION = "outlier_detection"
    CHANGE_POINT_DETECTION = "change_point_detection"
    JUMP_DETECTION = "jump_detection"
    MANUAL_ANNOTATION = "manual"
    SPECTROGRAM_THRESHOLD = "spectrogram_threshold"


class DataTypes(Enum):
    TIME_SERIES = "time_series"
    IMAGE = "image"


class AnnotatorParams(BaseModel):
    pass


class PeakDetectionParams(AnnotatorParams):
    signal_name: str
    prominence: float
    distance: int
    time_min: float | None = None
    time_max: float | None = None


class OutlierDetectionParams(AnnotatorParams):
    signal_name: str
    method: Literal["mad", "isoforest"]
    threshold: float | None = None
    contamination: float | None = None


class ChangePointDetectionParams(AnnotatorParams):
    signal_name: str
    method: Literal["pelt", "hmm"]
    num_points: int
    penalty: float | None = None
    num_components: int | None = None  # Only used if method is 'hmm'


class JumpDetectionParams(AnnotatorParams):
    signal_name: str
    threshold: float
    min_distance: int
    smoothing: float
    num_points: int


class SpectrogramThresholdParams(AnnotatorParams):
    signal_name: str
    percentile: float


AnnotatorParamTypes = (
    PeakDetectionParams
    | OutlierDetectionParams
    | ChangePointDetectionParams
    | JumpDetectionParams
    | SpectrogramThresholdParams
)
