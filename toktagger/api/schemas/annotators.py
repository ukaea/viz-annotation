from typing import Literal, Optional, Union
from pydantic import BaseModel
from enum import Enum
from toktagger.api.schemas.projects import Task


class AnnotatorTypes(str, Enum):
    PEAK_DETECTION = "peak_detection"
    OUTLIER_DETECTION = "outlier_detection"
    CHANGE_POINT_DETECTION = "change_point_detection"
    JUMP_DETECTION = "jump_detection"
    MANUAL_ANNOTATION = "manual"
    PROFILE_2D_THRESHOLD = "profile_2d_threshold"


class DataTypes(Enum):
    TIME_SERIES = "time_series"
    IMAGE = "image"


class AnnotatorParams(BaseModel):
    pass


class PeakDetectionParams(AnnotatorParams):
    signal_name: str
    prominence: float
    distance: int
    time_min: Optional[float] = None
    time_max: Optional[float] = None


class OutlierDetectionParams(AnnotatorParams):
    signal_name: str
    method: Literal["mad", "isoforest"]
    threshold: Optional[float] = None
    contamination: Optional[float] = None


class ChangePointDetectionParams(AnnotatorParams):
    signal_name: str
    method: Literal["pelt", "hmm"]
    num_points: int
    penalty: Optional[float] = None
    num_components: Optional[int] = None  # Only used if method is 'hmm'


class JumpDetectionParams(AnnotatorParams):
    signal_name: str
    threshold: float
    min_distance: int
    smoothing: float
    num_points: int


class Profile2DThresholdParams(AnnotatorParams):
    signal_name: str
    percentile: float
    dim_1_min: Optional[float] = None
    dim_1_max: Optional[float] = None
    sigma: float = 0.1
    min_size: int = 150
    line_filter_width: int = 3
    max_polygon_points: int = 30


AnnotatorParamTypes = Union[
    PeakDetectionParams,
    OutlierDetectionParams,
    ChangePointDetectionParams,
    JumpDetectionParams,
    Profile2DThresholdParams,
]

# Currently a fixed list of annotators with no custom option
# Provide direct Task -> Annotator mapping
ANNOTATOR_REGISTRY = {
    Task.TIME_SERIES: [
        AnnotatorTypes.PEAK_DETECTION,
        AnnotatorTypes.CHANGE_POINT_DETECTION,
        AnnotatorTypes.JUMP_DETECTION,
        AnnotatorTypes.OUTLIER_DETECTION,
    ],
    Task.PROFILE_2D: [AnnotatorTypes.PROFILE_2D_THRESHOLD],
    Task.VIDEO: [],
}
