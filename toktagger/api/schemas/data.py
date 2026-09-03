from typing import Union, Literal
from pydantic import BaseModel
from toktagger.api.schemas import ConfiguredModel


class Data(BaseModel):
    pass


class TimeSeriesData(Data):
    time: list[float]
    values: list[float]


class MultiVariateTimeSeriesData(Data):
    values: dict[str, TimeSeriesData | None]


class Profile2DData(Data):
    time: list[float]
    dim_1: list[float]
    values: list[list[float]]


class MultiProfile2DData(Data):
    values: dict[str, Profile2DData | None]


class ImageData(Data):
    frame: int
    values: str | list[int]  # Base64-encoded string or raw encoded file bytes


class DataParams(ConfiguredModel):
    name: Literal["identity"] = "identity"


class ImageParams(DataParams):
    name: Literal["image"] = "image"
    frame: int | None
    # Optional: Return raw encoded image bytes instead of base64.
    return_raw: bool = False


class SampleSummary(BaseModel):
    type: str
    description: str
    num_signals: int


class SummaryAxes(BaseModel):
    count: int
    max: float
    min: float


class SummaryValues(SummaryAxes):
    mean: float


class Summary2DValues(SummaryValues):
    shape: tuple[int, int]


class SignalSummary(BaseModel):
    time: SummaryAxes
    values: SummaryValues


class Signal2DSummary(BaseModel):
    time: SummaryAxes
    dim_1: SummaryAxes
    values: Summary2DValues


class ImageSampleSummary(SampleSummary, SummaryValues):
    type: Literal["video"]
    frame_number: int
    shape: tuple[int, ...]
    height: int
    width: int
    colour_mode: str


class TimeSeriesSampleSummary(SampleSummary):
    type: Literal["time-series"]
    signals: dict[str, SignalSummary]


class Profile2DSampleSummary(SampleSummary):
    type: Literal["profile-2d"]
    signals: dict[str, Signal2DSummary]


DataResponseType = Union[
    ImageData,
    MultiVariateTimeSeriesData,
    MultiProfile2DData,
]

DataParamTypes = Union[DataParams, ImageParams]

SampleSummaryTypes = Union[
    ImageSampleSummary,
    TimeSeriesSampleSummary,
    Profile2DSampleSummary,
]
