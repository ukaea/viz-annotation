from typing import Union, Literal
from pydantic import BaseModel


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
    values: str  # Base64 encoded string


class DataParams(BaseModel):
    name: Literal["identity"] = "identity"


class ImageParams(DataParams):
    name: Literal["image"] = "image"
    frame: int | None


DataResponseType = Union[
    Data,
    ImageData,
    MultiVariateTimeSeriesData,
    Profile2DData,
    MultiProfile2DData,
]

DataParamTypes = Union[DataParams, ImageParams]
