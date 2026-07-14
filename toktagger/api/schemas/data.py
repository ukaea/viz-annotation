from typing import Union, Literal
from pydantic import BaseModel
from toktagger.api.schemas import ConfiguredModel
from PIL.Image import Image as PILImage


class Data(BaseModel):
    pass


class TimeSeriesData(Data):
    time: list[float]
    values: list[float]


class MultiVariateTimeSeriesData(Data):
    values: dict[str, TimeSeriesData | None]


class CompositeData(Data):
    values: dict[str, "DataResponseType"]


class SpectrogramData(Data):
    time: list[float]
    frequency: list[float]
    amplitude: list[list[float]]


class ImageData(Data):
    frame: int
    values: str  # Base64 encoded string


class DataParams(ConfiguredModel):
    name: Literal["identity"] = "identity"


class RawImageData():
    def __init__(self, frame: int, values: PILImage):
        """
        Return a raw image without any encoding for ML use.
        RawImageData is not the part of the API schema.
        """
        self.frame = frame
        self.values = values


class ImageParams(DataParams):
    name: Literal["image"] = "image"
    frame: int | None
    return_raw: bool = False # optional boolean flag to return raw image


DataResponseType = Union[
    Data,
    ImageData,
    MultiVariateTimeSeriesData,
    CompositeData,
    SpectrogramData,
]

DataParamTypes = Union[DataParams, ImageParams]
