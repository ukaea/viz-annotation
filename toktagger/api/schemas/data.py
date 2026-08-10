from typing import Literal

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


DataResponseType = (
    Data | ImageData | MultiVariateTimeSeriesData | Profile2DData | MultiProfile2DData
)

DataParamTypes = DataParams | ImageParams
