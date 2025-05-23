from pydantic import BaseModel


class Data(BaseModel):
    pass


class TimeSeriesData(Data):
    time: list[float]
    value: list[float]


class MultiVariateTimeSeriesData(Data):
    time: list[float]
    values: dict[str, list[float]]


class SpectrogramData(Data):
    time: list[float]
    frequency: list[float]
    value: list[float]


class ImageData(Data):
    data: list[list[float]]
