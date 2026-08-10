from typing import Annotated, Literal

from pydantic import BaseModel, Field, computed_field

from toktagger.api.schemas import ConfiguredModel
from toktagger.api.schemas.annotations import AnnotationBatchTypes


class FileData(BaseModel):
    file_name: str
    protocol: Literal["file", "s3"] = "file"


class ImageFileData(FileData):
    type: Literal["png", "jpeg"]


class ImageArrayFileData(FileData):
    type: Literal["npy", "npz"]
    signal_name: str | None = None


class TimeSeriesFileData(FileData):
    type: Literal["csv", "tsv", "parquet", "feather", "json", "xlsx"]
    signal_names: list[str] | None = None


class ShotData(BaseModel):
    protocol: Literal["uda", "uda_camera", "sal", "fair_mast"]
    signal_names: Annotated[list[str], Field(min_length=1)]


DataTypes = TimeSeriesFileData | ImageFileData | ImageArrayFileData | ShotData


class SampleBase(ConfiguredModel):
    shot_id: int
    data: DataTypes


class SampleIn(SampleBase):
    annotations: list[AnnotationBatchTypes] | None = None

    @computed_field
    @property
    def validated_annotations(self) -> bool:
        if not self.annotations:
            return False

        return any(
            annotation.validated for annotation in self.annotations
        )  # TODO any or all?


class Sample(SampleBase):
    validated_annotations: bool
    id: str = Field(..., alias="_id")
    project_id: str


class SampleUpdate(ConfiguredModel):
    validated_annotations: bool | None = None


class SampleUpdateBatchItem(ConfiguredModel):
    id: str = Field(..., alias="_id")
    updates: SampleUpdate


class SampleSummary(BaseModel):
    total: int
    shot_min: int | None = None
    shot_max: int | None = None
    data: DataTypes | None = None
