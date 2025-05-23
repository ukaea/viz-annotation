from typing import Annotated, List, Optional
from enum import Enum
from pydantic import BaseModel, Field

from services.api.schemas.annotations import Annotation


class FileType(Enum, str):
    CSV = "csv"
    PARQUET = "parquet"
    MP4 = "mp4"


class FileProtocol(Enum, str):
    S3 = "s3"
    LOCAL = "file"


class ShotProtocol(Enum, str):
    UDA = "uda"
    SAL = "sal"


class FileData(BaseModel):
    file_name: str
    type: FileType
    protocol: FileProtocol = FileProtocol.LOCAL
    column_names: Optional[list[str]] = None


class ShotData(BaseModel):
    shot_id: int
    protocol: ShotProtocol
    signal_names: Annotated[list[str], Field(min_items=1)]


class Sample(BaseModel):
    data: FileData | ShotData
    annotations: Optional[List[Annotation]] = None
