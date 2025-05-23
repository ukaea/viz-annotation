from typing import List
from pydantic import BaseModel
from enum import Enum

from services.api.schemas.samples import Sample


class Task(Enum):
    ELM = "ELM"
    UFO = "UFO"


class QueryStrategyType(str, Enum):
    RANDOM = "random"
    SEQUENTIAL = "sequential"


class DataLoaderType(str, Enum):
    PARQUET = "parquet"
    UDA = "uda"


class Project(BaseModel):
    name: str
    samples: List[Sample]
    task: Task
    query_strategy: QueryStrategyType
    data_loader: DataLoaderType
