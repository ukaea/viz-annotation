from typing import List
from pydantic import BaseModel
from enum import Enum

from services.api.schemas.samples import Sample


class Task(Enum):
    ELM = "ELM"
    UFO = "UFO"


class Project(BaseModel):
    name: str
    samples: List[Sample]
    task: Task
