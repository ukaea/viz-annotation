from typing import Tuple, List
from pydantic import BaseModel
from enum import Enum

from services.api.schemas.events import Event


class Task(Enum):
    ELM = "ELM"
    UFO = "UFO"


class Project(BaseModel):
    name: str
    samples: Tuple[int, int]
    task: Task = None
    events: List[Event] = None
