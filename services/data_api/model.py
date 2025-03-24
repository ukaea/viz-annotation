from datetime import datetime
from enum import Enum
from typing import Dict, List
from pydantic import BaseModel


class Region(BaseModel):
    time_min: float
    time_max: float


class Event(BaseModel):
    time: float


class ELMRegion(Region):
    type: int


class ELM(Event):
    height: float
    valid: bool


class Shot(BaseModel):
    created: str = datetime.now().isoformat()
    shot_id: int
    validated: bool = False
    elms: List[ELM]
    elm_type: str = ""
    regions: List[ELMRegion]
