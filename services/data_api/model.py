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
    shot_id: int
    elms: List[ELM]
    regions: List[ELMRegion]


class ShotInDB(Shot):
    id: str
