from typing import List
from pydantic import BaseModel


class Region(BaseModel):
    time_min: float
    time_max: float


class Event(BaseModel):
    time: float


class ELM(Event):
    height: float
    valid: bool


class Shot(BaseModel):
    shot_id: int
    elms: List[ELM]

class ShotInDB(Shot):
    id: str
