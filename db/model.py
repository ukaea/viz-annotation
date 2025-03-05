from typing import List
from pydantic import BaseModel


class Region(BaseModel):
    time_min: float
    time_max: float


class Event(BaseModel):
    time: float


class FlatTop(Region):
    pass


class RampUp(Region):
    pass


class ELM(Event):
    pass


class Shot(BaseModel):
    shot_id: int
    flat_top: FlatTop
    ramp_up: RampUp
    elms: List[ELM]


class ShotInDB(Shot):
    id: str
