from datetime import datetime
from typing import Tuple
from pydantic import BaseModel


class Annotation(BaseModel):
    sample_id: int
    created: str = datetime.now().isoformat()
    validated: bool = False
    label: str


class TimePoint(Annotation):
    time: int


class TimeRegion(Annotation):
    time_min: float
    time_max: float


class BoundingBox(Annotation):
    height: float = None
    width: float = None
    centre: Tuple[float, float] = None


class VideoBoundingBox(BoundingBox):
    frame: int
