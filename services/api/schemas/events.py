from datetime import datetime
from typing import Literal, Tuple
from pydantic import BaseModel

class Event(BaseModel):
    sample_id: int
    created: str = datetime.now().isoformat()
    validated: bool = False
    
class Point(Event):
    time: int

class Region(Event):
    time_min: float
    time_max: float
    
class UFO(Point):
    type: Literal["minor", "major"]
    height: float = None
    width: float = None
    centre: Tuple[float, float] = None
    
class ELMRegion(Region):
    type: int

class ELM(Point):
    height: float
    valid: bool
