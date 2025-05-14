from datetime import datetime
from typing import Tuple, List
from pydantic import BaseModel
from enum import Enum

from services.api.schemas.events import Event

class Sample(BaseModel):
    # Does this take anything by default..?
    pass

class UFOSample(BaseModel):
    camera: str
    frame: int