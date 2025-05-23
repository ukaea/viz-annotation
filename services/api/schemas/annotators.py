from typing import List
from pydantic import BaseModel
from enum import Enum

class DataTypes(Enum):
    TIME_SERIES = "time_series"
    IMAGE = "image"

class Annotator(BaseModel):
    supported_datatypes: List[DataTypes] # This is just a placeholder, not sure what this would need?
    
class FindPeaks(Annotator):
    whatever_you_can_change: int