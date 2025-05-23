from typing import List
from pydantic import BaseModel
from enum import Enum

class ModelTypes(Enum):
    CNN = "cnn"
    UNET = "unet"

class Model(BaseModel):
    type: ModelTypes
    # and whatever else we need....