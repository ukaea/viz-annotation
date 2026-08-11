from enum import Enum
from typing import Literal, Optional, Union
from toktagger.api.schemas import ConfiguredModel


class ViewType(str, Enum):
    IDENTITY = "identity"
    PROFILE_2D = "profile_2d"


class ViewParams(ConfiguredModel):
    name: Literal[ViewType.IDENTITY] = ViewType.IDENTITY


class Profile2DViewParams(ViewParams):
    name: Literal[ViewType.PROFILE_2D] = ViewType.PROFILE_2D
    signal_name: str
    time_min: Optional[float] = None
    time_max: Optional[float] = None
    dim_1_min: Optional[float] = None
    dim_1_max: Optional[float] = None
    values_min: Optional[float] = None
    values_max: Optional[float] = None
    log_scale: bool = False


ViewParamTypes = Union[ViewParams, Profile2DViewParams]
