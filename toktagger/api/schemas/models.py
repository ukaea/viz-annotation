from typing import Literal, Annotated, Optional
from pydantic import Field, field_validator
from toktagger.api.schemas import ConfiguredModel
from enum import Enum


class ModelIn(ConfiguredModel):
    type: str
    version: int
    training_status: Literal["queued", "started", "failed", "completed", "aborted"]
    progress: Annotated[float, Field(strict=True, ge=0, le=100)]
    score: float
    task_id: Optional[str] | None = None

    @field_validator("type")
    def check_model_type(cls, value):
        from toktagger.api.models.base import ModelRegistry

        if value not in (names := ModelRegistry.names()):
            raise ValueError(
                f"Invalid model type '{value}' - valid options are '{names}'."
            )

        return value


class ModelUpdate(ConfiguredModel):
    training_status: Optional[
        Literal["queued", "started", "failed", "completed", "aborted"]
    ] = None
    progress: Optional[Annotated[float, Field(strict=True, ge=0, le=100)]] = None
    score: Optional[float] = None
    task_id: Optional[str] = None


class Model(ModelIn):
    id: str = Field(..., alias="_id")
    project_id: str


class LoadTypes(str, Enum):
    LOCAL = "local"
