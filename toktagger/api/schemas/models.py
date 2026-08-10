from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, field_validator

from toktagger.api.schemas import ConfiguredModel


class ModelIn(ConfiguredModel):
    type: str
    version: int
    training_status: Literal["queued", "started", "failed", "completed", "aborted"]
    progress: Annotated[float, Field(strict=True, ge=0, le=100)]
    score: float
    task_id: str | None = None

    @field_validator("type")
    def check_model_type(cls, value):
        from toktagger.api.models.base import ModelRegistry

        if value not in (names := ModelRegistry.names()):
            raise ValueError(
                f"Invalid model type '{value}' - valid options are '{names}'."
            )

        return value


class ModelUpdate(ConfiguredModel):
    training_status: (
        Literal["queued", "started", "failed", "completed", "aborted"] | None
    ) = None
    progress: Annotated[float, Field(strict=True, ge=0, le=100)] | None = None
    score: float | None = None
    task_id: str | None = None


class Model(ModelIn):
    id: str = Field(..., alias="_id")
    project_id: str


class LoadTypes(str, Enum):
    LOCAL = "local"
