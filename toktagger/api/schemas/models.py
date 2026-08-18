from enum import Enum
from typing import Annotated, Literal

import pydantic
from pydantic import Field, field_validator

from toktagger.api.schemas import ConfiguredModel


class ModelIn(ConfiguredModel):
    type: str
    version: int
    status: Literal["queued", "training", "loading", "failed", "completed", "aborted"]
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
    status: (
        Literal["queued", "training", "loading", "failed", "completed", "aborted"]
        | None
    ) = None
    progress: Annotated[float, Field(strict=True, ge=0, le=100)] | None = None
    score: float | None = None
    task_id: str | None = None


class Model(ModelIn):
    id: str = Field(..., alias="_id")
    project_id: str


class LoadMethods(str, Enum):
    LOCAL = "local"
    GITLAB = "gitlab"
    HUGGINGFACE = "hugging_face"


class LoadParams(pydantic.BaseModel):
    weights_path: str


class LocalLoadParams(LoadParams):
    pass


class RemoteLoadParams(LoadParams):
    model_name: str
    model_version: str | None = None


class GitlabLoadParams(RemoteLoadParams):
    gitlab_project_id: int | None = None


class HuggingfaceLoadParams(RemoteLoadParams):
    huggingface_userspace: str | None = None
