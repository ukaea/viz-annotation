from typing import Literal, Annotated, Optional
from pydantic import Field, field_validator
from toktagger.api.schemas import ConfiguredModel
from enum import Enum
import pydantic


class ModelIn(ConfiguredModel):
    type: str
    name: Optional[str] = None
    version: int
    status: Literal["queued", "training", "loading", "failed", "completed", "aborted"]
    progress: Annotated[float, Field(strict=True, ge=0, le=100)]
    score: float
    task_id: Optional[str] | None = None

    @property
    def annotator_name(self) -> str:
        """Name recorded against the annotations this model produces. Models loaded
        from pretrained weights have no name, so fall back to their type."""
        return self.name or self.type

    @field_validator("type")
    def check_model_type(cls, value):
        from toktagger.api.models.base import ModelRegistry

        if value not in (names := ModelRegistry.names()):
            raise ValueError(
                f"Invalid model type '{value}' - valid options are '{names}'."
            )

        return value


class ModelUpdate(ConfiguredModel):
    status: Optional[
        Literal["queued", "training", "loading", "failed", "completed", "aborted"]
    ] = None
    progress: Optional[Annotated[float, Field(strict=True, ge=0, le=100)]] = None
    score: Optional[float] = None
    task_id: Optional[str] = None


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
