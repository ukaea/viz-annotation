from pydantic import Field, field_validator, computed_field
from typing import Optional, List
from enum import Enum


from toktagger.api.core.data_loaders import LoaderRegistry
from toktagger.api.schemas import ConfiguredModel


class Task(str, Enum):
    """The type of labelling task for a project."""

    TIME_SERIES = "time-series"
    PROFILE_2D = "profile-2d"
    VIDEO = "video"


class QueryStrategyType(str, Enum):
    """The strategy to use when selecting the next sample to annotate."""

    RANDOM = "random"
    SEQUENTIAL = "sequential"
    UNCERTAINTY = "uncertainty"


class ProjectIn(ConfiguredModel):
    name: str = Field(..., description="The name of the project.")
    task: Task = Field(..., description="The type of labelling task.")
    query_strategy: QueryStrategyType = Field(
        ...,
        description="The strategy to use when selecting the next sample to annotate.",
    )

    data_loader: str = Field(
        ...,
        description="The type of data which will need to be loaded for this project.",
    )

    time_min: Optional[float] = Field(
        None,
        description="The minimum time (in seconds) for samples in this project.",
    )

    time_max: Optional[float] = Field(
        None,
        description="The maximum time (in seconds) for samples in this project.",
    )

    min_time_step: Optional[float] = Field(
        None,
        description="The minimum time step (in seconds) between samples in this project.",
    )

    shot_labels: Optional[List[str]] = Field(
        default=["Valid", "Invalid"],
        description="The list of labels to use for shot labelling (if applicable).",
    )

    time_region_labels: Optional[List[str]] = Field(
        default=[
            "ELM",
            "L-mode",
            "H-mode",
            "Thermal Quench",
            "Current Quench",
            "Sawtooth",
            "IRE",
            "Locked Mode",
            "VDE",
            "Flat Top",
            "Ramp Up",
            "Ramp Down",
            "NTM",
            "LLM",
            "Sawteeth",
            "Unknown",
        ],
        description="The list of labels to use for time region labelling (if applicable).",
    )

    time_point_labels: Optional[List[str]] = Field(
        default=[
            "Disruption",
            "Thermal Quench",
            "Current Quench",
            "Control Loss",
            "Locked Mode",
            "VDE",
        ],
        description="The list of labels to use for time point labelling (if applicable).",
    )

    bounding_box_labels: Optional[List[str]] = Field(
        default=["NTM", "LLM", "Sawteeth", "Locked Mode"],
        description="The list of labels to use for bounding box labelling (if applicable).",
    )

    polygon_labels: Optional[List[str]] = Field(
        default=["NTM", "LLM", "Sawteeth", "Locked Mode"],
        description="The list of labels to use for polygon labelling (if applicable).",
    )

    video_bounding_box_labels: Optional[List[str]] = Field(
        default=[
            "UFO",
            "Minor UFO",
            "Major UFO",
            "MARFE",
            "Other Anomaly",
        ],
        description="The list of labels to use for video bounding box labelling (if applicable).",
    )

    @computed_field
    @property
    def model_types(self) -> List[str]:
        from toktagger.api.models import models_dependencies_installed

        if not models_dependencies_installed():
            return []
        from toktagger.api.models.base import ModelRegistry

        return ModelRegistry.names(Task(self.task))

    @field_validator("data_loader")
    def check_data_loader(cls, value):
        if value not in (names := LoaderRegistry.names()):
            raise ValueError(
                f"Invalid data loader '{value}' - valid options are '{names}'."
            )

        return value


class Project(ProjectIn):
    id: str = Field(..., alias="_id", description="The ID of this project.")
