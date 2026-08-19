from fastapi import APIRouter, Request, Depends, Query
from toktagger.api.core.data_loaders import LoaderRegistry, DataLoaderError
from toktagger.api.crud import utils
from toktagger.api.crud.db import MongoDBClient
from toktagger.api.schemas.data import DataParams
from toktagger.api.schemas.models import LoadMethods
from toktagger.api.schemas.projects import Project
from toktagger.api.schemas.samples import ShotData, TimeSeriesFileData
from toktagger.api.models import models_dependencies_installed, check_models_enabled
import logging
import typing
import toktagger.api.config as config

logger = logging.getLogger(__name__)

if models_dependencies_installed():
    from toktagger.api.models.base import ModelRegistry

router = APIRouter(prefix="/meta", tags=["Metadata"])


async def get_project_signals(db_client: MongoDBClient, project: Project) -> list[str]:
    """Get the signal names which models in this project can use.

    A sample records its signal names only when the data loader needs them to
    fetch the data, e.g. for UDA shots. File-based loaders treat them as an
    optional column filter, so when the sample does not give them, read the
    names from the data itself.
    """
    samples = await utils.get_samples(db_client, project.id, count=1)
    if not samples:
        return []

    sample = samples[0]
    if (
        isinstance(sample.data, (ShotData, TimeSeriesFileData))
        and sample.data.signal_names
    ):
        return list(sample.data.signal_names)

    data_loader = LoaderRegistry.get(project.data_loader)()
    try:
        data = data_loader.get_sample(sample, params=DataParams())
    except (FileNotFoundError, DataLoaderError, ValueError) as e:
        logger.warning(f"Could not read the signal names for project {project.id}: {e}")
        return []

    # Multi-signal responses are keyed by signal name; single-signal ones are not.
    values = getattr(data, "values", None)
    return list(values) if isinstance(values, dict) else []


@router.get("/dataloader")
async def get_dataloaders(request: Request) -> list[str]:
    """Get list of available dataloaders."""
    return LoaderRegistry.names()


@router.get("/dataloader/{loader}")
async def get_data_schema(loader: str) -> dict[str, typing.Any]:
    """Get schema which is required for getting data with this dataloader"""
    return LoaderRegistry.get_data_schema(loader)


@router.get(
    "/models",
    dependencies=[Depends(check_models_enabled)],
)
async def get_model_types(task: str) -> list[str]:
    """Get list of available models for a given task."""
    return ModelRegistry.names(task)


@router.get(
    "/models/load",
    dependencies=[Depends(check_models_enabled)],
)
async def get_model_load_methods() -> list[str]:
    """Get list of enabled ways to load pretrained weights into the server."""
    enabled = []
    if config.settings.models.local_load_enabled:
        enabled.append(LoadMethods.LOCAL)
    if config.settings.models.gitlab_load_enabled:
        enabled.append(LoadMethods.GITLAB)
    if config.settings.models.huggingface_load_enabled:
        enabled.append(LoadMethods.HUGGINGFACE)

    return enabled


@router.get(
    "/models/load/{load_method}",
    dependencies=[Depends(check_models_enabled)],
)
async def get_model_load_method_allowlist(load_method: LoadMethods) -> str | None:
    """Get allowed ID for loading from online projects, if applicable."""
    match load_method:
        case LoadMethods.LOCAL:
            return None
        case LoadMethods.GITLAB:
            return (
                str(config.settings.models.gitlab_project_id)
                if config.settings.models.gitlab_project_id
                else None
            )
        case LoadMethods.HUGGINGFACE:
            return config.settings.models.huggingface_userspace


@router.get(
    "/models/{model}",
    dependencies=[Depends(check_models_enabled)],
)
async def get_model_meta(model: str) -> dict[str, typing.Any]:
    """Get metadata (name, description, tasks) for a specific model type."""
    description = ModelRegistry.get_description(model)
    tasks = [str(t) for t in ModelRegistry.tasks(model)]
    return {"name": model, "description": description, "tasks": tasks}


@router.get(
    "/models/{model}/train",
    dependencies=[Depends(check_models_enabled)],
)
async def get_model_training_schema(
    request: Request,
    model: str,
    project_id: str | None = Query(
        None,
        description=(
            "If provided, populate any class_label field's dropdown options "
            "from this project's configured time-region annotation labels, and "
            "any signal_names field's options from this project's sample signals."
        ),
    ),
) -> dict[str, typing.Any] | None:
    """Get params required for training this model."""
    schema = ModelRegistry.get_params_schema(
        model, schema_type="training", return_draft_07=True
    )
    if not schema or not project_id:
        return schema

    db_client = request.app.state.db_client
    properties = schema.get("properties", {})
    project = await utils.get_project(db_client, project_id)

    if "class_label" in properties:
        labels = list(project.time_region_labels or [])
        # Fields with a default (e.g. the optional filter on template-matching
        # models) allow an unselected/blank value, so it must stay a valid
        # enum choice. Required fields (e.g. minirocket/shapelet) must not
        # offer a blank option, since the user must always pick a real label.
        if "class_label" not in schema.get("required", []):
            labels = [""] + labels
        properties["class_label"]["enum"] = labels

    if "signal_names" in properties:
        signals = await get_project_signals(db_client, project)
        # An empty enum would leave the user unable to fill in a required field,
        # so projects with no readable signals keep the free-text input.
        if signals:
            properties["signal_names"]["items"]["enum"] = signals

    return schema


@router.get(
    "/models/{model}/predict",
    dependencies=[Depends(check_models_enabled)],
)
async def get_model_prediction_schema(model: str) -> dict[str, typing.Any] | None:
    """Get params required for predicting with this model."""
    return ModelRegistry.get_params_schema(
        model, schema_type="prediction", return_draft_07=True
    )
