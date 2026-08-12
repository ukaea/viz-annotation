from fastapi import APIRouter, Request, Depends, Query
from toktagger.api.core.data_loaders import LoaderRegistry
from toktagger.api.crud import utils
from toktagger.api.schemas.models import LoadTypes
from toktagger.api.models import models_dependencies_installed, check_models_enabled
import typing
import toktagger.api.config as config

if models_dependencies_installed():
    from toktagger.api.models.base import ModelRegistry

router = APIRouter(prefix="/meta", tags=["Metadata"])


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
    return [LoadTypes.LOCAL] if config.settings.models.local_load_enabled else []


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
            "from this project's configured time-region annotation labels."
        ),
    ),
) -> dict[str, typing.Any] | None:
    """Get params required for training this model."""
    schema = ModelRegistry.get_params_schema(
        model, schema_type="training", return_draft_07=True
    )
    if schema and project_id and "class_label" in schema.get("properties", {}):
        db_client = request.app.state.db_client
        project = await utils.get_project(db_client, project_id)
        labels = list(project.time_region_labels or [])
        # Fields with a default (e.g. the optional filter on template-matching
        # models) allow an unselected/blank value, so it must stay a valid
        # enum choice. Required fields (e.g. minirocket/shapelet) must not
        # offer a blank option, since the user must always pick a real label.
        if "class_label" not in schema.get("required", []):
            labels = [""] + labels
        schema["properties"]["class_label"]["enum"] = labels
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
