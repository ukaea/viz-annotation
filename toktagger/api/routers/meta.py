from fastapi import APIRouter, Request, Depends
from toktagger.api.core.data_loaders import LoaderRegistry
from toktagger.api.schemas.models import LoadMethods
from toktagger.api.models import models_dependencies_installed, check_models_enabled
import typing
import toktagger.api.config as config

if models_dependencies_installed():
    from toktagger.api.models.base import ModelRegistry

router = APIRouter(prefix="/meta", tags=["Metadata"])


@router.get("/dataloader", operation_id="get_dataloaders")
async def get_dataloaders(request: Request) -> list[str]:
    """Get list of available dataloaders."""
    return LoaderRegistry.names()


@router.get("/dataloader/{loader}", operation_id="get_data_schema")
async def get_data_schema(loader: str) -> dict[str, typing.Any]:
    """Get schema which is required for getting data with this dataloader"""
    return LoaderRegistry.get_data_schema(loader)


@router.get(
    "/models",
    operation_id="get_model_types",
    dependencies=[Depends(check_models_enabled)],
)
async def get_model_types(task: str) -> list[str]:
    """Get list of available models for a given task."""
    return ModelRegistry.names(task)


@router.get(
    "/models/load",
    operation_id="get_model_load_methods",
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
    operation_id="get_model_load_method_allowlist",
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
    "/models/{model}/train",
    operation_id="get_model_training_schema",
    dependencies=[Depends(check_models_enabled)],
)
async def get_model_training_schema(model: str) -> dict[str, typing.Any] | None:
    """Get params required for training this model."""
    return ModelRegistry.get_params_schema(
        model, schema_type="training", return_draft_07=True
    )


@router.get(
    "/models/{model}/predict",
    operation_id="get_model_prediction_schema",
    dependencies=[Depends(check_models_enabled)],
)
async def get_model_prediction_schema(model: str) -> dict[str, typing.Any] | None:
    """Get params required for predicting with this model."""
    return ModelRegistry.get_params_schema(
        model, schema_type="prediction", return_draft_07=True
    )
