from fastapi import APIRouter, Request, Depends
from toktagger.api.core.data_loaders import LoaderRegistry
from toktagger.api.schemas.models import LoadMethods
from toktagger.api.models import models_dependencies_installed, check_models_enabled
import typing
import toktagger.api.config as config

if models_dependencies_installed():
    from toktagger.api.models.base import ModelRegistry

router = APIRouter(prefix="/meta", tags=["Metadata", "MCP"])


@router.get("/dataloader", operation_id="get_dataloaders")
async def get_dataloaders(request: Request) -> list[str]:
    """
    Get list of available dataloaders.
    ----------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Retrieve the list of all registered data loader names (e.g. "uda", "tabular", "image", "uda_camera", "fair_mast", "sal").

    Use When:
        - Creating a new project and need to choose a valid data loader
        - Discovering which data sources are available
        - Validating a data_loader name before creating samples

    Do Not Use When:
        - You already know the data loader and need its schema - use get_data_schema instead
        - You need to know which dataloader a specific project uses - use get_projects instead
        - You need actual diagnostic data for a sample - use get_sample_data instead

    Returns:
        A list of string data loader names

    Example User Requests:
        - "What data loaders are available?"
        - "Can I load data from UDA?"
    """
    return LoaderRegistry.names()


@router.get("/dataloader/{loader}", operation_id="get_data_schema")
async def get_data_schema(loader: str) -> dict[str, typing.Any]:
    """
    Get schema which is required for getting data with this dataloader.
    -------------------------------------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Retrieve the data parameter schema for a specific data loader, describing what parameters are needed to fetch sample data.

    Use When:
        - You need to know what parameters to pass to the data endpoint
        - You need to know which parameters to prompt the user for when getting data for a sample

    Do Not Use When:
        - You need to know which dataloader a specific project uses - use get_projects instead
        - You already know the required parameters - use get_sample_data instead
        - You need actual diagnostic data for a sample - use get_sample_data instead

    Returns:
        A dict describing the data schema (parameter names, types, defaults)

    Example User Requests:
        - "What parameters does the UDA loader need?"
        - "Show me the schema for the tabular data loader"
    """
    return LoaderRegistry.get_data_schema(loader)


@router.get(
    "/models",
    operation_id="get_model_types",
    dependencies=[Depends(check_models_enabled)],
)
async def get_model_types(task: str) -> list[str]:
    """
    Get list of available models for a given task.
    -----------------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Retrieve the list of available ML model types for a specified task ("time-series", "video", "profile-2d").

    Use When:
        - Seeing what models can be trained or used for predictions
        - Validating a model_type before starting training

    Do Not Use When:
        - You need model training parameters - use get_model_training_schema instead
        - You need model prediction parameters - use get_model_prediction_schema instead
        - ML models are not enabled - this endpoint returns an error if ML is disabled

    Returns:
        A list of model type strings (e.g. ["disruption_cnn"])

    Example User Requests:
        - "What ML models are available for time-series tasks?"
        - "Can I train a disruption detection model for this project?"
    """
    return ModelRegistry.names(task)


@router.get(
    "/models/load",
    operation_id="get_model_load_methods",
    dependencies=[Depends(check_models_enabled)],
)
async def get_model_load_methods() -> list[str]:
    """
    Get list of enabled ways to load pretrained weights into the server.
    --------------------------------------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Retrieve the list of enabled methods for loading pretrained ML model weights (local path, GitLab, Hugging Face).

    Use When:
        - You want to know which weight loading methods are configured and available
        - You are planning to load model weights and need to choose a method

    Do Not Use When:
        - You want to actually load model weights - use load_model_weights_local, load_model_weights_gitlab, or load_model_weights_huggingface instead
        - You need a list of available model types for the given project/task - use get_model_types instead
        - ML models are not enabled - this endpoint returns an error if ML is disabled

    Returns:
        A list of enabled load method strings (e.g. ["local", "huggingface"])

    Example User Requests:
        - "What methods are available for loading model weights?"
        - "Can I load models from Hugging Face?"
    """
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
    """
    Get allowed ID for loading from online projects, if applicable.
    ---------------------------------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Retrieve the allowed project/organization ID for a specific model load method (GitLab project ID or Hugging Face userspace).
        This may be set on the server for enhanced security, or return null if any user specified project ID / userspace is accepted.

    Use When:
        - You need the GitLab project ID before calling the GitLab weights loader
        - You need the Hugging Face userspace/organization before loading from HuggingFace

    Do Not Use When:
        - You want to load weights directly - use load_model_weights_gitlab or load_model_weights_huggingface instead
        - You need the list of enabled load methods - use get_model_load_methods instead
        - ML models are not enabled - this endpoint returns an error if ML is disabled

    Returns:
        A string (project ID or userspace) if selection is restricted by the server, or null if no restriction applies

    Example User Requests:
        - "What GitLab projects can I load models from?"
        - "Which Hugging Face userspace(s) can I load models from?"
        - "Can I load an ML model from my personal gitlab project?"
        - "Can I load an ML model from the Ultralytics HuggingFace repository?"
    """
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
    """
    Get params required for training this model.
    ---------------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Retrieve the required training parameter schema for a specific model type.

    Use When:
        - You are about to start model training and need to know what parameters are required
        - You want to validate training parameters before calling the training endpoint

    Do Not Use When:
        - You want to actually train a model - use start_model_training instead
        - You need prediction parameters - use get_model_prediction_schema instead

    Returns:
        A JSON schema dict describing required training parameters, or null if no parameters are required

    Example User Requests:
        - "What parameters do I need to train a disruption CNN model?"
    """
    return ModelRegistry.get_params_schema(
        model, schema_type="training", return_draft_07=True
    )


@router.get(
    "/models/{model}/predict",
    operation_id="get_model_prediction_schema",
    dependencies=[Depends(check_models_enabled)],
)
async def get_model_prediction_schema(model: str) -> dict[str, typing.Any] | None:
    """
    Get params required for predicting with this model.
    ----------------------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Retrieve the required prediction parameter schema for a specific model type.

    Use When:
        - You are about to create model predictions and need to know what parameters are required
        - You want to validate prediction parameters before calling the prediction endpoint

    Do Not Use When:
        - You want to actually run predictions - use create_model_predictions or create_sample_model_predictions instead
        - You need training parameters - use get_model_training_schema instead

    Returns:
        A JSON schema dict describing required prediction parameters, or null if no parameters are required

    Example User Requests:
        - "What parameters do I need to run predictions with th disruption CNN model?"
    """
    return ModelRegistry.get_params_schema(
        model, schema_type="prediction", return_draft_07=True
    )
