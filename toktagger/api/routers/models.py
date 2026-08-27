from fastapi import APIRouter, Request, Depends, Path, Query, Body, HTTPException
from fastapi.responses import JSONResponse
import random
from bson.objectid import ObjectId
from toktagger.api.crud import utils
from toktagger.api.schemas.annotations import AnnotationBatchTypes
from toktagger.api.schemas.data import DataParamTypes, DataParams
from toktagger.api.schemas.models import (
    Model,
    ModelIn,
    ModelUpdate,
    LocalLoadParams,
    GitlabLoadParams,
    HuggingfaceLoadParams,
)
from toktagger.api.schemas.projects import Project
from toktagger.api.models import models_dependencies_installed, check_models_enabled
from pydantic import ValidationError
from collections import defaultdict
import toktagger.api.config as config
import pathlib
import shutil

# Only import large packages if models dependencies installed
if models_dependencies_installed():
    from toktagger.api.core.worker import (
        load_model_local,
        load_model_gitlab,
        load_model_huggingface,
        train_model,
        get_predictions,
    )
    from toktagger.api.models.base import ModelRegistry
    import ray

import logging

logger = logging.getLogger(__name__)


def validate_model_params(model_type: str, schema_type: str, params: dict):
    # Get model params model from registry and validate
    params_model = ModelRegistry.get_params(model_type, schema_type)
    if params_model and not params:
        raise HTTPException(
            status_code=422,
            detail=f"Model training parameters are missing! Requires '{params_model.__name__}' parameters.",
        )
    try:
        params_validated = params_model.model_validate(params) if params_model else None
    except ValidationError as e:
        error_str = ""
        for error in e.errors():
            loc = error.get("loc", [])
            msg = error.get("msg", "Invalid Field!")
            error_str += f"'{loc[0] if len(loc) == 1 else loc}': {msg} \n"
        raise HTTPException(
            status_code=422,
            detail=error_str,
        )
    return params_validated


async def create_model(db_client, project: Project, model_type: str) -> Model:
    # Check that this model type is valid for this project
    if model_type not in project.model_types:
        raise HTTPException(
            status_code=422,
            detail=f"This model type is not valid for your current project! Valid types are: {project.model_types}",
        )

    # Try to get model for this project from database if it exists
    db_models = await utils.get_models(db_client, project.id, model_type)

    if (
        len(
            [
                db_model
                for db_model in db_models
                if db_model.status in ["queued", "training", "loading"]
            ]
        )
        > 0
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Training or loading of {model_type} model already in progress!",
        )

    if len(db_models) == 0:
        # This is the first time a model has been saved for this project, so version = 1
        version = 1
    else:
        version = db_models[0].version + 1

    model_in = ModelIn(
        type=model_type,
        version=version,
        status="queued",
        progress=0,
        score=0,
    )

    model_id = await utils.add_model(
        db_client=db_client, project_id=project.id, model=model_in
    )

    # Find the latest queued model for this project
    model = await utils.get_model(
        db_client, project.id, model_type=model_type, model_id=model_id
    )

    return model


router = APIRouter(
    prefix="/projects/{project_id}",
    tags=["Models"],
    # Check models are enabled whenever an endpoint is called
    dependencies=[Depends(check_models_enabled)],
)


@router.get("/models", operation_id="get_trained_models")
async def get_models(
    request: Request,
    project_id: str = Path(description="The ID of the project to get models for."),
    start: int = Query(
        0,
        description="Index of the first model you want returned when sorted by version",
    ),
    end: int = Query(
        None,
        description="Index of the last model you want returned when sorted by version, leave blank to return all entries",
    ),
) -> list[Model]:
    """
    Return details about models being used by this project.
    --------------------------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Retrieve a list of trained/loaded models for a project with version information.

    Use When:
        - You want to see which models have been trained or loaded for a project
        - You need to know available model versions before making predictions
        - You are tracking model training progress and accuracy scores

    Do Not Use When:
        - You need a single model's details — use toktagger_read_get_model (GET by model_type) instead
        - You want to start training — use toktagger_start_model_training instead
        - You are querying the project itself — use toktagger_read_get_projects instead

    Returns:
        A list of Model objects with: id, type, version, status, progress, score, project_id

    Example User Requests:
        - "What models have been trained for this project?"
        - "Show me all model versions for disruption_cnn"
    """
    # Could be eg the ID, type of model, the accuracy, the version. link to mlflow / simvue instance, etc...
    db_client = request.app.state.db_client
    models = await utils.get_models(
        db_client=db_client,
        project_id=project_id,
        model_type=None,
        start=start,
        end=end,
    )
    return models


@router.get("/models/{model_type}", operation_id="get_model")
async def get_model(
    request: Request,
    project_id: str = Path(description="The ID of the project to get models for."),
    model_type: str = Path(
        description="The type of model to return information about."
    ),
    version: int = Query(
        None,
        description="The version of the model to return, leave blank to return the latest model.",
    ),
) -> Model:
    """
    Get details about a specific model type.
    -----------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Retrieve full details about a specific model version for a project.

    Use When:
        - You need the status, version, and score of a trained model
        - You are checking if a model is ready for predictions (status = "completed")
        - You want to verify a specific model version exists

    Do Not Use When:
        - You need all models — use toktagger_read_get_trained_models (GET list) instead
        - You want to train a model — use toktagger_start_model_training instead

    Returns:
        A Model object with: id, type, version, status, progress, score, project_id

    Example User Requests:
        - "What is the status of the disruption_cnn model?"
        - "Show me version 1 of this model"
    """
    db_client = request.app.state.db_client
    model = await utils.get_model(
        db_client, project_id=project_id, model_type=model_type, version=version
    )
    return model


@router.delete("/models/{model_type}", operation_id="delete_models")
async def delete_models(
    request: Request,
    project_id: str = Path(description="The ID of the project to get models for."),
    model_type: str = Path(description="The type of model to delete."),
    version: int = Query(
        None,
        description="The version of the model to delete, leave blank to delete all models",
    ),
):
    db_client = request.app.state.db_client

    await utils.get_project(db_client, project_id)

    if version:
        models_to_delete = [
            await utils.get_model(
                db_client, project_id=project_id, model_type=model_type, version=version
            )
        ]
    else:
        models_to_delete = await utils.get_models(db_client, project_id, model_type)

    if not models_to_delete:
        raise HTTPException(
            status_code=404,
            detail=f"Version {version} of model type {model_type} not found!"
            if version
            else f"No models of type {model_type} found for this project!",
        )

    # Delete from DB
    for model in models_to_delete:
        await utils.delete_model(
            db_client=db_client, project_id=project_id, model_id=model.id
        )

        # And delete file from storage (if it exists - may not if the job failed)
        model_dir = config.settings.models.cache_dir.joinpath(model.id)
        if model_dir.exists():
            shutil.rmtree(model_dir)


@router.get("/models/{model_type}/train", operation_id="get_model_training_info")
async def get_training_info(
    request: Request, project_id: str, model_type: str
) -> Model:
    """
    Get information about an in-progress model training job.
    --------------------------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Check the status of a model training job that is currently queued, training, or loading weights.

    Use When:
        - You started training and want to check progress
        - You need to verify a training job is still running
        - You are polling for training completion

    Do Not Use When:
        - You want to start training — use toktagger_start_model_training instead
        - The training is already complete — use toktagger_read_get_model instead
        - You want to stop training — use the DELETE endpoint (stop_model_training) instead

    Returns:
        A Model object if training is in progress; raises 404 if no training is active

    Example User Requests:
        - "How far along is the disruption_cnn training?"
        - "Is the model still training?"
    """
    db_client = request.app.state.db_client
    await utils.get_project(db_client, project_id)
    latest_model = await utils.get_model(
        db_client, project_id=project_id, model_type=model_type
    )
    if latest_model.status not in ("queued", "training", "loading"):
        raise HTTPException(
            status_code=404, detail=f"No training in progress for {model_type}"
        )
    return latest_model


@router.put("/models/{model_type}/train", operation_id="start_model_training")
async def start_model_training(
    request: Request,
    project_id: str,
    model_type: str,
    use_gpu: bool = Query(False, description="Whether to use GPU to train the model"),
    params: dict = Body(
        {}, description="Optional parameters for training the model", embed=True
    ),
):
    """
    Start Model Training.
    --------------------

    MCP Documentation
    -----------------
    Purpose:
        Begin training an ML model on validated annotations for a project.

    Use When:
        - You have enough validated samples/annotations to train a model
        - You want to use GPU acceleration for faster training
        - You are setting up a model for future predictions

    Do Not Use When:
        - There are no validated annotations — the endpoint returns 404
        - Training for this model type is already in progress — returns 409
        - You want to load pre-trained weights instead — use one of the load_model_weights_* endpoints

    Returns:
        A dict with task_id and model_id for tracking training progress

    Example User Requests:
        - "Start training the disruption_cnn model"
        - "Train this model on GPU with custom hidden layers"
    """
    db_client = request.app.state.db_client
    task_registry = request.app.state.task_registry

    # If GPU requested but not available, return error
    if use_gpu and not task_registry.gpu_enabled:
        raise HTTPException(
            status_code=409,
            detail="GPU was requested but GPU support not enabled on server!",
        )

    project = await utils.get_project(db_client, project_id)
    # Check that this model type is valid for this project
    if model_type not in project.model_types:
        raise HTTPException(
            status_code=422,
            detail=f"This model type is not valid for your current project! Valid types are: {project.model_types}",
        )

    # Get model params model from registry and validate
    params_validated = validate_model_params(model_type, "training", params)

    # Get annotations and samples
    annotations = await utils.get_annotations(db_client, project.id, validated=True)
    samples = await utils.get_samples(db_client, project.id, validated=True)

    # Get all validated samples and annotations for this project
    logger.info(f"Collected {len(annotations)} annotations.")
    logger.info(f"Collected {len(samples)} samples.")

    if len(samples) == 0:
        raise HTTPException(
            status_code=404, detail="No validated samples found to train a model on!"
        )
    if len(annotations) == 0:
        raise HTTPException(
            status_code=404,
            detail="No validated annotations found to train a model on!",
        )

    # Create model
    # Try to get model for this project from database if it exists
    db_models = await utils.get_models(db_client, project_id, model_type)

    if (
        len(
            [
                db_model
                for db_model in db_models
                if db_model.status in ["queued", "training", "loading"]
            ]
        )
        > 0
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Training of {model_type} model already in progress!",
        )

    if len(db_models) == 0:
        # This is the first time a model has been saved for this project, so version = 1
        version = 1
    else:
        version = db_models[0].version + 1

    model_in = ModelIn(
        type=model_type,
        version=version,
        status="queued",
        progress=0,
        score=0,
    )

    model_id = await utils.add_model(
        db_client=db_client, project_id=project.id, model=model_in
    )

    # Split annotations into 2D list, so annotations[idx] is a list of annotations for samples[idx]
    sample_annotations_mapping = defaultdict(list)
    for annotation in annotations:
        sample_annotations_mapping[annotation.sample_id].append(annotation)
    annotations_2d = [sample_annotations_mapping[sample.id] for sample in samples]

    model = Model(**model_in.model_dump(), id=model_id, project_id=project.id)

    task_registry.update_actors(model.id, use_gpu)

    train_task = train_model.remote(
        model=model,
        project=project,
        samples=samples,
        annotations=annotations_2d,
        params=params_validated,
        use_gpu=use_gpu,
    )

    task_id = task_registry.register(train_task)

    # Associate the task ID with the model in the database
    await utils.update_model(
        db_client=db_client, model_id=model_id, updates=ModelUpdate(task_id=task_id)
    )

    return {"task_id": task_id, "model_id": model_id}


@router.delete("/models/{model_type}/train", operation_id="stop_model_training")
async def stop_model_training(
    request: Request,
    project_id: str,
    model_type: str,
    version: int | None = Query(
        None, description="Version of model to use, leave blank for latest version"
    ),
):
    db_client = request.app.state.db_client
    task_registry = request.app.state.task_registry

    # If version provided, get only that model
    if version:
        model = await utils.get_model(
            db_client, project_id, model_type=model_type, version=version
        )
        if model.status not in ("queued", "training", "loading"):
            raise HTTPException(
                status_code=409,
                detail="Model training is not in progress for this model!",
            )
        models = [model]
    else:
        # Get models which are either queued or in progress
        models = await utils.get_models(
            db_client=db_client,
            project_id=project_id,
            model_type=model_type,
            status="queued",
        )
        models += await utils.get_models(
            db_client=db_client,
            project_id=project_id,
            model_type=model_type,
            status="training",
        )

    # Get the task IDs and stop them
    for model in models:
        if model.task_id:
            task = task_registry.get(model.task_id)
            if task is not None:
                ray.cancel(task)
            try:
                actor = ray.get_actor(model.id)
                ray.kill(actor)
            except ValueError:
                pass
        await utils.update_model(
            db_client=db_client,
            model_id=model.id,
            updates=ModelUpdate(status="aborted"),
        )

    # Return list of model IDs which were stopped
    return [model.id for model in models]


@router.post("/models/{model_type}/load/local", operation_id="load_model_weights_local")
async def load_model_weights_local(
    request: Request, project_id: str, model_type: str, params: LocalLoadParams
):
    """
    Load Model Weights Local.
    -------------------------

    MCP Documentation
    -----------------
    Purpose:
        Load pre-trained model weights from a local file system path.

    Use When:
        - You have model weights saved on the server's local disk
        - You want to use a locally trained model for predictions
        - You are running in an offline environment without GitLab/HuggingFace access

    Do Not Use When:
        - The weights file doesn't exist at the specified path — returns 422
        - Local loading is disabled in config — returns 403
        - You want to load from GitLab or HuggingFace — use those endpoints instead

    Returns:
        A dict with task_id and model_id for tracking load progress

    Example User Requests:
        - "Load model weights from /path/to/model.pt"
        - "Import the locally trained disruption model"
    """
    db_client = request.app.state.db_client
    task_registry = request.app.state.task_registry

    # Check file available at weights path
    weights_path: pathlib.Path = pathlib.Path(params.weights_path)
    if not weights_path.exists():
        raise HTTPException(
            status_code=422, detail="Weights file not found at specified path!"
        )

    # Check if that load method is enabled
    if not config.settings.models.local_load_enabled:
        raise HTTPException(
            status_code=403, detail="Loading from local weights is disabled."
        )

    project = await utils.get_project(db_client, project_id)
    model = await create_model(db_client, project, model_type)

    task = load_model_local.remote(project=project, model=model, params=params)
    task_id = task_registry.register(task)
    task_registry.update_actors(model.id, use_gpu=False)

    # Associate the task ID with the model in the database
    await utils.update_model(
        db_client=db_client, model_id=model.id, updates=ModelUpdate(task_id=task_id)
    )

    return {"task_id": task_id, "model_id": model.id}


@router.post(
    "/models/{model_type}/load/gitlab", operation_id="load_model_weights_gitlab"
)
async def load_model_weights_gitlab(
    request: Request, project_id: str, model_type: str, params: GitlabLoadParams
):
    """
    Load Model Weights Gitlab.
    --------------------------

    MCP Documentation
    -----------------
    Purpose:
        Load pre-trained model weights from a GitLab project/artifacts.

    Use When:
        - You want to load weights hosted on a GitLab project
        - You are using a shared model repository via GitLab
        - GitLab loading is enabled and configured on the server

    Do Not Use When:
        - GitLab loading is disabled — returns 403
        - Required env vars (GITLAB_URL, GITLAB_TOKEN) are not set — returns 409
        - You want to load from local files or HuggingFace — use those endpoints instead

    Returns:
        A dict with task_id and model_id for tracking load progress

    Example User Requests:
        - "Load the model from the shared GitLab repository"
        - "Import disruption_cnn weights from GitLab"
    """
    db_client = request.app.state.db_client
    task_registry = request.app.state.task_registry

    # Check if Gitlab load method is enabled
    if not config.settings.models.gitlab_load_enabled:
        raise HTTPException(
            status_code=403, detail="Loading model weights from Gitlab is disabled."
        )

    # Check if required env vars have been set
    if not all(
        (config.settings.models.gitlab_url, config.settings.models.gitlab_token)
    ):
        raise HTTPException(
            status_code=409,
            detail="Gitlab URL and Token env vars must be set for ML Model loading from Gitlab.",
        )
    if (
        not params.gitlab_project_id
        and config.settings.models.gitlab_project_id is None
    ):
        raise HTTPException(
            status_code=422,
            detail="Must set a Gitlab Project ID either via UI or config setting.",
        )
    elif config.settings.models.gitlab_project_id:
        params.gitlab_project_id = config.settings.models.gitlab_project_id

    project = await utils.get_project(db_client, project_id)
    model = await create_model(db_client, project, model_type)

    task = load_model_gitlab.remote(project=project, model=model, params=params)

    task_id = task_registry.register(task)
    task_registry.update_actors(model.id, use_gpu=False)

    # Associate the task ID with the model in the database
    await utils.update_model(
        db_client=db_client, model_id=model.id, updates=ModelUpdate(task_id=task_id)
    )

    return {"task_id": task_id, "model_id": model.id}


@router.post(
    "/models/{model_type}/load/hugging_face",
    operation_id="load_model_weights_hugging_face",
)
async def load_model_weights_hugging_face(
    request: Request, project_id: str, model_type: str, params: HuggingfaceLoadParams
):
    """
    Load Model Weights Hugging Face.
    ---------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Load pre-trained model weights from a Hugging Face model repository.

    Use When:
        - You want to load weights from a Hugging Face model hub repository
        - You are using community-trained tokamak disruption prediction models
        - HuggingFace loading is enabled and configured on the server

    Do Not Use When:
        - HuggingFace loading is disabled — returns 403
        - Required userspace/organization is not configured — returns 422
        - You want to load from local files or GitLab — use those endpoints instead

    Returns:
        A dict with task_id and model_id for tracking load progress

    Example User Requests:
        - "Load the disruption_cnn model from Hugging Face"
        - "Import model from user/disruption-cnn-v1"
    """
    db_client = request.app.state.db_client
    task_registry = request.app.state.task_registry

    # Check if HF load method is enabled
    if not config.settings.models.huggingface_load_enabled:
        raise HTTPException(
            status_code=403,
            detail="Loading model weights from Hugging Face is disabled.",
        )
    # Check if config setting limits userspace to load from
    if (
        not params.huggingface_userspace
        and config.settings.models.huggingface_userspace is None
    ):
        raise HTTPException(
            status_code=422,
            detail="Must set a Hugging Face userspace / organisation to load from, either via UI or config setting.",
        )
    elif config.settings.models.huggingface_userspace:
        params.huggingface_userspace = config.settings.models.huggingface_userspace

    project = await utils.get_project(db_client, project_id)
    model = await create_model(db_client, project, model_type)

    task = load_model_huggingface.remote(project=project, model=model, params=params)

    task_id = task_registry.register(task)
    task_registry.update_actors(model.id, use_gpu=False)

    # Associate the task ID with the model in the database
    await utils.update_model(
        db_client=db_client, model_id=model.id, updates=ModelUpdate(task_id=task_id)
    )

    return {"task_id": task_id, "model_id": model.id}


@router.get("/models/{model_type}/load/{task_id}", operation_id="get_load_model_status")
async def get_load_model_status(
    request: Request,
    project_id: str = Path(description="The ID of the project to load a model for."),
    model_type: str = Path(description="The type of model to load."),
    task_id: str = Path(description="The load task to get results from."),
) -> bool | str:
    """
    Get the status of a model weight loading task.
    -----------------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Check whether a model weight loading task is still queued, in progress, or completed.

    Use When:
        - You started loading weights and need to check if it's finished
        - You are polling for load completion before making predictions
        - You want to detect load failures

    Do Not Use When:
        - You want to actually load weights — use the load endpoint (toktagger_load_model_weights_local, etc.) instead
        - You want to check training status — use toktagger_read_get_model_training_info instead

    Returns:
        true on success, or HTTP 202 with {"message": "Load task in the queue!"} while loading

    Example User Requests:
        - "Has the model loading finished?"
        - "Check the status of the weight loading task"
    """
    db_client = request.app.state.db_client
    task_registry = request.app.state.task_registry

    project = await utils.get_project(db_client, project_id)

    if model_type not in project.model_types:
        raise HTTPException(
            status_code=422,
            detail=f"This model type is not valid for your current project! Valid types are: {project.model_types}",
        )

    # Check whether predictions are complete
    task = task_registry.get(task_id)
    if task is None:
        raise HTTPException(detail="Load task not found with that ID!", status_code=404)

    ready, waiting = ray.wait([task], timeout=0)

    if waiting:
        return JSONResponse(
            content={"message": "Load task in the queue!"}, status_code=202
        )
    elif ready:
        # Get model which has this task ID associated
        model = await utils.get_model(
            db_client,
            project_id,
            model_type=model_type,
            task_id=task_id,
        )
        try:
            result: dict[str, str | None] = ray.get(task)

        except Exception as e:
            err_lines = str(e).strip().splitlines()
            err_msg = err_lines[-1] if err_lines else repr(e)
            await utils.update_model(
                db_client=db_client,
                model_id=model.id,
                updates=ModelUpdate(status="failed", progress=0),
            )
            raise HTTPException(
                detail=f"Load task failed unexpectedly - {err_msg}",
                status_code=500,
            )

        if result.get("message"):
            await utils.update_model(
                db_client=db_client,
                model_id=result["model_id"],
                updates=ModelUpdate(status="failed", progress=0),
            )
            raise HTTPException(
                detail=f"Failed to load weights - {result['message']}",
                status_code=500,
            )

        return True

    else:
        raise HTTPException(status_code=404, detail="Load task not found with that ID!")


@router.post("/models/{model_type}/predict", operation_id="create_model_predictions")
async def predict(
    request: Request,
    project_id: str = Path(description="The ID of the project to get models for."),
    model_type: str = Path(description="The type of model to use for predictions."),
    version: int = Query(
        None, description="Version of model to use, leave blank for latest version"
    ),
    num_predictions: int = Query(
        20,
        description="The maximum number of samples to make predictions for, default is 20.",
    ),
    sample_ids: list[str] = Query(
        None,
        description="A list of specific sample IDs to make predictions for, leave blank for random selection.",
    ),
    use_gpu: bool = Query(
        False, description="Whether to use GPU to create these predictions"
    ),
    params: dict = Body(
        {}, description="Optional parameters for training the model", embed=True
    ),
):
    """
    Predict.
    -------

    MCP Documentation
    -----------------
    Purpose:
        Run inference with a trained model on selected samples to generate predicted annotations.

    Use When:
        - You have a trained model and want to generate predictions on unannotated samples
        - You want to pre-populate annotations for human review
        - You need model-assisted labeling for batch samples
        - You want to evaluate model performance on new data

    Do Not Use When:
        - The model hasn't finished training (status != "completed") — returns 409
        - You want predictions for a single specific sample — use toktagger_create_sample_model_predictions instead
        - You are querying model info — use toktagger_read_get_trained_models instead

    Returns:
        A dict with task_id for tracking prediction progress

    Example User Requests:
        - "Run the disruption_cnn model on 20 samples"
        - "Generate predictions for these specific samples on GPU"
    """
    db_client = request.app.state.db_client
    task_registry = request.app.state.task_registry

    # If GPU requested but not available, return error
    if use_gpu and not task_registry.gpu_enabled:
        raise HTTPException(
            status_code=409,
            detail="GPU was requested but GPU support not enabled on server!",
        )

    project = await utils.get_project(db_client, project_id)

    if model_type not in project.model_types:
        raise HTTPException(
            status_code=422,
            detail=f"This model type is not valid for your current project! Valid types are: {project.model_types}",
        )

    # Find the latest created model for this project
    model = await utils.get_model(
        db_client,
        project_id,
        model_type=model_type,
        status="completed",
        version=version,
    )
    if model.status != "completed":
        raise HTTPException(
            status_code=409,
            detail="Cannot make predictions using a model version which has not successfully finished training.",
        )

    # Get model params model from registry and validate
    params_validated = validate_model_params(model_type, "prediction", params)

    # Create predictions using the given model for this project
    # Predict on samples as specified by filters
    # Stores results in the database with validated=False
    if not sample_ids:
        # Get samples with no human annotations
        selected_samples = await utils.get_samples(
            db_client, project.id, validated=False
        )
    else:
        selected_samples = [
            await utils.get_sample(db_client, project_id, sample_id)
            for sample_id in sample_ids
        ]
        if None in selected_samples:
            selected_samples.remove(
                None
            )  # Better way to handle this if user provides non existant sample IDs?
    if len(selected_samples) == 0:
        raise HTTPException(
            status_code=404, detail="No samples found to perform predictions on!"
        )
    elif num_predictions > len(selected_samples):
        samples = selected_samples
    else:
        samples = random.sample(selected_samples, num_predictions)

    task_registry.update_actors(model.id, use_gpu)

    predict_task = get_predictions.remote(
        project=project,
        model=model,
        samples=samples,
        params=params_validated,
        use_gpu=use_gpu,
    )
    task_id = task_registry.register(predict_task)

    return {"task_id": task_id}


@router.delete("/models/{model_type}/predict", operation_id="delete_model_predictions")
async def delete_predictions(
    request: Request,
    project_id: str = Path(description="The ID of the project to get models for."),
    model_type: str = Path(description="The type of model to delete predictions from."),
):
    db_client = request.app.state.db_client
    # Delete predictions using the given model for this project
    # Predict on samples as specified by filters
    project = await utils.get_project(db_client, project_id)

    if model_type not in project.model_types:
        raise HTTPException(
            status_code=422,
            detail=f"This model type is not valid for your current project! Valid types are: {project.model_types}",
        )

    result = await request.app.state.db_client.delete_filtered_documents(
        collection="annotations",
        filters={"project_id": ObjectId(project.id), "created_by": model_type},
    )

    if result.deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No annotations produced by {model_type} could be found for this Project.",
        )


@router.post(
    "/samples/{sample_id}/models/{model_type}/predict",
    operation_id="create_sample_model_predictions",
)
async def create_sample_predictions(
    request: Request,
    project_id: str = Path(
        description="The ID of the project to make model predictions for."
    ),
    sample_id: str = Path(
        description="The ID of the sample to make model predictions for."
    ),
    model_type: str = Path(description="The type of model to make predictions from."),
    use_gpu: bool = Query(
        False, description="Whether to use GPU to create these predictions"
    ),
    params: dict = Body(
        {}, description="Optional parameters for training the model", embed=True
    ),
    data_params: DataParamTypes = Body(
        DataParams(), description="Data parameters fort this sample", embed=True
    ),
) -> dict[str, str]:
    """
    Create Sample Predictions.
    --------------------------

    MCP Documentation
    -----------------
    Purpose:
        Run model inference on a single specific sample to generate predicted annotations.

    Use When:
        - You want predictions for one specific sample before annotating it
        - You are comparing model predictions against your own annotations
        - You need quick inference for a single data point

    Do Not Use When:
        - You need batch predictions across many samples — use toktagger_create_model_predictions instead
        - The model isn't trained — use toktagger_start_model_training first
        - You are querying the project — use toktagger_read_get_projects instead

    Returns:
        A dict with task_id for tracking prediction progress

    Example User Requests:
        - "Get predictions for shot 30421 using the disruption_cnn model"
        - "Show me the model prediction for this sample before I annotate it"
    """
    db_client = request.app.state.db_client
    task_registry = request.app.state.task_registry

    # If GPU requested but not available, return error
    if use_gpu and not task_registry.gpu_enabled:
        raise HTTPException(
            status_code=409,
            detail="GPU was requested but GPU support not enabled on server!",
        )

    project = await utils.get_project(db_client, project_id)

    if model_type not in project.model_types:
        raise HTTPException(
            status_code=422,
            detail=f"This model type is not valid for your current project! Valid types are: {project.model_types}",
        )

    # Find the latest created model for this project
    model = await utils.get_model(
        db_client, project_id=project.id, model_type=model_type, status="completed"
    )

    # Get model params model from registry and validate
    params_validated = validate_model_params(model_type, "prediction", params)

    sample = await utils.get_sample(db_client, project_id, sample_id)

    task_registry.update_actors(model.id, use_gpu)

    task = get_predictions.remote(
        project=project,
        model=model,
        samples=[sample],
        params=params_validated,
        data_params=data_params,
        use_gpu=use_gpu,
    )
    task_id = task_registry.register(task)

    return {"task_id": task_id}


@router.get(
    "/samples/{sample_id}/models/{model_type}/predict/{task_id}",
    operation_id="get_sample_model_predictions",
)
async def get_sample_predictions(
    request: Request,
    project_id: str = Path(
        description="The ID of the project to get model predictions for."
    ),
    sample_id: str = Path(
        description="The ID of the sample to get model predictions for."
    ),
    model_type: str = Path(description="The type of model to get predictions from."),
    task_id: str = Path(description="The prediction task to get results from."),
) -> list[AnnotationBatchTypes]:
    """
    Get model prediction results for a sample.
    ------------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Retrieve the annotation predictions produced by a model for a specific sample.

    Use When:
        - You started a prediction task and need to fetch the results
        - You want to see model predictions before human annotation
        - You are building a comparison UI (model vs. human annotations)

    Do Not Use When:
        - You want to create predictions — use toktagger_create_sample_model_predictions instead
        - You want to load model weights — use the load_model_weights_* endpoints instead

    Returns:
        A list of predicted Annotation objects for the specified sample

    Example User Requests:
        - "Show me the predictions from task abc123"
        - "What did the disruption_cnn model predict for this sample?"
    """
    db_client = request.app.state.db_client
    task_registry = request.app.state.task_registry

    project = await utils.get_project(db_client, project_id)

    if model_type not in project.model_types:
        raise HTTPException(
            status_code=422,
            detail=f"This model type is not valid for your current project! Valid types are: {project.model_types}",
        )

    await utils.get_sample(db_client, project_id, sample_id)

    # Check whether predictions are complete
    task = task_registry.get(task_id)
    if task is None:
        raise HTTPException(
            detail="Predict task not found with that ID!", status_code=404
        )

    ready, waiting = ray.wait([task], timeout=0)

    if waiting:
        return JSONResponse(
            content={"message": "Predict task in the queue!"}, status_code=202
        )
    elif ready:
        try:
            result = ray.get(task)
        except Exception as e:
            raise HTTPException(
                detail="Predict task failed - no predictions available",
                status_code=500,
            ) from e

        # Check project ID and model type match those expected by user
        if result["project_id"] != project_id:
            raise HTTPException(
                detail="Project ID for this task does not match!", status_code=422
            )

        # Check model type matches
        if result["model_type"] != model_type:
            raise HTTPException(
                detail="Model used for this task does not match!", status_code=422
            )

        prediction_annotations = result.get("annotations_batch")

        # Check that annotations contain results for this sample ID
        if prediction_annotations and not all(
            ann.sample_id == sample_id for ann in prediction_annotations
        ):
            raise HTTPException(
                status_code=404,
                detail="This task does not have results for the specified sample!",
            )

        return prediction_annotations
    else:
        raise HTTPException(
            status_code=404, detail="Predict task not found with that ID!"
        )


@router.put("/models/{model_id}", operation_id="update_model_status")
async def update_model(
    request: Request,
    model_updates: ModelUpdate,
    project_id: str = Path(
        description="The ID of the project to make model predictions for."
    ),
    model_id: str = Path(
        description="The ID of the model to update information about."
    ),
) -> None:
    """
    Update Model.
    -------------

    MCP Documentation
    -----------------
    Purpose:
        Update model status fields (e.g. progress, score, status) after training or loading completes.

    Use When:
        - You need to manually update a model's progress or score
        - A background task completed and needs to report results
        - You are syncing model metadata from an external tracking system (MLflow, etc.)

    Do Not Use When:
        - You want to train a new model — use toktagger_start_model_training instead
        - You want to load weights — use the load_model_weights_* endpoints instead
        - You are querying model info — use toktagger_read_get_model or toktagger_read_get_trained_models instead

    Returns:
        None (no response body on success)

    Example User Requests:
        - "Update the disruption_cnn model to completed with score 0.95"
        - "Sync the model progress from MLflow"
    """
    db_client = request.app.state.db_client
    await utils.get_project(db_client, project_id)
    await utils.update_model(
        db_client=db_client, model_id=model_id, updates=model_updates
    )


# @router.get("/models/{model_id}/evaluate", operation_id="evaluate")
# async def evaluate(project_id: str, model_id: str):
#     # Get evaluation of model by comparing model predictions to human evaluations
#     # Specify samples to use via filters
#     # Return overall statistics, as well as correct/incorrect for each sample ID
#     pass
