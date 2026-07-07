from fastapi import APIRouter, Request, Depends, Path, Query, Body, HTTPException
from fastapi.responses import JSONResponse
import random
from bson.objectid import ObjectId
from toktagger.api.crud import utils
from toktagger.api.schemas.annotations import AnnotationBatchTypes
from toktagger.api.schemas.data import DataParamTypes, DataParams
from toktagger.api.schemas.models import Model, ModelIn, ModelUpdate, LoadTypes
from toktagger.api.models import models_dependencies_installed, check_models_enabled
from pydantic import ValidationError
from collections import defaultdict
import toktagger.api.config as config
import pathlib

# Only import large packages if models dependencies installed
if models_dependencies_installed():
    from toktagger.api.worker import load_model, train_model, get_predictions
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


router = APIRouter(
    prefix="/projects/{project_id}",
    tags=["Models"],
    # Check models are enabled whenever an endpoint is called
    dependencies=[Depends(check_models_enabled)],
)


@router.get("/models")
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
    # Return details about models being used by this project
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


@router.get("/models/{model_type}")
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
    db_client = request.app.state.db_client
    model = await utils.get_model(
        db_client, project_id=project_id, model_type=model_type, version=version
    )
    return model


@router.delete("/models/{model_type}")
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
        config.settings.models.cache_dir.joinpath(f"{model.id}.model").unlink(
            missing_ok=True
        )


@router.get("/models/{model_type}/train")
async def get_training_info(
    request: Request, project_id: str, model_type: str
) -> Model:
    db_client = request.app.state.db_client
    await utils.get_project(db_client, project_id)
    latest_model = await utils.get_model(
        db_client, project_id=project_id, model_type=model_type
    )
    if latest_model.training_status not in ("queued", "started"):
        raise HTTPException(
            status_code=404, detail=f"No training in progress for {model_type}"
        )
    return latest_model


@router.put("/models/{model_type}/train")
async def start_model_training(
    request: Request,
    project_id: str,
    model_type: str,
    use_gpu: bool = Query(False, description="Whether to use GPU to train the model"),
    params: dict = Body(
        {}, description="Optional parameters for training the model", embed=True
    ),
):
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
                if db_model.training_status in ["queued", "started"]
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
        training_status="queued",
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


@router.delete("/models/{model_type}/train")
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
        if model.training_status not in ("started", "queued"):
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
            status="started",
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
            updates=ModelUpdate(training_status="aborted"),
        )

    # Return list of model IDs which were stopped
    return [model.id for model in models]


@router.post("/models/{model_type}/load")
async def load_model_weights(
    request: Request,
    project_id: str,
    model_type: str,
    method: LoadTypes,
    weights_path: str,
):
    db_client = request.app.state.db_client
    task_registry = request.app.state.task_registry

    # Check file available at weights path
    weights_path: pathlib.Path = pathlib.Path(weights_path)
    if not weights_path.exists():
        raise HTTPException(
            status_code=422, detail="Weights file not found at specified path!"
        )

    # Check if that load method is enabled
    if method == LoadTypes.LOCAL and not config.settings.models.local_load_enabled:
        raise HTTPException(
            status_code=403, detail="Loading from local weights is disabled."
        )

    # If local load, create a model db instance and return the model ID
    elif method == LoadTypes.LOCAL:
        project = await utils.get_project(db_client, project_id)
        # Check that this model type is valid for this project
        if model_type not in project.model_types:
            raise HTTPException(
                status_code=422,
                detail=f"This model type is not valid for your current project! Valid types are: {project.model_types}",
            )

        # Try to get model for this project from database if it exists
        db_models = await utils.get_models(db_client, project_id, model_type)

        if (
            len(
                [
                    db_model
                    for db_model in db_models
                    if db_model.training_status in ["queued", "started"]
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
            training_status="queued",
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

        task_registry.update_actors(model.id, False)

        task = load_model.remote(
            project=project,
            model=model,
            weights_path=weights_path,
        )
        task_id = task_registry.register(task)

        # Associate the task ID with the model in the database
        await utils.update_model(
            db_client=db_client, model_id=model_id, updates=ModelUpdate(task_id=task_id)
        )

        return {"task_id": task_id, "model_id": model.id}
    else:
        raise HTTPException(
            status_code=501, detail=f"Loading method {method} not implemented!"
        )


@router.get("/models/{model_type}/load/{task_id}")
async def get_load_model_status(
    request: Request,
    project_id: str = Path(description="The ID of the project to load a model for."),
    model_type: str = Path(description="The type of model to load."),
    task_id: str = Path(description="The load task to get results from."),
) -> bool | str:
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
            # Find model ID of latest in progress model

            await utils.update_model(
                db_client=db_client,
                model_id=model.id,
                updates=ModelUpdate(training_status="failed", progress=0),
            )
            raise HTTPException(
                detail=f"Load task failed unexpectedly - {e}.",
                status_code=500,
            )

        if result.get("message"):
            await utils.update_model(
                db_client=db_client,
                model_id=result["model_id"],
                updates=ModelUpdate(training_status="failed", progress=0),
            )
            raise HTTPException(
                detail=f"Load task failed - {result['message']}.",
                status_code=500,
            )

        # Update model to be completed and ready for predictions
        await utils.update_model(
            db_client=db_client,
            model_id=result["model_id"],
            updates=ModelUpdate(training_status="completed", progress=100),
        )

        return True

    else:
        raise HTTPException(status_code=404, detail="Load task not found with that ID!")


@router.post("/models/{model_type}/predict")
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
    if model.training_status != "completed":
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


@router.delete("/models/{model_type}/predict")
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


@router.post("/samples/{sample_id}/models/{model_type}/predict")
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


@router.get("/samples/{sample_id}/models/{model_type}/predict/{task_id}")
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


@router.put("/models/{model_id}")
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
    # Update model status
    db_client = request.app.state.db_client
    await utils.get_project(db_client, project_id)
    await utils.update_model(
        db_client=db_client, model_id=model_id, updates=model_updates
    )


@router.get("/models/{model_id}/evaluate")
async def evaluate(project_id: str, model_id: str):
    # Get evaluation of model by comparing model predictions to human evaluations
    # Specify samples to use via filters
    # Return overall statistics, as well as correct/incorrect for each sample ID
    pass
