import os
import ray
import pathlib
from toktagger.api.schemas.projects import Project
from toktagger.api.schemas.samples import Sample, SampleUpdate, SampleUpdateBatchItem
from toktagger.api.schemas.data import DataParamTypes
from toktagger.api.schemas.annotations import (
    AnnotationBatchTypeAdapter,
    AnnotationOutTypes,
)
from pydantic import ValidationError
from toktagger.api.schemas.models import Model, ModelUpdate, GitlabLoadParams
from toktagger.api.core.sender import (
    send_batch_samples,
    send_batch_annotations,
    send_model_updates,
)
import logging
from platformdirs import user_cache_dir
import pydantic
from mlflow import MlflowClient, MlflowException

logger = logging.getLogger("ray")
logger.setLevel("DEBUG")

models_dir_default = pathlib.Path(user_cache_dir("toktagger", "ukaea")).joinpath(
    "models"
)
models_dir_default.mkdir(parents=True, exist_ok=True)

# Set model storage to default path if not already set
os.environ["MODEL_STORAGE"] = os.environ.get("MODEL_STORAGE", str(models_dir_default))


def get_actor(project, model):
    try:
        logger.info(f"Finding actor for model {model.id}")
        ml_model = ray.get_actor(model.id)
        logger.info("Found existing actor!")
    except ValueError:
        # Actor not alive, so load from weights
        logger.info("Actor not found, loading from disk...")

        model_registry = ray.get_actor("WorkerModelRegistry")
        model_type = ray.get(model_registry.get.remote(model.type))

        ml_model = (
            ray.remote(model_type)
            .options(name=model.id, lifetime="detached")
            .remote(
                model_id=str(model.id),
                project=project,
            )
        )

        model_path = next(
            pathlib.Path(os.environ["MODEL_STORAGE"]).glob(f"{str(model.id)}*"), None
        )
        if model_path:
            ml_model.wrapped_load.remote(model_path)
        else:
            logger.debug("No saved weights found, initializing blank model")

    return ml_model


@ray.remote
def load_model_local(
    model: Model, project: Project, weights_path: pathlib.Path
) -> tuple[str, str | None]:
    # Change status to started
    send_model_updates(
        project_id=project.id,
        model_id=model.id,
        updates=ModelUpdate(training_status="started"),
    )

    # Make sure model storage location in cache dir exists
    model_dir = pathlib.Path(os.environ["MODEL_STORAGE"])
    model_dir.mkdir(exist_ok=True)

    # Check worker can see weights file
    if not weights_path.exists():
        send_model_updates(
            project_id=project.id,
            model_id=model.id,
            updates=ModelUpdate(training_status="failed"),
        )
        return {
            "project_id": project.id,
            "model_id": model.id,
            "message": f"Worker node cannot find weights file at location {weights_path}",
        }

    model_actor = get_actor(project=project, model=model)
    # Try loading actor with weights file, catch and reraise any errors
    try:
        load_temp_weights_task = model_actor.wrapped_load.remote(str(weights_path))
        ray.get(load_temp_weights_task)
    except Exception as e:
        logger.error(e)
        send_model_updates(
            project_id=project.id,
            model_id=model.id,
            updates=ModelUpdate(training_status="failed"),
        )
        return {
            "project_id": project.id,
            "model_id": model.id,
            "message": f"Failed to load weights - {str(e)}",
        }

    # Save the model with the correct file name, delete temporary file
    save_weights_task = model_actor.wrapped_save.remote(
        model_dir.joinpath(str(model.id))
    )
    ray.get(save_weights_task)

    return {"project_id": project.id, "model_id": model.id, "message": None}


@ray.remote
def load_model_gitlab(
    model: Model, project: Project, params: GitlabLoadParams
) -> tuple[str, str | None]:
    # Make sure model storage location in cache dir exists
    model_dir = pathlib.Path(os.environ["MODEL_STORAGE"])
    model_dir.mkdir(exist_ok=True)

    # Change status to started
    send_model_updates(
        project_id=project.id,
        model_id=model.id,
        updates=ModelUpdate(training_status="started"),
    )

    model_actor = get_actor(project=project, model=model)

    # Construct URI required
    if not all(
        os.environ.get("MODELS_GITLAB_URL"),
        os.environ.get("MODELS_GITLAB_TOKEN"),
        params.gitlab_project_id,
    ):
        logger.error(
            "Gitlab URL, Token or Project ID not specified when trying to load ML model!"
        )
        send_model_updates(
            project_id=project.id,
            model_id=model.id,
            updates=ModelUpdate(training_status="failed"),
        )
        return {
            "project_id": project.id,
            "model_id": model.id,
            "message": "Failed to load weights - required variables not defined.",
        }
    os.environ["MLFLOW_TRACKING_URI"] = (
        f"{os.environ.get('MODELS_GITLAB_URL')}/api/v4/projects/{params.gitlab_project_id}/ml/mlflow"
    )
    os.environ["MLFLOW_TRACKING_TOKEN"] = os.environ.get("MODELS_GITLAB_TOKEN")

    # Pull object from ML Model registry
    client = MlflowClient()
    if params.model_version:
        try:
            mlflow_model = client.get_model_version(
                params.model_name, params.model_version
            )
        except MlflowException:
            mlflow_model = None
    else:
        mlflow_model = max(
            client.search_model_versions(f"name='{params.model_name}'"),
            key=lambda mv: int(mv.version),
            default=None,
        )

    if not mlflow_model:
        logger.error("Requested version of selected model could not be found!")
        send_model_updates(
            project_id=project.id,
            model_id=model.id,
            updates=ModelUpdate(training_status="failed"),
        )
        return {
            "project_id": project.id,
            "model_id": model.id,
            "message": "Failed to load weights - requested version of selected model could not be found!",
        }

    # Download artifacts
    weights_path = client.download_artifacts(
        run_id=mlflow_model.run_id, path=params.weights_path, dst_path=str(model_dir)
    )

    # If only safetensors allowed, check it is one
    # if os.environ.get("MODELS_SAFETENSORS_ONLY"):
    #     try:
    #         with safe_open(weights_path, framework="pt", device="cpu") as f:
    #             _ = list(f.keys())
    #     except Exception:
    #         # Not a safetensors object, delete local file and return error
    #         pathlib.Path(weights_path).unlink()
    #         logger.error(
    #             "Only permitted to load SafeTensors files due to env var setting!"
    #         )
    #         send_model_updates(
    #             project_id=project.id,
    #             model_id=model.id,
    #             updates=ModelUpdate(training_status="failed"),
    #         )
    #         return {
    #             "project_id": project.id,
    #             "model_id": model.id,
    #             "message": "Failed to load weights - retrieved file is not a SafeTensor!",
    #         }

    # Try loading actor with weights file, catch and reraise any errors
    try:
        load_temp_weights_task = model_actor.wrapped_load.remote(str(weights_path))
        ray.get(load_temp_weights_task)
    except Exception as e:
        pathlib.Path(weights_path).unlink()
        logger.error(e)
        send_model_updates(
            project_id=project.id,
            model_id=model.id,
            updates=ModelUpdate(training_status="failed"),
        )
        return {
            "project_id": project.id,
            "model_id": model.id,
            "message": f"Failed to load weights - {str(e)}",
        }

    # Save the model with the correct file name, delete temporary file
    save_weights_task = model_actor.wrapped_save.remote(
        model_dir.joinpath(str(model.id))
    )
    ray.get(save_weights_task)

    pathlib.Path(weights_path).unlink()

    return {"project_id": project.id, "model_id": model.id, "message": None}


@ray.remote
def train_model(
    model: Model,
    project: Project,
    samples: list[Sample],
    annotations: list[list[AnnotationOutTypes]],
    params: pydantic.BaseModel | None,
):  # TODO: do we want to support retraining where we only get annotations not previously put into model?
    model_actor = get_actor(project=project, model=model)
    try:
        logger.info(f"Running model training for project {project.id}")
        model_actor.log_progress.remote(training_status="started", progress=0)
        train_task = model_actor.wrapped_train.remote(
            samples=samples, annotations=annotations, params=params
        )

        # Wait for train task to complete
        score = ray.get(train_task)

        model_dir = pathlib.Path(os.environ["MODEL_STORAGE"])
        model_dir.mkdir(exist_ok=True)  # Do i need to do this every time?
        model_actor.wrapped_save.remote(model_dir.joinpath(str(model.id)))

        send_model_updates(
            project_id=project.id,
            model_id=model.id,
            updates=ModelUpdate(training_status="completed", progress=100, score=score),
        )

        return {"project_id": project.id, "model_id": model.id, "score": score}

    except Exception as e:
        # If anything goes wrong, update model to failed status
        # This is important as if this does not happen, your model will be stuck in 'training' forever,
        # Preventing you from ever starting a new training session again. TODO should we have some kind of timeout in case this fails?
        logger.error(e)
        send_model_updates(
            project_id=project.id,
            model_id=model.id,
            updates=ModelUpdate(training_status="failed"),
        )
        raise e


@ray.remote
def get_predictions(
    project: Project,
    model: Model,
    samples: list[Sample],
    params: pydantic.BaseModel,
    data_params: DataParamTypes | None = None,
):
    # For a first pass, when you get next sample on the web UI, run the model to get predictions
    # In the future, can improve that for smarter sampling in active learning
    # Where inference is run on some batch of samples first
    logger.info(
        f"Creating predictions for project {project.id} on {len(samples)} samples."
    )
    model_actor = get_actor(project=project, model=model)

    predictions_task = model_actor.wrapped_predict.remote(
        samples=samples, params=params, data_params=data_params
    )
    predictions = ray.get(predictions_task)

    samples_batch = [
        SampleUpdateBatchItem(
            id=sample.id, updates=SampleUpdate(validated_annotations=False)
        )
        for sample in samples
    ]

    annotations_batch = []
    for sample, annotations in zip(samples, predictions):
        for annotation in annotations:
            annotation = annotation.model_dump(mode="python")
            annotation["sample_id"] = sample.id
            annotation["project_id"] = project.id
            annotation["shot_id"] = sample.shot_id
            annotation["created_by"] = model.type
            try:
                annotation = AnnotationBatchTypeAdapter.validate_python(annotation)
            except ValidationError as e:
                logger.error(f"Failed to validate annotation: {e}")
            annotations_batch.append(annotation)

    # Return predictions over rest API to server
    send_batch_samples(project.id, samples_batch)
    send_batch_annotations(project.id, annotations_batch)

    logger.info(f"Predictions for project {project.id} complete!")

    return {
        "project_id": project.id,
        "model_type": model.type,
        "samples_batch": samples_batch,
        "annotations_batch": annotations_batch,
    }
