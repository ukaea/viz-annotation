import os
import ray
from ray.exceptions import ActorDiedError
import pathlib
from toktagger.api.schemas.projects import Project
from toktagger.api.schemas.samples import Sample, SampleUpdate, SampleUpdateBatchItem
from toktagger.api.schemas.data import DataParamTypes
from toktagger.api.schemas.annotations import (
    AnnotationBatchTypeAdapter,
    AnnotationOutTypes,
)
from pydantic import ValidationError
from toktagger.api.schemas.models import (
    Model,
    ModelUpdate,
    GitlabLoadParams,
    HuggingfaceLoadParams,
)
from toktagger.api.core.sender import (
    send_batch_samples,
    send_batch_annotations,
    send_model_updates,
)
import logging
import shutil
import pydantic
from mlflow import MlflowClient, MlflowException
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import requests

logger = logging.getLogger("ray")
logger.setLevel("DEBUG")

# Create model storage directory if it doesn't already exist
# Note that we still use env vars here since this is inside a worker node...
if not (models_dir := os.environ.get("MODEL_STORAGE")):
    raise ValueError("Model storage directory not provided to worker node.")

models_dir = pathlib.Path(models_dir)
models_dir.mkdir(parents=True, exist_ok=True)


def get_actor(project: Project, model: Model, use_gpu: bool):
    try:
        logger.info(f"Finding actor for model {model.id}")
        ml_model = ray.get_actor(model.id)
        logger.info("Found existing actor!")
    except ValueError:
        ml_model = None

    # Check if actor has GPU enabled
    if ml_model:
        gpu_available = ray.get(ml_model.gpu_available.remote())
        if use_gpu and not gpu_available:
            # Stop existing actor
            logger.info("Existing Actor does not have access to a GPU!")
            try:
                ray.get(ml_model.__ray_terminate__.remote())
            except ActorDiedError:
                ml_model = None
                logger.info("Stopped existing Actor, restarting with GPU...")

    if not ml_model:
        # Actor not alive, so load from weights
        logger.info("Actor not found, loading from disk...")

        model_registry = ray.get_actor("WorkerModelRegistry")
        model_type = ray.get(model_registry.get.remote(model.type))

        ml_model = (
            ray.remote(num_gpus=1 if use_gpu else 0)(model_type)
            .options(name=model.id, lifetime="detached")
            .remote(
                model_id=str(model.id),
                project=project,
            )
        )

        results_dir = pathlib.Path(os.environ["MODEL_STORAGE"]).joinpath(str(model.id))
        if results_dir.exists():
            ml_model.wrapped_load.remote(results_dir)
        else:
            logger.debug("No saved weights found, initializing blank model")

    return ml_model


def check_safetensor(
    weights_path: str | pathlib.Path, project_id: str, model_id: str
) -> dict[str, str | ModelUpdate] | None:
    if os.environ.get("MODELS_SAFETENSORS_ONLY", "False").lower() == "true":
        try:
            with safe_open(weights_path, framework="pt", device="cpu") as f:
                _ = list(f.keys())
        except Exception:
            # Not a safetensors object, return error
            logger.error(
                "Only permitted to load SafeTensors files due to env var setting!"
            )
            send_model_updates(
                project_id=project_id,
                model_id=model_id,
                updates=ModelUpdate(training_status="failed"),
            )
            return {
                "project_id": project_id,
                "model_id": model_id,
                "message": "retrieved file is not a SafeTensor!",
            }
        return None


def load_model_local(
    model: Model, project: Project, weights_path: pathlib.Path
) -> dict[str : str | None]:
    # Change status to started
    send_model_updates(
        project_id=project.id,
        model_id=model.id,
        updates=ModelUpdate(training_status="started"),
    )

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
    # Check if this file is a safetensor, if required
    unsafe = check_safetensor(weights_path, project.id, model.id)
    if unsafe:
        return unsafe

    model_actor = get_actor(project=project, model=model, use_gpu=False)
    # Try loading actor with weights file, catch and reraise any errors
    try:
        load_temp_weights_task = model_actor.wrapped_load.remote(
            results_dir=weights_path.parent, weights_filename=weights_path.name
        )
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
            "message": str(e),
        }

    # Save the model with the correct dir path
    results_dir = pathlib.Path(os.environ["MODEL_STORAGE"]).joinpath(str(model.id))

    save_weights_task = model_actor.wrapped_save.remote(results_dir)
    ray.get(save_weights_task)

    send_model_updates(
        project_id=project.id,
        model_id=model.id,
        updates=ModelUpdate(training_status="completed", progress=100),
    )

    return {"project_id": project.id, "model_id": model.id, "message": None}


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

    model_actor = get_actor(project=project, model=model, use_gpu=False)

    # Construct URI required
    if not all(
        (
            os.environ.get("MODELS_GITLAB_URL"),
            os.environ.get("MODELS_GITLAB_TOKEN"),
            params.gitlab_project_id,
        )
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
            "message": "required variables not defined.",
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
        except MlflowException as e:
            logger.debug(e)
            mlflow_model = None
    else:
        mlflow_model = max(
            client.get_latest_versions(params.model_name),
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
            "message": "requested version of selected model could not be found!",
        }

    # Download artifacts
    # Note that it seems like download_artifacts and list_artifacts methods are broken
    # https://gitlab.com/gitlab-org/gitlab/-/work_items/591960
    # Will perform a workaround by downloading directly from API
    download_path = model_dir.joinpath(pathlib.Path(params.weights_path).name)
    try:
        with requests.get(
            f"{os.environ.get('MODELS_GITLAB_URL')}/api/v4/projects/{params.gitlab_project_id}/packages/ml_models/{mlflow_model.version}/files/{params.weights_path}",
            headers={"Authorization": f"Bearer {os.environ['MLFLOW_TRACKING_TOKEN']}"},
            stream=True,
            timeout=600,
        ) as response:
            response.raise_for_status()
            with download_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)

    except requests.exceptions.HTTPError:
        if response.status_code == 404:
            err_msg = "could not find model weights at provided file location!"
        elif response.status_code == 403:
            err_msg = "Gitlab token does not have the correct permissions to access this model!"
        else:
            err_msg = "server error from Gitlab!"

        logger.error(f"Failed to load weights - {err_msg}")
        send_model_updates(
            project_id=project.id,
            model_id=model.id,
            updates=ModelUpdate(training_status="failed"),
        )
        return {
            "project_id": project.id,
            "model_id": model.id,
            "message": err_msg,
        }

    except requests.exceptions.Timeout:
        download_path.unlink(missing_ok=True)
        logger.error("Failed to load weights - download timed out!")
        send_model_updates(
            project_id=project.id,
            model_id=model.id,
            updates=ModelUpdate(training_status="failed"),
        )
        return {
            "project_id": project.id,
            "model_id": model.id,
            "message": "download timed out!",
        }

    # Check if this file is a safetensor, if required
    unsafe = check_safetensor(download_path, project.id, model.id)
    if unsafe:
        # Delete downloaded file
        download_path.unlink()
        return unsafe

    # Try loading actor with weights file, catch and reraise any errors
    try:
        load_temp_weights_task = model_actor.wrapped_load.remote(
            results_dir=download_path.parent, weights_filename=download_path.name
        )
        ray.get(load_temp_weights_task)
    except Exception as e:
        download_path.unlink()
        logger.error(e)
        send_model_updates(
            project_id=project.id,
            model_id=model.id,
            updates=ModelUpdate(training_status="failed"),
        )
        return {
            "project_id": project.id,
            "model_id": model.id,
            "message": str(e),
        }

    # Save the model with the correct dir path
    results_dir = pathlib.Path(os.environ["MODEL_STORAGE"]).joinpath(str(model.id))

    save_weights_task = model_actor.wrapped_save.remote(results_dir)
    ray.get(save_weights_task)

    download_path.unlink()

    send_model_updates(
        project_id=project.id,
        model_id=model.id,
        updates=ModelUpdate(training_status="completed", progress=100),
    )

    return {"project_id": project.id, "model_id": model.id, "message": None}


def load_model_huggingface(
    model: Model, project: Project, params: HuggingfaceLoadParams
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

    model_actor = get_actor(project=project, model=model, use_gpu=False)

    # Pull object from Hugging Face
    try:
        weights_path = hf_hub_download(
            repo_id=f"{params.huggingface_userspace}/{params.model_name}",
            filename=params.weights_path,
            revision=params.model_version,
            local_dir=str(model_dir),
        )
    except Exception as e:
        logger.error("Requested model could not be found!")
        logger.error(e)
        send_model_updates(
            project_id=project.id,
            model_id=model.id,
            updates=ModelUpdate(training_status="failed"),
        )
        return {
            "project_id": project.id,
            "model_id": model.id,
            "message": "requested model could not be found!",
        }

    download_path = pathlib.Path(weights_path)

    # Check if this file is a safetensor, if required
    unsafe = check_safetensor(download_path, project.id, model.id)
    if unsafe:
        # Delete downloaded file
        download_path.unlink()
        return unsafe

    # Try loading actor with weights file, catch and reraise any errors
    # Try loading actor with weights file, catch and reraise any errors
    try:
        load_temp_weights_task = model_actor.wrapped_load.remote(
            results_dir=download_path.parent, weights_filename=download_path.name
        )
        ray.get(load_temp_weights_task)
    except Exception as e:
        download_path.unlink()
        logger.error(e)
        send_model_updates(
            project_id=project.id,
            model_id=model.id,
            updates=ModelUpdate(training_status="failed"),
        )
        return {
            "project_id": project.id,
            "model_id": model.id,
            "message": str(e),
        }

    # Save the model with the correct dir path
    results_dir = pathlib.Path(os.environ["MODEL_STORAGE"]).joinpath(str(model.id))

    save_weights_task = model_actor.wrapped_save.remote(results_dir)
    ray.get(save_weights_task)

    download_path.unlink()

    send_model_updates(
        project_id=project.id,
        model_id=model.id,
        updates=ModelUpdate(training_status="completed", progress=100),
    )

    return {"project_id": project.id, "model_id": model.id, "message": None}


def train_model(
    model: Model,
    project: Project,
    samples: list[Sample],
    annotations: list[list[AnnotationOutTypes]],
    params: pydantic.BaseModel | None,
    use_gpu: bool = False,
):  # TODO: do we want to support retraining where we only get annotations not previously put into model?
    model_actor = get_actor(project=project, model=model, use_gpu=use_gpu)
    results_dir = pathlib.Path(os.environ["MODEL_STORAGE"]).joinpath(str(model.id))
    try:
        logger.info(f"Running model training for project {project.id}")
        model_actor.log_progress.remote(training_status="started", progress=0)
        train_task = model_actor.wrapped_train.remote(
            samples=samples, annotations=annotations, params=params
        )

        # Wait for train task to complete
        score = ray.get(train_task)

        save_task = model_actor.wrapped_save.remote(results_dir)
        ray.get(save_task)  # Block until save is done

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

        # Also delete directory of results, if it has already been created
        if results_dir.exists():
            shutil.rmtree(results_dir)

        raise e


def get_predictions(
    project: Project,
    model: Model,
    samples: list[Sample],
    params: pydantic.BaseModel,
    data_params: DataParamTypes | None = None,
    use_gpu: bool = False,
):
    # For a first pass, when you get next sample on the web UI, run the model to get predictions
    # In the future, can improve that for smarter sampling in active learning
    # Where inference is run on some batch of samples first
    logger.info(
        f"Creating predictions for project {project.id} on {len(samples)} samples."
    )
    model_actor = get_actor(project=project, model=model, use_gpu=use_gpu)

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
