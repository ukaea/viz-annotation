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
    LocalLoadParams,
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
from toktagger.api.models.loaders import LocalLoader, GitlabLoader, HuggingfaceLoader

logger = logging.getLogger("ray")
logger.setLevel("DEBUG")


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


@ray.remote(num_cpus=0.1)
def load_model_local(
    model: Model, project: Project, params: LocalLoadParams
) -> dict[str : str | None]:
    model_actor = get_actor(project=project, model=model, use_gpu=False)
    # TODO make LocalLoadParams come from endpoint
    loader = LocalLoader(
        project=project,
        model=model,
        model_actor=model_actor,
        params=params,
    )
    return loader.load()


@ray.remote(num_cpus=0.1)
def load_model_gitlab(
    model: Model, project: Project, params: GitlabLoadParams
) -> tuple[str, str | None]:
    model_actor = get_actor(project=project, model=model, use_gpu=False)
    loader = GitlabLoader(
        project=project, model=model, model_actor=model_actor, params=params
    )
    return loader.load()


@ray.remote(num_cpus=0.1)
def load_model_huggingface(
    model: Model, project: Project, params: HuggingfaceLoadParams
) -> tuple[str, str | None]:
    model_actor = get_actor(project=project, model=model, use_gpu=False)
    loader = HuggingfaceLoader(
        project=project, model=model, model_actor=model_actor, params=params
    )
    return loader.load()


@ray.remote(num_cpus=0.1)
def train_model(
    model: Model,
    project: Project,
    samples: list[Sample],
    annotations: list[list[AnnotationOutTypes]],
    params: pydantic.BaseModel | None,
    use_gpu: bool = False,
):  # TODO: do we want to support retraining where we only get annotations not previously put into model?
    model_actor = get_actor(project=project, model=model, use_gpu=use_gpu)

    if not (models_dir := os.environ.get("MODEL_STORAGE")):
        raise ValueError("Model storage directory not provided to worker node.")
    results_dir = pathlib.Path(models_dir).joinpath(str(model.id))
    results_dir.mkdir(parents=True)

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


@ray.remote(num_cpus=0.1)
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
