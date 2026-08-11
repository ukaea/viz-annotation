from toktagger.api.schemas.samples import Sample
from toktagger.api.schemas.annotations import Annotation, AnnotationBase
from toktagger.api.schemas.projects import Project, Task
from toktagger.api.schemas.data import DataParamTypes
from toktagger.api.core.data_loaders import DataLoader
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
import inspect
import typing
import math
from toktagger.api.core.sender import send_model_updates
from toktagger.api.schemas.models import ModelUpdate
import pydantic
import uuid
from collections import OrderedDict
import logging
import pathlib

from toktagger.api.models import models_dependencies_installed

logger = logging.getLogger("ray")

if models_dependencies_installed():
    import ray


# Recursively walk through schema, finding things which need to be changed
def _update_schema(schema: dict) -> dict:
    """Mutates schema in place and returns draft-7 compliant version."""
    # Convert $defs to definitions
    if "$defs" in schema:
        defs = schema.pop("$defs")
        if "definitions" in schema:
            schema["definitions"].update(defs)
        else:
            schema["definitions"] = defs

    # Convert prefixItems to items, items to additionalItems
    if "prefixItems" in schema:
        additional_items = schema.pop("items", None)
        schema["items"] = schema.pop("prefixItems")
        if additional_items is not None:
            schema["additionalItems"] = additional_items

    # Remove unevaluatedProperties or unevaluatedItems
    schema.pop("unevaluatedProperties", None)
    schema.pop("unevaluatedItems", None)

    return schema


def walk_schema(schema: dict | list) -> dict | list:
    """Walk through a JSON Schema and update relevant items."""
    if isinstance(schema, list):
        schema = [walk_schema(item) for item in schema]

    elif isinstance(schema, dict):
        for key, value in list(schema.items()):
            if isinstance(value, (dict, list)):
                schema[key] = walk_schema(value)

        schema = _update_schema(schema)

    return schema


class Model(ABC):
    type: str | None = None

    def __init__(
        self,
        model_id: str,
        project: Project,
    ) -> None:
        self.id = model_id
        self.project = project
        self.model = self.define_model()
        loader_registry: WorkerRegistry = ray.get_actor("WorkerLoaderRegistry")
        data_loader: typing.Type[DataLoader] = ray.get(
            loader_registry.get.remote(project.data_loader)
        )
        self.data_loader: DataLoader = data_loader()
        self._trained = False

    @typing.final
    def wrapped_train(
        self,
        samples: list[Sample],
        annotations: list[list[Annotation]],
        params: pydantic.BaseModel,
    ) -> float:
        score = self.train(samples=samples, annotations=annotations, params=params)
        self._trained = True
        return score

    @typing.final
    def wrapped_predict(
        self,
        samples: list[Sample],
        params: pydantic.BaseModel | None,
        data_params: DataParamTypes | None,
    ) -> list[list[AnnotationBase]]:
        if not self._trained:
            raise RuntimeError("Cannot make predictions using an untrained model!")
        return self.predict(samples=samples, params=params, data_params=data_params)

    @typing.final
    def wrapped_save(self, results_dir: pathlib.Path) -> None:
        # Make sure results_dir exists
        results_dir.mkdir(parents=True, exist_ok=True)

        if not self._trained:
            raise RuntimeError("Cannot save a model before it has been trained!")
        self.save(results_dir=results_dir)

    @typing.final
    def wrapped_load(
        self, results_dir: pathlib.Path, weights_filename: str | None = None
    ) -> None:
        self.load(results_dir=results_dir, weights_filename=weights_filename)
        self._trained = True

    @typing.final
    def gpu_available(self) -> bool:
        assigned_resources = ray.get_runtime_context().get_assigned_resources()
        return bool(assigned_resources.get("GPU"))

    def log_progress(
        self,
        training_status: typing.Literal[
            "queued", "started", "failed", "completed", "aborted"
        ]
        | None = None,
        progress: float | None = None,
        score: float | None = None,
    ) -> None:
        """Send progress updates during model training to the TokTagger server.

        Parameters
        ----------
        training_status : typing.Literal[ "queued", "started", "failed", "completed", "aborted" ] | None, optional
            The current stage of model training, by default None
        progress : float | None, optional
            How far through training the model currently is (percentage, 0-100), by default None
        score : float | None, optional
            A metric of how well the model is performing (such as accuracy or loss), by default None
        """
        model_update = ModelUpdate(
            training_status=training_status, progress=progress, score=score
        )
        send_model_updates(self.project.id, self.id, model_update)

    def split_data(
        self,
        samples: list[Sample],
        annotations: list[list[Annotation]],
        train_val_test_split: typing.Tuple[float, float, float],
    ) -> None:
        """Split annotations into training, validation and testing sets.

        Parameters
        ----------
        samples : list[Sample]
            The samples which this model is training on
        annotations : list[list[Annotation]]
            The annotations to split into groups
        train_val_test_split : typing.Tuple[float, float, float]
            The fraction of the total annotations to put in each of the training, validation and test sets respectively.
            These should each be fractions which sum to 1.

        Raises
        ------
        ValueError
            Raised if annotations are missing for some samples provided
        ValueError
            Raised if the splits do not sum to 1
        ValueError
            Raised if the requested splits would leave no samples in the training set
        """
        if len(samples) != len(annotations):
            raise ValueError("Annotations missing for some samples!")
        if not math.isclose(sum(train_val_test_split), 1):
            raise ValueError("Ratios in train_val_test split must sum to 1!")

        train_fraction, val_fraction, test_fraction = train_val_test_split

        if train_fraction == 0:
            raise ValueError("Must be samples in the training set!")

        # If train ratio is 1, no splitting required, just set train sets and return
        if train_fraction == 1:
            self.train_samples = samples
            self.train_annotations = annotations
            self.val_samples = None
            self.val_annotations = None
            self.test_samples = None
            self.test_annotations = None
            return

        # Otherwise need to do some splitting, split into train set and (val + test) set
        train_samples, val_test_samples, train_annotations, val_test_annotations = (
            train_test_split(
                samples, annotations, test_size=val_fraction + test_fraction
            )
        )
        self.train_samples = train_samples
        self.train_annotations = train_annotations

        # If no validation split requested, return test set
        if not val_fraction:
            self.val_samples = None
            self.val_annotations = None
            self.test_samples = val_test_samples
            self.test_annotations = val_test_annotations

        # If no test split requested, return val set
        elif not test_fraction:
            self.val_samples = val_test_samples
            self.val_annotations = val_test_annotations
            self.test_samples = None
            self.test_annotations = None

        # Otherwise split again and return both val and test sets
        else:
            (
                self.val_samples,
                self.test_samples,
                self.val_annotations,
                self.test_annotations,
            ) = train_test_split(
                val_test_samples,
                val_test_annotations,
                test_size=test_fraction / (val_fraction + test_fraction),
            )

    @abstractmethod
    def define_model(self) -> typing.Any:
        """Define and return your model architecture here."""
        pass

    @abstractmethod
    def train(
        self,
        samples: list[Sample],
        annotations: list[list[Annotation]],
        params: pydantic.BaseModel | None = None,
    ) -> float:
        """Train your model here.

        Parameters
        ----------
        samples : list[Sample]
            A list of samples to use when training the model.
        annotations : list[list[Annotation]]
            A list of lists of annotations, one list of annotations per sample.
        params : pydantic.BaseModel | None, optional
            User specified parameters to be used when training the model, by default None

        Returns
        -------
        float
            The score of the trained model, such as accuracy or loss.
        """
        pass

    @abstractmethod
    def predict(
        self,
        samples: list[Sample],
        params: pydantic.BaseModel | None = None,
        data_params: DataParamTypes | None = None,
    ) -> list[list[AnnotationBase]]:
        """Predict over unseen samples with your model here.

        Parameters
        ----------
        samples : list[Sample]
            A list of samples to make predictions over
        params : pydantic.BaseModel | None, optional
            User specified parameters to be used when making predictions, by default None
        data_params : DataParamTypes | None, optional
            Data parameters to use when loading samples for the model to predict over, by default None

        Returns
        -------
        list[list[AnnotationBase]]
            A list of predicted annotations for each sample
        """
        pass

    @abstractmethod
    def save(self, results_dir: pathlib.Path) -> None:
        """Save the model

        Parameters
        ----------
        results_dir : pathlib.Path
            The path to a directory in the model cache where your model weights and accompanying information should be saved.
        """
        pass

    @abstractmethod
    def load(
        self, results_dir: pathlib.Path, weights_filename: str | None = None
    ) -> None:
        """Load a pretrained model.

        Parameters
        ----------
        results_dir : pathlib.Path
            The path to a directory in the model cache where model weights should be loaded from.
        weights_filename : str | None, optional
            The name of the weights file within the above directory to load, by default None
            Note that this is optional, but should take precedence if provided.
        """
        pass


class ModelRegistry:
    _registry: dict[str, typing.Type[Model]] = {}
    _tasks: dict[str, list[Task]] = {}
    _training_params: dict[str, typing.Type[pydantic.BaseModel]] = {}
    _prediction_params: dict[str, typing.Type[pydantic.BaseModel]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        tasks: list[Task | str],
        training_params: typing.Type[pydantic.BaseModel] | None = None,
        prediction_params: typing.Type[pydantic.BaseModel] | None = None,
    ):
        def decorator(model_class: typing.Type[Model]):
            if not issubclass(model_class, Model):
                raise ValueError(
                    f"Loader '{name}' does not inherit from Model base class."
                )
            if training_params and not issubclass(training_params, pydantic.BaseModel):
                raise ValueError(
                    "Must provide training params as a Pydantic BaseModel."
                )
            if prediction_params and not issubclass(
                prediction_params, pydantic.BaseModel
            ):
                raise ValueError(
                    "Must provide prediction params as a Pydantic BaseModel."
                )
            model_class.type = name
            cls._registry[name] = model_class
            cls._tasks[name] = [Task(_task) for _task in tasks]
            cls._training_params[name] = training_params
            cls._prediction_params[name] = prediction_params

            return model_class

        return decorator

    @classmethod
    def get(cls, name: str):
        model_class: typing.Type[Model] | None = cls._registry.get(name)
        if not model_class:
            raise ValueError(f"No Model class called '{name}' found in registry!")
        return ray.remote(model_class)

    @classmethod
    def get_name(cls, model_class: typing.Type[Model]) -> str:
        return next(
            name for name, model in cls._registry.items() if model_class == model
        )

    @classmethod
    def names(cls, task: Task | None = None) -> list[str]:
        if not task:
            return list(cls._registry.keys())
        return [name for name, tasks in cls._tasks.items() if task in tasks]

    @classmethod
    def tasks(cls, name: str) -> list[Task]:
        tasks: list[Task] | None = cls._tasks.get(name)
        if not tasks:
            raise ValueError(f"No tasks associated with model '{name}'!")
        return tasks

    @classmethod
    def get_params(
        cls, name: str, schema_type: typing.Literal["training", "prediction"]
    ) -> typing.Type[pydantic.BaseModel] | None:
        if schema_type == "training":
            params: typing.Type[pydantic.BaseModel] | None = cls._training_params.get(
                name, False
            )
        elif schema_type == "prediction":
            params = cls._prediction_params.get(name, False)
        else:
            raise ValueError(
                "Unexpected type of params - should be training or prediction."
            )

        if params is False:
            raise ValueError(f"No Model class called '{name}' found in registry!")
        return params

    @classmethod
    def get_description(cls, name: str) -> str | None:
        model_class = cls._registry.get(name)
        return inspect.getdoc(model_class) if model_class else None

    @classmethod
    def get_params_schema(
        cls,
        name: str,
        schema_type: typing.Literal["training", "prediction"],
        return_draft_07: bool = False,
    ) -> dict | None:
        """
        Return a schema of the parameters required when training the specified model.

        Parameters
        ----------
        name : str
            The name of the model to return a schema for
        type : Literal["training", "prediction"]
            The type of parameters to get a schema for
        return_draft_07 : bool, optional
            Whether to convert the schema to JSONSchema draft-07, by default False

        Returns
        -------
        schema : dict | None
            The JSONSchema of the params model, if required.
        """

        params: typing.Type[pydantic.BaseModel] | None = cls.get_params(
            name, schema_type
        )
        if not params:
            return None

        schema = params.model_json_schema()

        if not return_draft_07:
            return schema

        return walk_schema(schema)


@ray.remote(num_cpus=0.1)
class WorkerRegistry:
    def __init__(self, registry):
        self._registry = registry

    def get(self, name):
        registered = self._registry.get(name)
        if not registered:
            raise ValueError(f"No class called '{name}' found in registry!")
        return registered


class ActorRegistry:
    """Registry to keep track of Ray actors, and the task they are associated with."""

    def __init__(self, max_actors: int, max_gpu_actors: int):
        """Create task registry

        Parameters
        ----------
        max_actors : int
            Maximum number of actors to keep alive simultaneously
        max_gpu_actors : int
            Maximum number of GPU actors to keep alive simultaneously
        """

        if max_actors < 1:
            raise ValueError(
                "Insufficient CPU cores available for ML model functionality"
            )

        self.gpu_enabled = True if max_gpu_actors > 0 else False
        self.max_actors = max_actors
        self.max_gpu_actors = max_gpu_actors
        self.tasks = {}
        self.actors = OrderedDict()

    def register(self, task_ref: ray.ObjectRef) -> str:
        """Store a Ray task reference in the registry and associate with a UUID.

        Parameters
        ----------
        task_ref : ray.ObjectRef
            The reference to the Ray task

        Returns
        -------
        str
            A unique identifier for this task
        """
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = task_ref
        return task_id

    def get(self, task_id: str) -> ray.ObjectRef | None:
        """Convert a task ID back into the Ray task reference

        Parameters
        ----------
        task_id : str
            The unique identifier for this task

        Returns
        -------
        ray.ObjectRef | None
            The Ray task reference, if it exists in the Registry
        """
        return self.tasks.get(task_id)

    def update_actors(self, actor_name: str, use_gpu: bool) -> None:
        """Record that a Ray Actor has been accessed, and kill any stale Actors.

        Parameters
        ----------
        actor_name : str
            The name of the Ray Actor
        """
        # Set this actor to be the most recently used
        if actor_name in self.actors:
            self.actors.move_to_end(actor_name)
        else:
            self.actors[actor_name] = use_gpu

        if not self.actors[actor_name] and use_gpu:
            # CPU actor may be upgraded to GPU (but not other way round)
            self.actors[actor_name] = use_gpu

        stale_actor = None
        # Check GPU limit first
        gpu_count = sum(1 for gpu in self.actors.values() if gpu)
        if self.gpu_enabled and gpu_count > self.max_gpu_actors:
            # Find first actor which requires GPU
            stale_actor = next(
                (actor for actor, gpu in self.actors.items() if gpu), None
            )
            if not stale_actor:
                raise ValueError("GPU count exceeds maximum, but no GPU actor found!")
            self.actors.pop(stale_actor)

        # Then check overall tasks limit
        elif len(self.actors) > self.max_actors:
            stale_actor, _ = self.actors.popitem(last=False)

        if stale_actor:
            try:
                actor = ray.get_actor(stale_actor)
                # Queue a kill job, letting any other in progress tasks finish first
                actor.__ray_terminate__.remote()
            except ValueError:
                return
