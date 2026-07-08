import os

# Ray (>=2.43) detects when the driver is launched under `uv` and re-runs its
# worker processes via `uv run`. That re-resolves the project environment
# WITHOUT the optional `models` extra, so workers crash with
# `ModuleNotFoundError: No module named 'ray'`. Disable the behaviour so workers
# inherit the driver's interpreter (which has ray/torch). Must be set before
# `import ray`, as Ray reads this flag at import time. setdefault so an operator
# can still opt back in explicitly.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import pathlib
import subprocess
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import tempfile
from toktagger.api.routers.annotations import router as annotations_router
from toktagger.api.routers.annotators import router as annotators_router
from toktagger.api.routers.auth import router as auth_router
from toktagger.api.routers.data import router as data_router
from toktagger.api.routers.models import router as models_router
from toktagger.api.routers.projects import router as projects_router
from toktagger.api.routers.samples import router as samples_router
from toktagger.api.routers.users import router as users_router
from toktagger.api.routers.base import router as base_router
from toktagger.api.routers.paths import router as paths_router
from toktagger.api.routers.meta import router as meta_router
from toktagger.api.core.data_loaders import LoaderRegistry
from toktagger.api.crud.db import MongoDBClient
from toktagger.api.auth.first_run import ensure_admin_user
from toktagger.api.auth.core import get_internal_token
from toktagger.api.models import models_dependencies_installed
import toktagger.api.config as config

# Only import large packages if models dependencies installed
if models_dependencies_installed():
    from toktagger.api.models.base import (
        ModelRegistry,
        WorkerRegistry,
        ActorRegistry,
    )
    import ray


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_name = "annotate_db"

    app.state.db_client = MongoDBClient(
        str(config.settings.database.mongo_url),
        db_name,
        str(config.settings.server.cache_dir),
    )
    app.state.project = None

    # Bootstrap admin user on first run.
    await ensure_admin_user(app.state.db_client)

    yield

    await app.state.db_client.client.close()

    # Tear down the Ray cluster we started so its raylet/worker processes don't
    # outlive the server on a graceful shutdown (Ctrl-C / SIGTERM). Guarded so
    # the `ray` global is only touched when models deps are installed.
    if models_dependencies_installed() and ray.is_initialized():
        ray.shutdown()


# Ray places detached actors in an anonymous, per-driver namespace unless one
# is given explicitly - each `ray.init()` call (i.e. each Gunicorn worker)
# would otherwise get its own anonymous namespace and be unable to see named
# actors (WorkerModelRegistry, TaskRegistry, per-model actors) created by
# other workers, even when all workers share the same underlying cluster.
RAY_NAMESPACE = "toktagger"


def _ray_runtime_env() -> dict:
    return {
        "env_vars": {
            "API_URL": f"http://{config.settings.server.host}:{config.settings.server.port}",
            "MODEL_STORAGE": str(config.settings.models.cache_dir),
            "API_TOKEN": get_internal_token(),
        }
    }


def start_ray_head() -> None:
    """Start a new Ray cluster head node.

    Call this once, in the parent process, before spawning Gunicorn workers -
    each worker then attaches to this cluster (see `run_with_gunicorn`) instead
    of independently starting its own.
    """
    num_gpus = None
    # ALlow the user to force overriding of number of GPUs available
    # This is so that eg Mac can work correctly
    if (
        config.settings.models.force_num_gpus
        and config.settings.models.max_gpu_actors is not None
    ):
        print("Warning: Overriding automatically detected GPU availablity!")
        num_gpus = config.settings.models.max_gpu_actors

    ray.init(
        num_gpus=num_gpus,
        namespace=RAY_NAMESPACE,
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env=_ray_runtime_env(),
    )


def run_with_gunicorn(host: str, port: int, workers: int) -> None:
    """Launch the app under Gunicorn with the given number of worker processes.

    Starts a single shared Ray cluster head node up front (if models are
    installed) and points workers at it via RAY_ADDRESS, since each Gunicorn
    worker attaching independently would otherwise bootstrap its own separate
    cluster.
    """
    if models_dependencies_installed():
        os.environ["TOKTAGGER_INTERNAL_TOKEN"] = get_internal_token()
        start_ray_head()
        os.environ["RAY_ADDRESS"] = "auto"

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "gunicorn",
                "toktagger.api.asgi:app",
                "--worker-class",
                "uvicorn.workers.UvicornWorker",
                "--workers",
                str(workers),
                "--bind",
                f"{host}:{port}",
            ],
            check=True,
        )
    finally:
        if models_dependencies_installed() and ray.is_initialized():
            ray.shutdown()


class Server:
    def __init__(self):
        self.frontend_path = pathlib.Path(__file__).parent / "static"
        self.testing_mode = False

    def _setup_ray(self):
        # If RAY_ADDRESS is set, a Ray cluster head node was already started
        # elsewhere (see start_ray_head/run_with_gunicorn) - attach to it
        # instead of starting a new one. Ray forbids passing num_cpus/num_gpus
        # when connecting to an existing cluster, so that config only applies
        # to start_ray_head().
        if "RAY_ADDRESS" in os.environ:
            ray.init(
                namespace=RAY_NAMESPACE,
                ignore_reinit_error=True,
                include_dashboard=False,
                runtime_env=_ray_runtime_env(),
            )
        else:
            start_ray_head()

        # Detect available resources
        cluster_resources = ray.cluster_resources()
        cpus_available = int(cluster_resources.get("CPU", 0))
        gpus_available = int(cluster_resources.get("GPU", 0))

        if not cpus_available:
            raise RuntimeError("Ray failed to detect any CPUs!")

        max_gpu_actors = (
            config.settings.models.max_gpu_actors
            if config.settings.models.max_gpu_actors is not None
            else gpus_available
        )

        max_actors = (
            config.settings.models.max_actors
            if config.settings.models.max_actors is not None
            else cpus_available
        )

        if max_gpu_actors > gpus_available:
            raise RuntimeError("More GPU actors requested than hardware supports!")

        if max_actors > cpus_available + gpus_available:
            raise RuntimeError(
                "More model actors requested than the detected hardware supports!"
            )

        # Create a ray actor for use as a model registry
        try:
            ray.get_actor("WorkerModelRegistry")
        except ValueError:
            WorkerRegistry.options(
                name="WorkerModelRegistry", lifetime="detached"
            ).remote(ModelRegistry._registry)
        # And one for use as a dataloader registry
        try:
            ray.get_actor("WorkerLoaderRegistry")
        except ValueError:
            WorkerRegistry.options(
                name="WorkerLoaderRegistry", lifetime="detached"
            ).remote(LoaderRegistry._registry)

        # Create a ray actor for use as the shared task/actor registry, so
        # max_actors/max_gpu_actors limits and in-flight task IDs are tracked
        # cluster-wide rather than independently per Gunicorn worker.
        try:
            ray.get_actor("TaskRegistry")
        except ValueError:
            ActorRegistry.options(name="TaskRegistry", lifetime="detached").remote(
                max_actors=max_actors,
                max_gpu_actors=max_gpu_actors,
            )
        self.app.state.task_registry = ray.get_actor("TaskRegistry")

    def _setup_app(self):
        # Check cache dirs are in /tmp if testing mode enabled
        if self.testing_mode:
            tempdir = pathlib.Path(tempfile.gettempdir())
            if (
                tempdir not in config.settings.models.cache_dir.parents
                or tempdir not in config.settings.server.cache_dir.parents
            ):
                raise ValueError(
                    "In testing mode, cache directories must be in temp directory!"
                )

        self.app = FastAPI(lifespan=lifespan)

        # Allow requests from the frontend dev server
        origins = [
            "http://localhost:5173",
        ]

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,  # or ["*"] to allow all
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Static front end files
        self.app.state.index_file = self.frontend_path / "index.html"
        self.app.state.testing_mode = self.testing_mode
        self.app.mount(
            "/assets",
            StaticFiles(directory=self.frontend_path / "assets"),
            name="assets",
        )

        self.app.include_router(auth_router)
        self.app.include_router(users_router)
        self.app.include_router(annotations_router)
        self.app.include_router(data_router)
        self.app.include_router(models_router)
        self.app.include_router(projects_router)
        self.app.include_router(samples_router)
        self.app.include_router(annotators_router)
        self.app.include_router(paths_router)
        self.app.include_router(meta_router)
        self.app.include_router(base_router)

    def run(self):
        """Launch the TokTagger server using the host/port from configuration."""
        self._setup_app()
        # Setup ray if required
        if models_dependencies_installed():
            self._setup_ray()
        uvicorn.run(
            self.app,
            host=config.settings.server.host,
            port=config.settings.server.port,
        )
