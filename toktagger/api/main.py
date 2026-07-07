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
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import warnings
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

    # Bootstrap admin user on first run; set auth_required flag.
    # TOKTAGGER_AUTH_REQUIRED=false disables auth (tests only).
    if os.environ.get("TOKTAGGER_AUTH_REQUIRED", "true").lower() == "false":
        app.state.auth_required = False
    else:
        app.state.auth_required = await ensure_admin_user(app.state.db_client)

    yield

    await app.state.db_client.client.close()

    # Tear down the Ray cluster we started so its raylet/worker processes don't
    # outlive the server on a graceful shutdown (Ctrl-C / SIGTERM). Guarded so
    # the `ray` global is only touched when models deps are installed.
    if models_dependencies_installed() and ray.is_initialized():
        ray.shutdown()


class Server:
    def __init__(self):
        self.frontend_path = pathlib.Path(__file__).parent / "static"
        self.testing_mode = False

    def _setup_ray(self):
        from toktagger.api.auth.core import get_internal_token

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
            ignore_reinit_error=True,
            include_dashboard=False,
            runtime_env={
                "env_vars": {
                    "API_URL": f"http://{config.settings.server.host}:{config.settings.server.port}",
                    "MODEL_STORAGE": str(config.settings.models.cache_dir),
                    "API_TOKEN": get_internal_token(),
                }
            },
        )
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

        self.app.state.task_registry = ActorRegistry(
            max_actors=max_actors,
            max_gpu_actors=max_gpu_actors,
        )

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

    def run(self, host: str | None = None, port: int | None = None):
        """
        Launch the TokTagger server.

        Parameters
        ----------
        host : str
            DEPRECATED - use config file or environment variables instead.
            The host to launch the server on, by default 'localhost'
        port : int
            DEPRECATED - use config file or environment variables instead.
            The port to launch the server on, by default 8002
        """
        # Provide deprecation warning
        if host or port:
            warnings.warn(
                """
                Specifying host and port within Server.run() is deprecated and will be removed in a future version.
                Please provide these arguments via configuration file or environment variable instead.
                See https://ukaea.github.io/toktagger/configuration for details.
                """,
                DeprecationWarning,
                stacklevel=2,
            )
        if host:
            config.settings.server.host = host
        if port:
            config.settings.server.port = port

        self._setup_app()
        # Setup ray if required
        if models_dependencies_installed():
            self._setup_ray()
        uvicorn.run(
            self.app,
            host=config.settings.server.host,
            port=config.settings.server.port,
        )
