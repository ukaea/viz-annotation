import random
import time

import pytest
import pytest_asyncio
import ray
from bson.objectid import ObjectId
from httpx import ASGITransport, AsyncClient

from tests import db_definitions
from toktagger.api import config
from toktagger.api.auth.core import create_access_token, get_internal_token
from toktagger.api.main import RAY_NAMESPACE, Server
from toktagger.api.schemas.annotations import TimePointBatch
from toktagger.api.schemas.samples import SampleIn, TimeSeriesFileData


@ray.remote(num_cpus=0.1)
def _pending_task():
    time.sleep(3600)


@pytest.fixture(scope="module")
def ray_session(settings):
    ray.init(
        num_gpus=1,  # Due to env vars set in models_api_client
        namespace=RAY_NAMESPACE,
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={
            "env_vars": {
                "MODEL_STORAGE": str(config.settings.models.cache_dir),
                "API_URL": "",
            }
        },
    )

    yield
    ray.shutdown()


@pytest_asyncio.fixture(scope="function")
async def models_api_client(monkeypatch, settings, db_client, ray_session):
    server = Server()
    server.testing_mode = True
    monkeypatch.setenv("API_URL", "http://test")
    monkeypatch.setenv("MAX_GPU_ACTORS", 1)
    monkeypatch.setenv("FORCE_NUM_GPUS", True)
    # So send_batch_annotations/send_batch_samples (replayed from the test process
    # in collect_predict_results) authenticate as the internal worker user, matching
    # what main.py._setup_ray does for real Ray workers in production.
    monkeypatch.setenv("API_TOKEN", get_internal_token())

    server._setup_app()
    server._setup_ray()
    app = server.app
    app.state.db_client = db_client
    app.state.project = None
    # This task ID is associated with a model in the db, so that cancelling training test works.
    # Registered as a real (long-running) task ref, since ray.cancel() runs for
    # real inside the TaskRegistry actor - it can't be mocked from this process.
    ray.get(
        app.state.task_registry.register.remote(
            [_pending_task.remote()], task_id="abc123"
        )
    )

    # Auth is always required now — seed the admin user and authenticate as them
    # by default, so existing tests keep behaving as an implicit admin.
    await db_client.insert("users", db_definitions.USER_ADMIN)
    admin_token = create_access_token({"sub": "admin"})

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {admin_token}"},
    ) as client:
        client.app = app
        yield client

    # Cancel the "abc123" pending task, in case this test didn't (e.g. it wasn't
    # the cancel-training test), so it doesn't linger for the rest of the module.
    ray.get(app.state.task_registry.cancel.remote("abc123"))

    # Kill any outstanding actors, since the task registry is recreated on a by test basis
    # But they ray cluster is only spun up on a per module basis to save time
    for actor_id in ray.get(app.state.task_registry.list_actors.remote()):
        try:
            actor = ray.get_actor(actor_id)
            # Queue a kill job, letting any other in progress tasks finish first
            ray.kill(actor)
        except ValueError:
            continue

    # TaskRegistry is a detached, named actor shared across the whole Ray
    # cluster (production behaviour: shared by all Gunicorn workers). Kill it
    # too, so the next test's _setup_ray() creates a fresh one instead of
    # inheriting this test's tasks/actors/limits from the shared ray_session.
    ray.kill(app.state.task_registry)


@pytest.fixture(scope="package")
def setup_model_samples():
    # Create sample data for training / predicting a Disruption model
    samples = []
    for i in range(9980, 10000):
        # Generate sample data
        disruption_time = random.randint(80, 100)
        annotation = TimePointBatch(
            shot_id=i,
            validated=True,
            label="Disruption",
            time=disruption_time,
            created_by="manual" if i < 9985 else "model::mock_disruption_cnn",
        )

        samples.append(
            SampleIn(
                shot_id=i,
                data=TimeSeriesFileData(
                    file_name=f"{i}.parquet",
                    type="parquet",
                ),
                annotations=[annotation] if i < 9990 else None,
            )
        )

    yield samples


@pytest_asyncio.fixture(scope="function")
async def setup_model_db(setup_model_samples, db_client):
    project_id = await db_client.insert("projects", db_definitions.PROJECT_2)
    sample_ids = []
    for sample in setup_model_samples:
        sample_id = await db_client.insert(
            "samples", sample, ids={"project_id": ObjectId(project_id)}
        )
        sample_ids.append(sample_id)

        if sample.annotations:
            await db_client.insert(
                "annotations",
                sample.annotations[0],
                ids={
                    "project_id": ObjectId(project_id),
                    "sample_id": ObjectId(sample_id),
                },
            )

    model_id_1 = await db_client.insert(
        "models", db_definitions.MODEL_1, ids={"project_id": ObjectId(project_id)}
    )

    model_id_2 = await db_client.insert(
        "models", db_definitions.MODEL_2, ids={"project_id": ObjectId(project_id)}
    )

    model_id_3 = await db_client.insert(
        "models",
        db_definitions.MODEL_3,
        ids={"project_id": ObjectId(project_id)},
    )

    model_id_4 = await db_client.insert(
        "models", db_definitions.MODEL_4, ids={"project_id": ObjectId(project_id)}
    )

    # Create temp files for each
    for _id in (model_id_1, model_id_2, model_id_3, model_id_4):
        results_dir = config.settings.models.cache_dir.joinpath(_id)
        results_dir.mkdir(parents=True)
        results_dir.joinpath("weights.model").write_text("Test Model")
    yield {
        "project_id": project_id,
        "sample_ids": sample_ids,
        "model_id_1": model_id_1,
        "model_id_2": model_id_2,
        "model_id_3": model_id_3,
        "model_id_4": model_id_4,
    }
