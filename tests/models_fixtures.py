from toktagger.api.schemas.annotations import TimePointBatch
from toktagger.api.schemas.samples import SampleIn, TimeSeriesFileData
import tests.db_definitions as db_definitions
from toktagger.api.main import Server
from toktagger.api.auth.core import get_internal_token
from httpx import AsyncClient, ASGITransport
from bson.objectid import ObjectId
import ray
import random
import pytest
import pytest_asyncio
import toktagger.api.config as config


@pytest.fixture(scope="module")
def ray_session(settings):
    ray.init(
        num_gpus=1,  # Due to env vars set in models_api_client
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
    app.state.auth_required = False
    # This task ID is associated with a model in the db, so that cancelling training test works
    app.state.task_registry.tasks["abc123"] = "Ray Task Object"

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        client.app = app
        yield client

    # Kill any outstanding actors, since the task registry is recreated on a by test basis
    # But they ray cluster is only spun up on a per module basis to save time
    for actor_id in app.state.task_registry.actors.keys():
        try:
            actor = ray.get_actor(actor_id)
            # Queue a kill job, letting any other in progress tasks finish first
            ray.kill(actor)
        except ValueError:
            continue


@pytest_asyncio.fixture(scope="function")
async def authenticated_models_api_client(models_api_client, db_client):
    """models_api_client with real per-user JWT auth enabled (auth_required=True).

    Seeds the built-in admin user directly (bypassing first-run bootstrap, which
    only runs from the app's lifespan). db_client's underlying mongita store is
    shared for the whole test session, so the seeded users are deleted again on
    teardown rather than left to leak into later tests.
    """
    models_api_client.app.state.auth_required = True
    await db_client.insert("users", db_definitions.USER_ADMIN)

    yield models_api_client

    # Clean up all users/memberships this test created — db_client's mongita store
    # is shared for the whole session, and no other test expects data here.
    await db_client.delete_filtered_documents(collection="users")
    await db_client.delete_filtered_documents(collection="project_members")


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
        config.settings.models.cache_dir.joinpath(f"{_id}.model").write_text(
            "Test Model"
        )
    yield {
        "project_id": project_id,
        "sample_ids": sample_ids,
        "model_id_1": model_id_1,
        "model_id_2": model_id_2,
        "model_id_3": model_id_3,
        "model_id_4": model_id_4,
    }
