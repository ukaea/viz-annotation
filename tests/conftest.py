import pytest
import pytest_asyncio
from toktagger.api.main import Server
from toktagger.api.crud.db import MongoDBClient
import tests.db_definitions as db_definitions
from bson.objectid import ObjectId
import asyncio
from httpx import AsyncClient, ASGITransport
import os
import multiprocessing
import requests
import time
import tempfile
import toktagger.api.config as config
import importlib
import pathlib

MODELS_ENABLED = importlib.util.find_spec("ray") is not None


@pytest.fixture(autouse=True)
def check_models_status(request):
    if MODELS_ENABLED and request.node.get_closest_marker("models_disabled"):
        pytest.skip("This test requires models dependencies to not be installed!")
    elif not MODELS_ENABLED and request.node.get_closest_marker("models_enabled"):
        pytest.skip("This test requires models dependencies to be installed!")


if MODELS_ENABLED:
    from tests.models_fixtures import (
        ray_session as ray_session,
        setup_model_samples as setup_model_samples,
        setup_model_db as setup_model_db,
        models_api_client as models_api_client,
    )

else:
    error_msg = """ You have attempted to run a test which uses a fixture that requires models,
            but the models optional dependencies (Ray) are not installed, 
            and this test was not marked as a 'models_enabled' test.
            Please review the fixture usage of this test, or mark it accurately.
            """

    @pytest.fixture()
    def ray_session():
        raise pytest.UsageError(error_msg)

    @pytest.fixture()
    def setup_model_samples():
        raise pytest.UsageError(error_msg)

    @pytest.fixture()
    def setup_model_db():
        raise pytest.UsageError(error_msg)

    @pytest.fixture()
    def models_api_client():
        raise pytest.UsageError(error_msg)


@pytest.fixture(scope="session")
def uda_env_vars():
    os.environ.setdefault("UDA_HOST", "uda2.mast.l")
    os.environ.setdefault("UDA_META_PLUGINNAME", "MASTU_DB")
    os.environ.setdefault("UDA_METANEW_PLUGINNAME", "MAST_DB")


@pytest.fixture(scope="function")
def uda_test(uda_env_vars):
    try:
        import pyuda

        pyuda.Client().get("help::help()")
    except Exception:
        pytest.skip("Could not contact UDA server")


@pytest.fixture(scope="session")
def settings():
    with tempfile.TemporaryDirectory(suffix="toktagger_") as tempd:
        pathlib.Path(tempd).joinpath("models").mkdir(exist_ok=True)
        settings = config.Settings(
            server=config.Server(cache_dir=tempd),
            models=config.Models(
                cache_dir=pathlib.Path(tempd).joinpath("models"), max_actors=1
            ),
            database=config.Database(mongo_url="./toktagger_test_db"),
            uda=config.UDA(),
            sal=config.SAL(),
        )
        config.settings = settings
        yield settings


@pytest_asyncio.fixture(scope="function")
async def db_client(settings):
    db_client = MongoDBClient(
        settings.database.mongo_url, "annotate_db", settings.server.cache_dir
    )
    yield db_client

    await db_client.delete_filtered_documents("projects")
    await db_client.delete_filtered_documents("samples")
    await db_client.delete_filtered_documents("annotations")
    await db_client.delete_filtered_documents("models")
    await db_client.client.close()


@pytest_asyncio.fixture(scope="function")
async def api_client(monkeypatch, db_client):
    # Have hit various issues getting this setup
    # Using fastAPI TestClient() doesn't play well with async pymongo as it tries to do stuff in different event loops
    # So have to use this AsyncClient from httpx, but this no longer just accepts an app
    # So have to wrap it in this Transport thing, but that for some reason doesnt run the lifespan in the app
    # So have to run this manually, however trying to run the close after the yield to close the db connection gives errors
    # So am just going to leave it open, since the db container will be deleted after anyway
    # Any alternative solution ideas are welcome.....
    os.environ["API_URL"] = "http://test"
    server = Server()
    server.testing_mode = True
    monkeypatch.setenv("API_URL", "http://test")
    server._setup_app()
    app = server.app
    app.state.db_client = db_client
    app.state.project = None

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        client.app = app
        yield client


@pytest_asyncio.fixture(scope="function")
async def setup_db(db_client):
    project_id_1 = await db_client.insert("projects", db_definitions.PROJECT_1)
    await asyncio.sleep(0.01)
    project_id_2 = await db_client.insert("projects", db_definitions.PROJECT_2)
    await asyncio.sleep(0.01)
    project_id_3 = await db_client.insert("projects", db_definitions.PROJECT_3)
    await asyncio.sleep(0.01)
    sample_id_1 = await db_client.insert(
        "samples", db_definitions.SAMPLE_1, ids={"project_id": ObjectId(project_id_1)}
    )
    await asyncio.sleep(0.01)
    sample_id_2 = await db_client.insert(
        "samples", db_definitions.SAMPLE_2, ids={"project_id": ObjectId(project_id_1)}
    )
    await asyncio.sleep(0.01)
    sample_id_3 = await db_client.insert(
        "samples", db_definitions.SAMPLE_3, ids={"project_id": ObjectId(project_id_2)}
    )
    await asyncio.sleep(0.01)
    sample_id_4 = await db_client.insert(
        "samples", db_definitions.SAMPLE_4, ids={"project_id": ObjectId(project_id_2)}
    )
    await asyncio.sleep(0.01)
    annotation_id_1 = await db_client.insert(
        "annotations",
        db_definitions.ANNOTATION_1,
        ids={"project_id": ObjectId(project_id_1), "sample_id": ObjectId(sample_id_1)},
    )
    await asyncio.sleep(0.01)
    annotation_id_2 = await db_client.insert(
        "annotations",
        db_definitions.ANNOTATION_2,
        ids={"project_id": ObjectId(project_id_1), "sample_id": ObjectId(sample_id_1)},
    )
    await asyncio.sleep(0.01)
    annotation_id_3 = await db_client.insert(
        "annotations",
        db_definitions.ANNOTATION_3,
        ids={"project_id": ObjectId(project_id_1), "sample_id": ObjectId(sample_id_1)},
    )
    await asyncio.sleep(0.01)
    annotation_id_4 = await db_client.insert(
        "annotations",
        db_definitions.ANNOTATION_4,
        ids={"project_id": ObjectId(project_id_1), "sample_id": ObjectId(sample_id_2)},
    )
    await asyncio.sleep(0.01)
    annotation_id_5 = await db_client.insert(
        "annotations",
        db_definitions.ANNOTATION_5,
        ids={"project_id": ObjectId(project_id_2), "sample_id": ObjectId(sample_id_4)},
    )
    await asyncio.sleep(0.01)
    yield {
        "project_id_1": project_id_1,
        "project_id_2": project_id_2,
        "project_id_3": project_id_3,
        "sample_id_1": sample_id_1,
        "sample_id_2": sample_id_2,
        "sample_id_3": sample_id_3,
        "sample_id_4": sample_id_4,
        "annotation_id_1": annotation_id_1,
        "annotation_id_2": annotation_id_2,
        "annotation_id_3": annotation_id_3,
        "annotation_id_4": annotation_id_4,
        "annotation_id_5": annotation_id_5,
    }


@pytest_asyncio.fixture(scope="function")
async def setup_db_small(db_client):
    ids = {}
    ids["projects"] = await db_client.insert("projects", db_definitions.PROJECT_1)
    ids["samples"] = await db_client.insert(
        "samples",
        db_definitions.SAMPLE_1,
        ids={"project_id": ObjectId(ids["projects"])},
    )
    ids["annotations"] = await db_client.insert(
        "annotations",
        db_definitions.ANNOTATION_1,
        ids={
            "project_id": ObjectId(ids["projects"]),
            "sample_id": ObjectId(ids["samples"]),
        },
    )

    yield ids

    await db_client.delete_filtered_documents("projects")
    await db_client.delete_filtered_documents("samples")
    await db_client.delete_filtered_documents("annotations")


def run_server():
    # Import to register mock model

    server = Server()
    server.testing_mode = True
    server.run()


@pytest.fixture(scope="package")
def start_server(settings):
    proc = multiprocessing.Process(target=run_server)
    proc.start()
    # Wait for server to start
    server_up = False
    for t in range(60):
        try:
            response = requests.get(
                "http://localhost:8002/health",
            )
            if response.status_code == 200:
                status = response.json()
                if not status["testing_mode"]:
                    raise RuntimeError(
                        "End to End test has connected to a live server!"
                    )
                if not status["db_connected"]:
                    raise RuntimeError("Database failed to connect.")
                if not status["name"] == "TokTagger":
                    raise RuntimeError(
                        "End to End test has connected to another process running on localhost:8002"
                    )
                server_up = True
                break
            time.sleep(1)
        except requests.exceptions.ConnectionError:
            time.sleep(1)

    if not server_up:
        proc.terminate()
        pytest.exit("Server failed to start for End-to-End tests to run!")

    yield
    proc.terminate()
    proc.join()


@pytest.fixture(scope="function")
def server_setup(start_server):
    yield
    response = requests.get(
        "http://localhost:8002/health",
    )
    if not response.json().get("testing_mode"):
        raise RuntimeError("End to End test has connected to a live server!")
    else:
        response = requests.delete(
            "http://localhost:8002/projects",
        )
        assert response.status_code == 200
