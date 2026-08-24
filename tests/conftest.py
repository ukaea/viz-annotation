import asyncio
import importlib
import multiprocessing
import os
import pathlib
import tempfile
import time

import pytest
import pytest_asyncio
import requests
from bson.objectid import ObjectId
from httpx import ASGITransport, AsyncClient

from tests import db_definitions, endpoints
from toktagger.api import config
from toktagger.api.auth.core import create_access_token
from toktagger.api.crud.db import MongoDBClient
from toktagger.api.main import Server
import toktagger.api.auth.core as auth_core

MODELS_ENABLED = importlib.util.find_spec("ray") is not None


@pytest.fixture(autouse=True)
def check_models_status(request):
    if MODELS_ENABLED and request.node.get_closest_marker("models_disabled"):
        pytest.skip("This test requires models dependencies to not be installed!")
    elif not MODELS_ENABLED and request.node.get_closest_marker("models_enabled"):
        pytest.skip("This test requires models dependencies to be installed!")


if MODELS_ENABLED:
    from tests.models_fixtures import (
        models_api_client as models_api_client,
    )
    from tests.models_fixtures import (
        ray_session as ray_session,
    )
    from tests.models_fixtures import (
        setup_model_db as setup_model_db,
    )
    from tests.models_fixtures import (
        setup_model_samples as setup_model_samples,
    )

else:
    _error_msg = (
        "You have attempted to run a test which uses a fixture that requires models, "
        "but the models optional dependencies (Ray) are not installed, "
        "and this test was not marked as a 'models_enabled' test. "
        "Please review the fixture usage of this test, or mark it accurately."
    )

    @pytest.fixture()
    def ray_session():
        raise pytest.UsageError(_error_msg)

    @pytest.fixture()
    def setup_model_samples():
        raise pytest.UsageError(_error_msg)

    @pytest.fixture()
    def setup_model_db():
        raise pytest.UsageError(_error_msg)

    @pytest.fixture()
    def models_api_client():
        raise pytest.UsageError(_error_msg)


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
    except Exception:  # noqa: BLE001 -- any failure means the external server is unreachable
        pytest.skip("Could not contact UDA server")


@pytest.fixture(scope="session")
def settings():
    """Session-scoped config object with temp dirs for models storage.

    Required by ray_session (models_fixtures.py) for MODEL_STORAGE env var.
    Also patches the module-level config.settings so model fixtures that
    reference config.settings.models.cache_dir work correctly.
    """
    with tempfile.TemporaryDirectory(suffix="toktagger_") as tempd:
        models_dir = pathlib.Path(tempd) / "models"
        models_dir.mkdir(exist_ok=True)
        s = config.Settings(
            server=config.Server(cache_dir=tempd),
            models=config.Models(cache_dir=models_dir, max_actors=1),
            database=config.Database(mongo_url="./toktagger_test_db"),
            uda=config.UDA(),
            sal=config.SAL(),
        )
        config.settings = s
        yield s


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
    await db_client.delete_filtered_documents("users")
    await db_client.delete_filtered_documents("project_members")
    await db_client.client.close()


@pytest_asyncio.fixture(scope="function")
async def unauthenticated_api_client(tmp_path, monkeypatch, db_client):
    monkeypatch.setattr(config.settings.server, "cache_dir", tmp_path)
    monkeypatch.setattr(auth_core, "_serializer", None)

    server = Server()
    server.testing_mode = True
    monkeypatch.setenv("API_URL", "http://test")
    server._setup_app()
    app = server.app
    app.state.db_client = db_client
    app.state.project = None

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        client.app = app
        yield client


@pytest_asyncio.fixture(scope="function")
async def api_client(unauthenticated_api_client, db_client):
    # Auth is always required now — seed the admin user and authenticate as them
    await db_client.insert("users", db_definitions.USER_ADMIN)
    admin_token = create_access_token({"sub": "admin"})
    unauthenticated_api_client.headers["Authorization"] = f"Bearer {admin_token}"
    yield unauthenticated_api_client


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


@pytest_asyncio.fixture(scope="function")
async def setup_db_auth(db_client):
    admin_id = await db_client.insert("users", db_definitions.USER_ADMIN)
    alice_id = await db_client.insert("users", db_definitions.USER_ALICE)
    bob_id = await db_client.insert("users", db_definitions.USER_BOB)

    yield {
        "admin_id": admin_id,
        "alice_id": alice_id,
        "bob_id": bob_id,
    }
    await db_client.delete_filtered_documents("users")


def run_server():
    server = Server()
    server.testing_mode = True
    server.run()


@pytest.fixture(scope="package")
def start_server(settings):
    # Explicit "fork" context (not just multiprocessing.Process, which defaults
    # to "spawn" on macOS since Python 3.8): "spawn" re-imports this module in
    # a fresh interpreter, so run_server() never sees the settings fixture's
    # mutated config.settings (temp cache dirs) and _setup_app()'s safety check
    # ("cache directories must be in temp directory") kills the child before it
    # can bind the port. "fork" inherits the parent's memory, including that
    # mutation, same as Linux's default start method (so this is a no-op on CI).
    proc = multiprocessing.get_context("fork").Process(target=run_server)
    proc.start()
    # Wait for server to start
    server_up = False
    for t in range(600):
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


@pytest.fixture(scope="package")
def admin_token(start_server) -> str:
    """Log in as the bootstrap admin (created by ensure_admin_user on first
    server start, see toktagger/api/auth/first_run.py) and authenticate all
    tests.endpoints.* requests as them for the rest of this server's lifetime.
    """
    response = requests.post(
        "http://localhost:8002/auth/token",
        data={"username": "admin", "password": "admin"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # The bootstrap admin ships with must_change_password set, so a real first login
    # is held on the profile page until the default password is replaced. Clear it
    # here so this session behaves like any other logged-in admin — the same opt-out
    # tests.endpoints.create_user applies to the accounts it creates. The forced
    # change itself is covered in tests/end_to_end/test_profile_page.py and
    # tests/api/auth/test_first_run.py.
    response = requests.get("http://localhost:8002/auth/me", headers=headers)
    assert response.status_code == 200, response.text
    admin_id = response.json()["_id"]
    response = requests.put(
        f"http://localhost:8002/users/{admin_id}",
        json={"must_change_password": False},
        headers=headers,
    )
    assert response.status_code == 200, response.text

    endpoints.set_auth_token(token)
    return token


@pytest.fixture(scope="function")
def server_setup(start_server, admin_token):
    yield
    response = requests.get(
        "http://localhost:8002/health",
    )
    if not response.json().get("testing_mode"):
        raise RuntimeError("End to End test has connected to a live server!")
    else:
        response = requests.delete(
            "http://localhost:8002/projects",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 200
