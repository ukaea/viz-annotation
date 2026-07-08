"""Conftest for auth tests — uses mongita (no Docker required)."""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

import toktagger.api.config as config
import toktagger.api.auth.core as auth_core
from toktagger.api.main import Server
from toktagger.api.crud.db import MongoDBClient
import tests.db_definitions as db_definitions


@pytest.fixture(autouse=True)
def _isolate_auth_cache(tmp_path, monkeypatch):
    """Keep the auth secret.key and first_run.lock out of the real user cache dir.

    Points server.cache_dir at the test's tmp_path (the same dir the DB fixtures
    use) so these files are written under the configured location and cleaned up
    with the test. Resetting the cached serializer forces it to be rebuilt against
    the patched location.
    """
    monkeypatch.setattr(config.settings.server, "cache_dir", tmp_path)
    monkeypatch.setattr(auth_core, "_serializer", None)


async def get_auth_token(client: AsyncClient, username: str, password: str) -> str:
    """Obtain a JWT access token for the given user."""
    resp = await client.post(
        "/auth/token",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 200, f"Login failed ({username}): {resp.text}"
    return resp.json()["access_token"]


@pytest_asyncio.fixture(scope="function")
async def auth_db_client(tmp_path):
    """Low-level DB client backed by mongita (per-test, no Docker)."""
    client = MongoDBClient("default", "annotate_db", cache_dir=str(tmp_path))
    yield client
    await client.client.close()


@pytest_asyncio.fixture(scope="function")
async def api_client():
    """Bare ASGI test client — db state is wired in by consuming fixtures."""
    server = Server()
    server._setup_app()
    app = server.app
    app.state.project = None

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        client.app = app
        yield client


@pytest_asyncio.fixture(scope="function")
async def auth_setup(tmp_path):
    """Auth-aware fixture with three pre-seeded users.

    Yields a dict with:
      - client:    AsyncClient for making requests
      - admin_id, alice_id, bob_id: inserted user IDs

    Use get_auth_token(client, username, password) to obtain JWT tokens.
    """
    db = MongoDBClient("default", "annotate_db", cache_dir=str(tmp_path))

    server = Server()
    server._setup_app()
    app = server.app
    app.state.db_client = db
    app.state.project = None

    admin_id = await db.insert("users", db_definitions.USER_ADMIN)
    alice_id = await db.insert("users", db_definitions.USER_ALICE)
    bob_id = await db.insert("users", db_definitions.USER_BOB)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        client.app = app
        yield {
            "client": client,
            "admin_id": admin_id,
            "alice_id": alice_id,
            "bob_id": bob_id,
        }

    await db.client.close()


async def create_project(
    client: AsyncClient, token: str, name: str = "test_project"
) -> str:
    """Create a project as the given token; return the project_id."""
    resp = await client.post(
        "/projects",
        json={
            "name": name,
            "task": "time-series",
            "query_strategy": "sequential",
            "data_loader": "tabular",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["_id"]


async def create_project_and_sample(
    client: AsyncClient, token: str, name: str = "test_project"
) -> tuple[str, str]:
    """Create a project then add one sample; return (project_id, sample_id)."""
    project_id = await create_project(client, token, name=name)
    sample_resp = await client.post(
        f"/projects/{project_id}/samples",
        json=[{"shot_id": 1, "data": {"file_name": "t.csv", "type": "csv"}}],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert sample_resp.status_code == 200, sample_resp.text
    sample_id = sample_resp.json()[0]
    return project_id, sample_id


@pytest_asyncio.fixture(scope="function")
async def project_setup(auth_setup):
    """auth_setup, plus an admin-created project with one sample."""
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)
    return {
        **auth_setup,
        "admin_token": admin_token,
        "project_id": project_id,
        "sample_id": sample_id,
    }
