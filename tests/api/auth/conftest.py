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
    """Bare ASGI test client — db/auth state is wired in by setup_auth_db."""
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
async def setup_auth_db(api_client, auth_db_client):
    """Wires an empty mongita-backed db into api_client (auth_required=False) — for first_run tests."""
    api_client.app.state.db_client = auth_db_client
    api_client.app.state.auth_required = False

    yield auth_db_client


@pytest_asyncio.fixture(scope="function")
async def auth_setup(tmp_path):
    """Auth-aware fixture: auth_required=True with three pre-seeded users.

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
    app.state.auth_required = True
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
