"""Conftest for auth tests — uses mongita (no Docker required)."""

import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from toktagger.api.main import Server
from toktagger.api.crud.db import MongoDBClient
from toktagger.api.auth.core import hash_password
from toktagger.api.schemas.users import UserIn


# ---------------------------------------------------------------------------
# Low-level DB client (per-test, fresh path each time)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="function")
async def db_client(tmp_path):
    client = MongoDBClient(str(tmp_path), "annotate_db")
    yield client
    await client.client.close()


# ---------------------------------------------------------------------------
# Passthrough API client (auth_required=False) — for first_run tests
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="function")
async def api_client(tmp_path):
    db = MongoDBClient(str(tmp_path), "annotate_db")

    server = Server()
    server._setup_app()
    app = server.app
    app.state.db_client = db
    app.state.auth_required = False
    app.state.project = None

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        client.app = app
        yield client

    await db.client.close()


# ---------------------------------------------------------------------------
# Auth-aware fixture: auth_required=True, three known users seeded
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="function")
async def auth_setup(tmp_path):
    """
    Returns a dict with:
      - client:    AsyncClient ready to make requests
      - get_token: async helper(username, password) -> str
      - admin_id, alice_id, bob_id: inserted user IDs
    """
    db = MongoDBClient(str(tmp_path), "annotate_db")

    server = Server()
    server._setup_app()
    app = server.app
    app.state.db_client = db
    app.state.auth_required = True
    app.state.project = None

    admin_id = await db.insert(
        "users",
        UserIn(
            username="admin",
            hashed_password=hash_password("admin_pass"),
            email="admin@test.com",
            global_role="admin",
            is_active=True,
        ),
    )
    alice_id = await db.insert(
        "users",
        UserIn(
            username="alice",
            hashed_password=hash_password("alice_pass"),
            email="alice@test.com",
            global_role="user",
            is_active=True,
        ),
    )
    bob_id = await db.insert(
        "users",
        UserIn(
            username="bob",
            hashed_password=hash_password("bob_pass"),
            email="bob@test.com",
            global_role="user",
            is_active=True,
        ),
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        client.app = app

        async def get_token(username: str, password: str) -> str:
            resp = await client.post(
                "/auth/token",
                data={"username": username, "password": password},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            assert resp.status_code == 200, f"Login failed ({username}): {resp.text}"
            return resp.json()["access_token"]

        yield {
            "client": client,
            "get_token": get_token,
            "admin_id": admin_id,
            "alice_id": alice_id,
            "bob_id": bob_id,
        }

    await db.client.close()
