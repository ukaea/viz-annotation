"""Integration tests for /auth/token and /auth/me endpoints."""

import pytest

from tests.api.auth.conftest import get_auth_token


@pytest.mark.asyncio
async def test_login_success(unauthenticated_api_client, setup_db_auth):
    client = unauthenticated_api_client
    response = await client.post(
        "/auth/token",
        data={"username": "admin", "password": "admin_pass"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert len(body["access_token"]) > 0


@pytest.mark.asyncio
async def test_login_wrong_password(unauthenticated_api_client, setup_db_auth):
    client = unauthenticated_api_client
    response = await client.post(
        "/auth/token",
        data={"username": "admin", "password": "wrong_password"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_login_unknown_user(unauthenticated_api_client, setup_db_auth):
    client = unauthenticated_api_client
    response = await client.post(
        "/auth/token",
        data={"username": "ghost", "password": "doesnt_matter"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_login_inactive_user(unauthenticated_api_client, setup_db_auth):
    """Deactivated users cannot log in."""
    client = unauthenticated_api_client
    admin_token = await get_auth_token(client, "admin", "admin_pass")

    # Deactivate alice via the admin API
    await client.put(
        f"/users/{setup_db_auth['alice_id']}",
        json={"is_active": False},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    response = await client.post(
        "/auth/token",
        data={"username": "alice", "password": "alice_pass"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_get_me_returns_current_user(unauthenticated_api_client, setup_db_auth):
    client = unauthenticated_api_client
    token = await get_auth_token(client, "alice", "alice_pass")
    response = await client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "alice"
    assert body["global_role"] == "user"
    assert body["is_active"] is True
    assert "hashed_password" not in body


@pytest.mark.asyncio
async def test_get_me_admin_role(unauthenticated_api_client, setup_db_auth):
    client = unauthenticated_api_client
    token = await get_auth_token(client, "admin", "admin_pass")
    response = await client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["global_role"] == "admin"


@pytest.mark.asyncio
async def test_get_me_no_token(unauthenticated_api_client, setup_db_auth):
    client = unauthenticated_api_client
    response = await client.get("/auth/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_me_invalid_token(unauthenticated_api_client, setup_db_auth):
    client = unauthenticated_api_client
    response = await client.get(
        "/auth/me",
        headers={"Authorization": "Bearer not.a.real.token"},
    )
    assert response.status_code == 401
