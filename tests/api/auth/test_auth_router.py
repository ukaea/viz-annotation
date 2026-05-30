"""Integration tests for /auth/token and /auth/me endpoints."""
import pytest


@pytest.mark.asyncio
async def test_login_success(auth_setup):
    client = auth_setup["client"]
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
async def test_login_wrong_password(auth_setup):
    client = auth_setup["client"]
    response = await client.post(
        "/auth/token",
        data={"username": "admin", "password": "wrong_password"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_login_unknown_user(auth_setup):
    client = auth_setup["client"]
    response = await client.post(
        "/auth/token",
        data={"username": "ghost", "password": "doesnt_matter"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_login_inactive_user(auth_setup):
    """Deactivated users cannot log in."""
    client = auth_setup["client"]
    admin_token = await auth_setup["get_token"]("admin", "admin_pass")

    # Deactivate alice via the admin API
    await client.put(
        f"/users/{auth_setup['alice_id']}",
        json={"is_active": False},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    response = await client.post(
        "/auth/token",
        data={"username": "alice", "password": "alice_pass"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code in (401, 403)


@pytest.mark.asyncio
async def test_get_me_returns_current_user(auth_setup):
    client = auth_setup["client"]
    token = await auth_setup["get_token"]("alice", "alice_pass")
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
async def test_get_me_admin_role(auth_setup):
    client = auth_setup["client"]
    token = await auth_setup["get_token"]("admin", "admin_pass")
    response = await client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["global_role"] == "admin"


@pytest.mark.asyncio
async def test_get_me_no_token(auth_setup):
    client = auth_setup["client"]
    response = await client.get("/auth/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_me_invalid_token(auth_setup):
    client = auth_setup["client"]
    response = await client.get(
        "/auth/me",
        headers={"Authorization": "Bearer not.a.real.token"},
    )
    assert response.status_code == 401
