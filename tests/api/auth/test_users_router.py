"""Integration tests for /users and /projects/{id}/members endpoints."""

import pytest

from tests.api.auth.conftest import get_auth_token


async def create_project(client, token):
    resp = await client.post(
        "/projects",
        json={
            "name": "test_project",
            "task": "time-series",
            "query_strategy": "sequential",
            "data_loader": "tabular",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["_id"]


@pytest.mark.asyncio
async def test_list_users_as_admin(auth_setup):
    client = auth_setup["client"]
    token = await get_auth_token(client, "admin", "admin_pass")
    response = await client.get("/users", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    users = response.json()
    usernames = [u["username"] for u in users]
    assert "admin" in usernames
    assert "alice" in usernames
    assert "bob" in usernames


@pytest.mark.asyncio
async def test_list_users_non_admin_forbidden(auth_setup):
    client = auth_setup["client"]
    token = await get_auth_token(client, "alice", "alice_pass")
    response = await client.get("/users", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_create_user_as_admin(auth_setup):
    client = auth_setup["client"]
    token = await get_auth_token(client, "admin", "admin_pass")
    response = await client.post(
        "/users",
        json={
            "username": "newuser",
            "password": "newpass123",
            "email": "new@test.com",
            "global_role": "user",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    # Endpoint returns {"_id": "<new_user_id>"}
    assert "_id" in body
    assert len(body["_id"]) > 0


@pytest.mark.asyncio
async def test_create_user_non_admin_forbidden(auth_setup):
    client = auth_setup["client"]
    token = await get_auth_token(client, "alice", "alice_pass")
    response = await client.post(
        "/users",
        json={
            "username": "sneaky",
            "password": "pass",
            "email": "",
            "global_role": "admin",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_get_user_by_id_self(auth_setup):
    client = auth_setup["client"]
    token = await get_auth_token(client, "alice", "alice_pass")
    alice_id = auth_setup["alice_id"]
    response = await client.get(
        f"/users/{alice_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["username"] == "alice"


@pytest.mark.asyncio
async def test_get_other_user_as_non_admin_forbidden(auth_setup):
    client = auth_setup["client"]
    token = await get_auth_token(client, "alice", "alice_pass")
    bob_id = auth_setup["bob_id"]
    response = await client.get(
        f"/users/{bob_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_update_own_user(auth_setup):
    client = auth_setup["client"]
    token = await get_auth_token(client, "alice", "alice_pass")
    alice_id = auth_setup["alice_id"]
    response = await client.put(
        f"/users/{alice_id}",
        json={"email": "alice_new@test.com"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200

    # Verify the update via GET /users/{alice_id}
    get_resp = await client.get(
        f"/users/{alice_id}", headers={"Authorization": f"Bearer {token}"}
    )
    assert get_resp.status_code == 200
    assert get_resp.json()["email"] == "alice_new@test.com"


@pytest.mark.asyncio
async def test_update_other_user_as_non_admin_forbidden(auth_setup):
    client = auth_setup["client"]
    token = await get_auth_token(client, "alice", "alice_pass")
    bob_id = auth_setup["bob_id"]
    response = await client.put(
        f"/users/{bob_id}",
        json={"email": "hacked@evil.com"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_delete_user_as_non_admin_forbidden(auth_setup):
    client = auth_setup["client"]
    token = await get_auth_token(client, "alice", "alice_pass")
    bob_id = auth_setup["bob_id"]
    response = await client.delete(
        f"/users/{bob_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_delete_user_as_admin(auth_setup):
    client = auth_setup["client"]
    token = await get_auth_token(client, "admin", "admin_pass")
    bob_id = auth_setup["bob_id"]
    response = await client.delete(
        f"/users/{bob_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200

    # Bob can no longer log in
    login_resp = await client.post(
        "/auth/token",
        data={"username": "bob", "password": "bob_pass"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert login_resp.status_code == 401


@pytest.mark.asyncio
async def test_add_and_list_project_members(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id = await create_project(client, admin_token)

    # Add alice as annotator (uses username, not user_id)
    resp = await client.post(
        f"/projects/{project_id}/members",
        json={"username": "alice", "role": "annotator"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200

    # List members
    list_resp = await client.get(
        f"/projects/{project_id}/members",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_resp.status_code == 200
    members = list_resp.json()
    usernames = [m["username"] for m in members]
    assert "alice" in usernames


@pytest.mark.asyncio
async def test_add_member_non_admin_forbidden(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    alice_token = await get_auth_token(client, "alice", "alice_pass")
    project_id = await create_project(client, admin_token)

    resp = await client.post(
        f"/projects/{project_id}/members",
        json={"username": "bob", "role": "annotator"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_update_member_show_others_annotations(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    alice_token = await get_auth_token(client, "alice", "alice_pass")
    project_id = await create_project(client, admin_token)

    # Add alice as annotator (uses username, not user_id)
    await client.post(
        f"/projects/{project_id}/members",
        json={"username": "alice", "role": "annotator"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    # Alice updates her own show_others_annotations preference
    resp = await client.put(
        f"/projects/{project_id}/members/{auth_setup['alice_id']}",
        json={"show_others_annotations": False},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 200

    # Verify the DB value changed
    members_resp = await client.get(
        f"/projects/{project_id}/members",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    alice_member = next(m for m in members_resp.json() if m["username"] == "alice")
    assert alice_member["show_others_annotations"] is False


@pytest.mark.asyncio
async def test_remove_project_member(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id = await create_project(client, admin_token)

    await client.post(
        f"/projects/{project_id}/members",
        json={"username": "alice", "role": "annotator"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    del_resp = await client.delete(
        f"/projects/{project_id}/members/{auth_setup['alice_id']}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert del_resp.status_code == 200

    list_resp = await client.get(
        f"/projects/{project_id}/members",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    usernames = [m["username"] for m in list_resp.json()]
    assert "alice" not in usernames
