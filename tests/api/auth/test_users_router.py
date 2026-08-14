"""Integration tests for /users and /projects/{id}/members endpoints."""

import pytest

from tests.api.auth.conftest import add_member, get_auth_token


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
        json={"password": "alice_new_pass123"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200

    # Verify the new password works for login
    login_resp = await client.post(
        "/auth/token",
        data={"username": "alice", "password": "alice_new_pass123"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert login_resp.status_code == 200


@pytest.mark.asyncio
async def test_user_cannot_self_promote_global_role(auth_setup):
    """A non-admin editing their own record must not be able to set
    global_role — self-edit bypasses the "editing someone else" check, so
    this has to be enforced separately or any user could self-promote."""
    client = auth_setup["client"]
    token = await get_auth_token(client, "alice", "alice_pass")
    alice_id = auth_setup["alice_id"]
    response = await client.put(
        f"/users/{alice_id}",
        json={"global_role": "admin"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403

    get_resp = await client.get(
        f"/users/{alice_id}", headers={"Authorization": f"Bearer {token}"}
    )
    assert get_resp.json()["global_role"] == "user"


@pytest.mark.asyncio
async def test_user_cannot_self_reactivate_via_is_active(auth_setup):
    """Same guard, is_active side: a non-admin must not be able to flip their
    own is_active flag either."""
    client = auth_setup["client"]
    token = await get_auth_token(client, "alice", "alice_pass")
    alice_id = auth_setup["alice_id"]
    response = await client.put(
        f"/users/{alice_id}",
        json={"is_active": False},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_update_other_user_as_non_admin_forbidden(auth_setup):
    client = auth_setup["client"]
    token = await get_auth_token(client, "alice", "alice_pass")
    bob_id = auth_setup["bob_id"]
    response = await client.put(
        f"/users/{bob_id}",
        json={"global_role": "admin"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_update_other_user_as_admin(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    bob_id = auth_setup["bob_id"]
    response = await client.put(
        f"/users/{bob_id}",
        json={"global_role": "admin"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200

    # Verify the update via GET /users/{bob_id}
    get_resp = await client.get(
        f"/users/{bob_id}", headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert get_resp.status_code == 200
    assert get_resp.json()["global_role"] == "admin"


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
async def test_delete_own_user_as_non_admin_forbidden(auth_setup):
    """Delete requires global admin unconditionally — there's no self-service
    exception the way update_user has (current_user.id == user_id)."""
    client = auth_setup["client"]
    token = await get_auth_token(client, "alice", "alice_pass")
    alice_id = auth_setup["alice_id"]
    response = await client.delete(
        f"/users/{alice_id}",
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
async def test_admin_cannot_delete_own_user_as_last_admin(auth_setup):
    """Deleting the sole active admin must be blocked — mirrors the
    demote/deactivate guard in update_user. Without it, the account list
    becomes unmanageable (no admin left to fix it)."""
    client = auth_setup["client"]
    token = await get_auth_token(client, "admin", "admin_pass")
    admin_id = auth_setup["admin_id"]
    response = await client.delete(
        f"/users/{admin_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422

    # Admin still exists and can still log in
    login_resp = await client.post(
        "/auth/token",
        data={"username": "admin", "password": "admin_pass"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert login_resp.status_code == 200


@pytest.mark.asyncio
async def test_admin_can_delete_own_user_when_another_admin_remains(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    admin_id = auth_setup["admin_id"]

    # Promote alice to admin so deleting the original admin is no longer
    # deleting the *last* one.
    promote_resp = await client.put(
        f"/users/{auth_setup['alice_id']}",
        json={"global_role": "admin"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert promote_resp.status_code == 200

    response = await client.delete(
        f"/users/{admin_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200

    login_resp = await client.post(
        "/auth/token",
        data={"username": "admin", "password": "admin_pass"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert login_resp.status_code == 401


@pytest.mark.asyncio
async def test_add_and_list_project_members(project_setup):
    client = project_setup["client"]
    admin_token = project_setup["admin_token"]
    project_id = project_setup["project_id"]

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
async def test_add_member_non_admin_forbidden(project_setup):
    client = project_setup["client"]
    project_id = project_setup["project_id"]
    alice_token = await get_auth_token(client, "alice", "alice_pass")

    resp = await client.post(
        f"/projects/{project_id}/members",
        json={"username": "bob", "role": "annotator"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_update_member_show_others_annotations(project_setup):
    client = project_setup["client"]
    admin_token = project_setup["admin_token"]
    project_id = project_setup["project_id"]
    alice_token = await get_auth_token(client, "alice", "alice_pass")

    # Add alice as annotator (uses username, not user_id)
    await client.post(
        f"/projects/{project_id}/members",
        json={"username": "alice", "role": "annotator"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    # Alice updates her own show_others_annotations preference
    resp = await client.put(
        f"/projects/{project_id}/members/{project_setup['alice_id']}",
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
async def test_remove_project_member(project_setup):
    client = project_setup["client"]
    admin_token = project_setup["admin_token"]
    project_id = project_setup["project_id"]

    await client.post(
        f"/projects/{project_id}/members",
        json={"username": "alice", "role": "annotator"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    del_resp = await client.delete(
        f"/projects/{project_id}/members/{project_setup['alice_id']}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert del_resp.status_code == 200

    list_resp = await client.get(
        f"/projects/{project_id}/members",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    usernames = [m["username"] for m in list_resp.json()]
    assert "alice" not in usernames


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["viewer", "annotator"])
async def test_member_cannot_self_promote_project_role(project_setup, role):
    """A member must not be able to promote themselves by editing their own membership.

    The self-edit path exists so a member can set show_others_annotations, but it
    must not extend to `role` — otherwise a viewer PUTs {"role": "admin"} on their
    own membership and becomes a project admin.
    """
    client = project_setup["client"]
    admin_token = project_setup["admin_token"]
    project_id = project_setup["project_id"]
    alice_id = project_setup["alice_id"]

    await add_member(client, admin_token, project_id, "alice", role)
    alice_token = await get_auth_token(client, "alice", "alice_pass")

    resp = await client.put(
        f"/projects/{project_id}/members/{alice_id}",
        json={"role": "admin"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 403

    # The stored role must be untouched.
    members_resp = await client.get(
        f"/projects/{project_id}/members",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    alice_member = next(m for m in members_resp.json() if m["username"] == "alice")
    assert alice_member["role"] == role


@pytest.mark.asyncio
async def test_project_admin_can_manage_members_without_global_admin(project_setup):
    """A project admin whose global_role is only "user" can still manage members."""
    client = project_setup["client"]
    admin_token = project_setup["admin_token"]
    project_id = project_setup["project_id"]

    await add_member(client, admin_token, project_id, "alice", "admin")
    alice_token = await get_auth_token(client, "alice", "alice_pass")

    # alice is a project admin but a plain global user
    me_resp = await client.get(
        "/auth/me", headers={"Authorization": f"Bearer {alice_token}"}
    )
    assert me_resp.json()["global_role"] == "user"

    add_resp = await client.post(
        f"/projects/{project_id}/members",
        json={"username": "bob", "role": "viewer"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert add_resp.status_code == 200, add_resp.text

    # ...including changing another member's role, which the self-edit guard must
    # not have broken.
    role_resp = await client.put(
        f"/projects/{project_id}/members/{project_setup['bob_id']}",
        json={"role": "annotator"},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert role_resp.status_code == 200, role_resp.text

    del_resp = await client.delete(
        f"/projects/{project_id}/members/{project_setup['bob_id']}",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert del_resp.status_code == 200, del_resp.text


@pytest.mark.asyncio
async def test_list_my_memberships_is_self_scoped(project_setup):
    """/users/me/memberships reports only the caller's own memberships."""
    client = project_setup["client"]
    admin_token = project_setup["admin_token"]
    project_id = project_setup["project_id"]

    await add_member(client, admin_token, project_id, "alice", "viewer")
    alice_token = await get_auth_token(client, "alice", "alice_pass")
    bob_token = await get_auth_token(client, "bob", "bob_pass")

    alice_resp = await client.get(
        "/users/me/memberships", headers={"Authorization": f"Bearer {alice_token}"}
    )
    assert alice_resp.status_code == 200
    alice_memberships = alice_resp.json()
    assert len(alice_memberships) == 1
    assert alice_memberships[0]["project_id"] == project_id
    assert alice_memberships[0]["role"] == "viewer"
    assert alice_memberships[0]["user_id"] == project_setup["alice_id"]

    # bob is in no projects
    bob_resp = await client.get(
        "/users/me/memberships", headers={"Authorization": f"Bearer {bob_token}"}
    )
    assert bob_resp.status_code == 200
    assert bob_resp.json() == []

    # "me" must not be read as a user_id by GET /users/{user_id}
    assert (await client.get("/users/me/memberships")).status_code == 401
