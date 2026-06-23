"""
Integration tests verifying that membership guards are enforced across all resource
endpoints (samples, project-level annotations, sample-level annotation delete, data).

Permission matrix:
  - Non-member          → 403 on everything
  - Viewer              → 200 on reads, 403 on writes/deletes
  - Annotator           → 200 on reads and writes, 403 on destructive deletes
  - Project admin (admin role) → 200 on everything
"""

import pytest

from tests.api.auth.conftest import get_auth_token


async def create_project_and_sample(client, token):
    proj_resp = await client.post(
        "/projects",
        json={
            "name": "guard_test",
            "task": "time-series",
            "query_strategy": "sequential",
            "data_loader": "tabular",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert proj_resp.status_code == 200, proj_resp.text
    project_id = proj_resp.json()["_id"]

    sample_resp = await client.post(
        f"/projects/{project_id}/samples",
        json=[{"shot_id": 1, "data": {"file_name": "t.csv", "type": "csv"}}],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert sample_resp.status_code == 200, sample_resp.text
    sample_id = sample_resp.json()[0]
    return project_id, sample_id


async def add_member(client, token, project_id, username, role):
    resp = await client.post(
        f"/projects/{project_id}/members",
        json={"username": username, "role": role},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text


def annotation_payload():
    return [
        {
            "label": "lbl",
            "time_min": 0.0,
            "time_max": 1.0,
            "type": "time_region",
            "validated": False,
            "created_by": "placeholder",
        }
    ]


@pytest.mark.asyncio
async def test_non_member_cannot_list_samples(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, _ = await create_project_and_sample(client, admin_token)

    bob_token = await get_auth_token(client, "bob", "bob_pass")
    resp = await client.get(
        f"/projects/{project_id}/samples",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_viewer_can_list_samples(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, _ = await create_project_and_sample(client, admin_token)
    await add_member(client, admin_token, project_id, "alice", "viewer")

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    resp = await client.get(
        f"/projects/{project_id}/samples",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_non_member_cannot_add_samples(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, _ = await create_project_and_sample(client, admin_token)

    bob_token = await get_auth_token(client, "bob", "bob_pass")
    resp = await client.post(
        f"/projects/{project_id}/samples",
        json=[{"shot_id": 99, "data": {"file_name": "x.csv", "type": "csv"}}],
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_viewer_cannot_add_samples(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, _ = await create_project_and_sample(client, admin_token)
    await add_member(client, admin_token, project_id, "alice", "viewer")

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    resp = await client.post(
        f"/projects/{project_id}/samples",
        json=[{"shot_id": 99, "data": {"file_name": "x.csv", "type": "csv"}}],
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_annotator_can_add_samples(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, _ = await create_project_and_sample(client, admin_token)
    await add_member(client, admin_token, project_id, "alice", "annotator")

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    resp = await client.post(
        f"/projects/{project_id}/samples",
        json=[{"shot_id": 99, "data": {"file_name": "x.csv", "type": "csv"}}],
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_annotator_cannot_delete_sample(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)
    await add_member(client, admin_token, project_id, "alice", "annotator")

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    resp = await client.delete(
        f"/projects/{project_id}/samples/{sample_id}",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_project_admin_can_delete_sample(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)

    # Admin (global) is also an implicit project admin; use them directly.
    resp = await client.delete(
        f"/projects/{project_id}/samples/{sample_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_annotator_cannot_delete_all_samples(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, _ = await create_project_and_sample(client, admin_token)
    await add_member(client, admin_token, project_id, "alice", "annotator")

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    resp = await client.delete(
        f"/projects/{project_id}/samples",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_global_admin_can_delete_all_samples(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, _ = await create_project_and_sample(client, admin_token)

    resp = await client.delete(
        f"/projects/{project_id}/samples",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_non_member_cannot_get_project_annotations(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, _ = await create_project_and_sample(client, admin_token)

    bob_token = await get_auth_token(client, "bob", "bob_pass")
    resp = await client.get(
        f"/projects/{project_id}/annotations",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_viewer_can_get_project_annotations(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, _ = await create_project_and_sample(client, admin_token)
    await add_member(client, admin_token, project_id, "alice", "viewer")

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    resp = await client.get(
        f"/projects/{project_id}/annotations",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_viewer_cannot_import_annotations(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, _ = await create_project_and_sample(client, admin_token)
    await add_member(client, admin_token, project_id, "alice", "viewer")

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    resp = await client.put(
        f"/projects/{project_id}/annotations",
        json=annotation_payload(),
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_annotator_can_import_annotations(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)
    await add_member(client, admin_token, project_id, "alice", "annotator")

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    # Bulk import requires full annotation docs (with sample_id embedded)
    payload = [
        {
            "label": "lbl",
            "time_min": 0.0,
            "time_max": 1.0,
            "type": "time_region",
            "validated": False,
            "created_by": "alice",
            "shot_id": 1,
            "sample_id": sample_id,
            "project_id": project_id,
        }
    ]
    resp = await client.put(
        f"/projects/{project_id}/annotations",
        json=payload,
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_annotator_cannot_delete_project_annotations(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, _ = await create_project_and_sample(client, admin_token)
    await add_member(client, admin_token, project_id, "alice", "annotator")

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    resp = await client.delete(
        f"/projects/{project_id}/annotations",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_global_admin_can_delete_project_annotations(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, _ = await create_project_and_sample(client, admin_token)

    resp = await client.delete(
        f"/projects/{project_id}/annotations",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_annotator_cannot_delete_all_sample_annotations(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)
    await add_member(client, admin_token, project_id, "alice", "annotator")

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    resp = await client.delete(
        f"/projects/{project_id}/samples/{sample_id}/annotations",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_global_admin_can_delete_all_sample_annotations(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)

    resp = await client.delete(
        f"/projects/{project_id}/samples/{sample_id}/annotations",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_non_member_cannot_get_data(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)

    bob_token = await get_auth_token(client, "bob", "bob_pass")
    resp = await client.post(
        f"/projects/{project_id}/samples/{sample_id}/data",
        json={},
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_viewer_passes_data_auth_check(auth_setup):
    """Viewer should pass the auth gate (may still get 404/422 for missing data file)."""
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)
    await add_member(client, admin_token, project_id, "alice", "viewer")

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    resp = await client.post(
        f"/projects/{project_id}/samples/{sample_id}/data",
        json={},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code != 403
