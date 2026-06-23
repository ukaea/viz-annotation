"""
Integration tests for concurrent annotation safety and per-user visibility.

Key invariants under test:
  1. User A saving annotations does NOT delete User B's annotations.
  2. The server overwrites `created_by` from the JWT — clients cannot spoof identity.
  3. When show_others_annotations=False, a user only sees their own annotations.
  4. Viewer-role users cannot PUT annotations (403).
  5. A project non-member cannot access annotations (403).
"""

import pytest

from tests.api.auth.conftest import get_auth_token


async def create_project_and_sample(client, token):
    """Create a project then add one sample; return (project_id, sample_id)."""
    proj_resp = await client.post(
        "/projects",
        json={
            "name": "concurrency_test",
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
        json=[
            {
                "shot_id": 42,
                "data": {"file_name": "test.csv", "type": "csv"},
            }
        ],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert sample_resp.status_code == 200, sample_resp.text
    sample_id = sample_resp.json()[0]
    return project_id, sample_id


def annotation_payload(label: str):
    return [
        {
            "label": label,
            "time_min": 0.1,
            "time_max": 0.5,
            "type": "time_region",
            "validated": True,
            "created_by": "placeholder",  # server overwrites from JWT
        }
    ]


async def put_annotations(client, project_id, sample_id, token, label):
    resp = await client.put(
        f"/projects/{project_id}/samples/{sample_id}/annotations",
        json=annotation_payload(label),
        headers={"Authorization": f"Bearer {token}"},
    )
    return resp


async def get_annotations(client, project_id, sample_id, token):
    resp = await client.get(
        f"/projects/{project_id}/samples/{sample_id}/annotations",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.asyncio
async def test_user_save_does_not_overwrite_other_users_annotations(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)

    for username in ("alice", "bob"):
        await client.post(
            f"/projects/{project_id}/members",
            json={"username": username, "role": "annotator"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    bob_token = await get_auth_token(client, "bob", "bob_pass")

    resp_a = await put_annotations(
        client, project_id, sample_id, alice_token, "alice_label"
    )
    assert resp_a.status_code == 200

    resp_b = await put_annotations(
        client, project_id, sample_id, bob_token, "bob_label"
    )
    assert resp_b.status_code == 200

    annotations = await get_annotations(client, project_id, sample_id, admin_token)
    labels = {a["label"] for a in annotations}
    assert "alice_label" in labels
    assert "bob_label" in labels


@pytest.mark.asyncio
async def test_user_save_replaces_only_own_previous_annotations(auth_setup):
    """Saving twice as the same user replaces only that user's annotations."""
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)

    for username in ("alice", "bob"):
        await client.post(
            f"/projects/{project_id}/members",
            json={"username": username, "role": "annotator"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    bob_token = await get_auth_token(client, "bob", "bob_pass")

    await put_annotations(client, project_id, sample_id, alice_token, "alice_v1")
    await put_annotations(client, project_id, sample_id, bob_token, "bob_v1")

    await put_annotations(client, project_id, sample_id, alice_token, "alice_v2")

    annotations = await get_annotations(client, project_id, sample_id, admin_token)
    labels = {a["label"] for a in annotations}
    assert "alice_v2" in labels
    assert "alice_v1" not in labels
    assert "bob_v1" in labels


@pytest.mark.asyncio
async def test_server_overwrites_created_by_from_jwt(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)

    await client.post(
        f"/projects/{project_id}/members",
        json={"username": "alice", "role": "annotator"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    alice_token = await get_auth_token(client, "alice", "alice_pass")

    spoofed = [
        {
            "label": "spoofed",
            "time_min": 0.0,
            "time_max": 1.0,
            "type": "time_region",
            "validated": True,
            "created_by": "admin",  # attempt to impersonate admin
        }
    ]
    resp = await client.put(
        f"/projects/{project_id}/samples/{sample_id}/annotations",
        json=spoofed,
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 200

    annotations = await get_annotations(client, project_id, sample_id, admin_token)
    assert len(annotations) == 1
    assert annotations[0]["created_by"] == "alice"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "show_others,expect_bobs_label", [(False, False), (True, True)]
)
async def test_show_others_annotations_filter(
    auth_setup, show_others, expect_bobs_label
):
    """When show_others_annotations is toggled, alice's view changes accordingly."""
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)

    for username in ("alice", "bob"):
        await client.post(
            f"/projects/{project_id}/members",
            json={"username": username, "role": "annotator"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    bob_token = await get_auth_token(client, "bob", "bob_pass")

    await put_annotations(client, project_id, sample_id, alice_token, "alice_ann")
    await put_annotations(client, project_id, sample_id, bob_token, "bob_ann")

    await client.put(
        f"/projects/{project_id}/members/{auth_setup['alice_id']}",
        json={"show_others_annotations": show_others},
        headers={"Authorization": f"Bearer {alice_token}"},
    )

    alice_view = await get_annotations(client, project_id, sample_id, alice_token)
    labels = {a["label"] for a in alice_view}
    assert "alice_ann" in labels
    assert ("bob_ann" in labels) == expect_bobs_label


@pytest.mark.asyncio
async def test_viewer_cannot_put_annotations(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)

    await client.post(
        f"/projects/{project_id}/members",
        json={"username": "alice", "role": "viewer"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    resp = await put_annotations(
        client, project_id, sample_id, alice_token, "viewer_attempt"
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_non_member_cannot_get_annotations(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)

    bob_token = await get_auth_token(client, "bob", "bob_pass")
    resp = await client.get(
        f"/projects/{project_id}/samples/{sample_id}/annotations",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_non_member_cannot_see_project_in_list(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    await create_project_and_sample(client, admin_token)

    bob_token = await get_auth_token(client, "bob", "bob_pass")
    resp = await client.get(
        "/projects",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_member_can_see_project_in_list(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, _ = await create_project_and_sample(client, admin_token)

    await client.post(
        f"/projects/{project_id}/members",
        json={"username": "alice", "role": "annotator"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    resp = await client.get(
        "/projects",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 200
    project_ids = [p["_id"] for p in resp.json()]
    assert project_id in project_ids
