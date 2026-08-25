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

from tests.api.auth.conftest import (
    add_member,
    get_auth_token,
)


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
async def test_user_save_does_not_overwrite_other_users_annotations(
    setup_db_auth, unauthenticated_api_client
):
    client = unauthenticated_api_client
    admin_token = await get_auth_token(
        unauthenticated_api_client, "admin", "admin_pass"
    )
    project_id = setup_db_auth["project_id"]
    sample_id = setup_db_auth["sample_id"]

    for username in ("alice", "bob"):
        await add_member(client, admin_token, project_id, username, "annotator")

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
async def test_user_save_replaces_only_own_previous_annotations(
    setup_db_auth, unauthenticated_api_client
):
    """Saving twice as the same user replaces only that user's annotations."""
    client = unauthenticated_api_client
    admin_token = await get_auth_token(
        unauthenticated_api_client, "admin", "admin_pass"
    )
    project_id = setup_db_auth["project_id"]
    sample_id = setup_db_auth["sample_id"]

    for username in ("alice", "bob"):
        await add_member(client, admin_token, project_id, username, "annotator")

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
async def test_server_overwrites_created_by_from_jwt(
    setup_db_auth, unauthenticated_api_client
):
    client = unauthenticated_api_client
    admin_token = await get_auth_token(
        unauthenticated_api_client, "admin", "admin_pass"
    )
    project_id = setup_db_auth["project_id"]
    sample_id = setup_db_auth["sample_id"]

    await add_member(client, admin_token, project_id, "alice", "annotator")

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
    setup_db_auth, unauthenticated_api_client, show_others, expect_bobs_label
):
    """When show_others_annotations is toggled, alice's view changes accordingly."""
    client = unauthenticated_api_client
    admin_token = await get_auth_token(
        unauthenticated_api_client, "admin", "admin_pass"
    )
    project_id = setup_db_auth["project_id"]
    sample_id = setup_db_auth["sample_id"]

    for username in ("alice", "bob"):
        await add_member(client, admin_token, project_id, username, "annotator")

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    bob_token = await get_auth_token(client, "bob", "bob_pass")

    await put_annotations(client, project_id, sample_id, alice_token, "alice_ann")
    await put_annotations(client, project_id, sample_id, bob_token, "bob_ann")

    await client.put(
        f"/projects/{project_id}/members/{setup_db_auth['alice_id']}",
        json={"show_others_annotations": show_others},
        headers={"Authorization": f"Bearer {alice_token}"},
    )

    alice_view = await get_annotations(client, project_id, sample_id, alice_token)
    labels = {a["label"] for a in alice_view}
    assert "alice_ann" in labels
    assert ("bob_ann" in labels) == expect_bobs_label


@pytest.mark.asyncio
async def test_viewer_cannot_put_annotations(setup_db_auth, unauthenticated_api_client):
    client = unauthenticated_api_client
    admin_token = await get_auth_token(
        unauthenticated_api_client, "admin", "admin_pass"
    )
    project_id = setup_db_auth["project_id"]
    sample_id = setup_db_auth["sample_id"]

    await add_member(client, admin_token, project_id, "alice", "viewer")

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    resp = await put_annotations(
        client, project_id, sample_id, alice_token, "viewer_attempt"
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_non_member_cannot_get_annotations(
    setup_db_auth, unauthenticated_api_client
):
    client = unauthenticated_api_client
    project_id = setup_db_auth["project_id"]
    sample_id = setup_db_auth["sample_id"]

    bob_token = await get_auth_token(client, "bob", "bob_pass")
    resp = await client.get(
        f"/projects/{project_id}/samples/{sample_id}/annotations",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_non_member_cannot_see_project_in_list(
    setup_db_auth, unauthenticated_api_client
):
    client = unauthenticated_api_client

    bob_token = await get_auth_token(client, "bob", "bob_pass")
    resp = await client.get(
        "/projects",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_member_can_see_project_in_list(
    setup_db_auth, unauthenticated_api_client
):
    client = unauthenticated_api_client
    admin_token = await get_auth_token(
        unauthenticated_api_client, "admin", "admin_pass"
    )
    project_id = setup_db_auth["project_id"]

    await add_member(client, admin_token, project_id, "alice", "annotator")

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    resp = await client.get(
        "/projects",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 200
    project_ids = [p["_id"] for p in resp.json()]
    assert project_id in project_ids


@pytest.mark.asyncio
async def test_save_preserves_model_created_by(
    setup_db_auth, unauthenticated_api_client
):
    """A freshly-predicted annotation's synthetic created_by survives a save."""
    client = unauthenticated_api_client
    admin_token = await get_auth_token(
        unauthenticated_api_client, "admin", "admin_pass"
    )
    project_id = setup_db_auth["project_id"]
    sample_id = setup_db_auth["sample_id"]

    await add_member(client, admin_token, project_id, "alice", "annotator")
    alice_token = await get_auth_token(client, "alice", "alice_pass")

    prediction = [
        {
            "label": "predicted",
            "time_min": 0.2,
            "time_max": 0.6,
            "type": "time_region",
            "validated": False,
            "created_by": "model::changepoint_detection",
        }
    ]
    resp = await client.put(
        f"/projects/{project_id}/samples/{sample_id}/annotations",
        json=prediction,
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 200

    annotations = await get_annotations(client, project_id, sample_id, admin_token)
    assert len(annotations) == 1
    assert annotations[0]["created_by"] == "model::changepoint_detection"


@pytest.mark.asyncio
async def test_save_does_not_duplicate_or_reattribute_others_annotation(
    setup_db_auth, unauthenticated_api_client
):
    """Resending an already-saved annotation owned by someone else (as loaded via
    GET) must not duplicate it or reassign it to the saving user."""
    client = unauthenticated_api_client
    admin_token = await get_auth_token(
        unauthenticated_api_client, "admin", "admin_pass"
    )
    project_id = setup_db_auth["project_id"]
    sample_id = setup_db_auth["sample_id"]

    for username in ("alice", "bob"):
        await add_member(client, admin_token, project_id, username, "annotator")

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    bob_token = await get_auth_token(client, "bob", "bob_pass")

    await put_annotations(client, project_id, sample_id, bob_token, "bob_ann")

    # Alice loads the sample's annotations (visible since show_others_annotations
    # defaults to True) and resends the full set she has loaded, as the frontend
    # does on every Save.
    loaded = await get_annotations(client, project_id, sample_id, alice_token)
    assert len(loaded) == 1
    resp = await client.put(
        f"/projects/{project_id}/samples/{sample_id}/annotations",
        json=loaded,
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 200

    annotations = await get_annotations(client, project_id, sample_id, admin_token)
    assert len(annotations) == 1
    assert annotations[0]["created_by"] == "bob"


@pytest.mark.asyncio
async def test_annotator_can_edit_another_users_annotation(
    setup_db_auth, unauthenticated_api_client
):
    """An annotator's edit to a colleague's annotation is applied, not silently dropped.

    The save path replaces the caller's own annotations by delete-then-reinsert, so
    someone else's annotation is edited in place instead — which must change the
    geometry while leaving the original author, and the model's prediction, alone.
    """
    client = unauthenticated_api_client
    admin_token = await get_auth_token(
        unauthenticated_api_client, "admin", "admin_pass"
    )
    project_id = setup_db_auth["project_id"]
    sample_id = setup_db_auth["sample_id"]

    for username in ("alice", "bob"):
        await add_member(client, admin_token, project_id, username, "annotator")

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    bob_token = await get_auth_token(client, "bob", "bob_pass")

    # Bob saves his own annotation plus a model prediction in one go — a second save
    # would replace his own, since his save is scoped to created_by="bob".
    await client.put(
        f"/projects/{project_id}/samples/{sample_id}/annotations",
        json=[
            {
                "label": "bob_ann",
                "time_min": 0.1,
                "time_max": 0.5,
                "type": "time_region",
                "validated": True,
                "created_by": "placeholder",  # server overwrites from JWT
            },
            {
                "label": "predicted",
                "time_min": 0.8,
                "time_max": 0.9,
                "type": "time_region",
                "validated": False,
                "created_by": "model::changepoint_detection",
            },
        ],
        headers={"Authorization": f"Bearer {bob_token}"},
    )

    loaded = await get_annotations(client, project_id, sample_id, alice_token)
    assert len(loaded) == 2

    # Alice drags bob's region and relabels it, then saves the whole loaded set.
    for annotation in loaded:
        if annotation["created_by"] == "bob":
            annotation["time_min"] = 2.5
            annotation["time_max"] = 3.5
            annotation["label"] = "edited_by_alice"

    resp = await client.put(
        f"/projects/{project_id}/samples/{sample_id}/annotations",
        json=loaded,
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 200, resp.text

    annotations = await get_annotations(client, project_id, sample_id, admin_token)
    assert len(annotations) == 2, "the edit must not duplicate or drop annotations"

    edited = next(a for a in annotations if a["created_by"] == "bob")
    assert edited["label"] == "edited_by_alice"
    assert edited["time_min"] == 2.5
    assert edited["time_max"] == 3.5

    prediction = next(
        a for a in annotations if a["created_by"] == "model::changepoint_detection"
    )
    assert prediction["time_min"] == 0.8
    assert prediction["label"] == "predicted"


@pytest.mark.asyncio
async def test_editing_others_annotation_does_not_delete_their_other_annotations(
    setup_db_auth, unauthenticated_api_client
):
    """Editing one of bob's annotations must leave his others in place."""
    client = unauthenticated_api_client
    admin_token = await get_auth_token(
        unauthenticated_api_client, "admin", "admin_pass"
    )
    project_id = setup_db_auth["project_id"]
    sample_id = setup_db_auth["sample_id"]

    for username in ("alice", "bob"):
        await add_member(client, admin_token, project_id, username, "annotator")

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    bob_token = await get_auth_token(client, "bob", "bob_pass")

    await client.put(
        f"/projects/{project_id}/samples/{sample_id}/annotations",
        json=[
            {
                "label": f"bob_{i}",
                "time_min": float(i),
                "time_max": float(i) + 0.5,
                "type": "time_region",
                "validated": True,
                "created_by": "placeholder",
            }
            for i in range(3)
        ],
        headers={"Authorization": f"Bearer {bob_token}"},
    )

    loaded = await get_annotations(client, project_id, sample_id, alice_token)
    assert len(loaded) == 3

    # Alice edits exactly one of them and sends only that one back.
    target = next(a for a in loaded if a["label"] == "bob_1")
    target["label"] = "bob_1_edited"
    resp = await client.put(
        f"/projects/{project_id}/samples/{sample_id}/annotations",
        json=[target],
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 200, resp.text

    annotations = await get_annotations(client, project_id, sample_id, admin_token)
    labels = {a["label"] for a in annotations}
    assert labels == {"bob_0", "bob_1_edited", "bob_2"}
    assert all(a["created_by"] == "bob" for a in annotations)


@pytest.mark.asyncio
async def test_annotator_can_delete_another_users_annotation(
    setup_db_auth, unauthenticated_api_client
):
    """The single-annotation delete works regardless of who created the annotation."""
    client = unauthenticated_api_client
    admin_token = await get_auth_token(
        unauthenticated_api_client, "admin", "admin_pass"
    )
    project_id = setup_db_auth["project_id"]
    sample_id = setup_db_auth["sample_id"]

    for username in ("alice", "bob"):
        await add_member(client, admin_token, project_id, username, "annotator")

    alice_token = await get_auth_token(client, "alice", "alice_pass")
    bob_token = await get_auth_token(client, "bob", "bob_pass")

    await put_annotations(client, project_id, sample_id, bob_token, "bob_ann")
    await put_annotations(client, project_id, sample_id, alice_token, "alice_ann")

    loaded = await get_annotations(client, project_id, sample_id, alice_token)
    bobs_annotation = next(a for a in loaded if a["created_by"] == "bob")

    resp = await client.delete(
        f"/projects/{project_id}/samples/{sample_id}/annotations/{bobs_annotation['_id']}",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 200, resp.text

    remaining = await get_annotations(client, project_id, sample_id, admin_token)
    assert [a["label"] for a in remaining] == ["alice_ann"]

    # Deleting it again is a 404, not a silent success.
    repeat = await client.delete(
        f"/projects/{project_id}/samples/{sample_id}/annotations/{bobs_annotation['_id']}",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert repeat.status_code == 404


@pytest.mark.asyncio
async def test_cross_project_annotation_edit_is_scoped_out(
    setup_db_auth, unauthenticated_api_client
):
    """An annotation ID from another project cannot be reached through this project.

    Both the in-place edit and the single delete filter on project and sample as well
    as the annotation ID, so passing a foreign ID must be a no-op rather than an edit.
    """
    client = unauthenticated_api_client
    admin_token = await get_auth_token(
        unauthenticated_api_client, "admin", "admin_pass"
    )
    project_id = setup_db_auth["project_id"]
    sample_id = setup_db_auth["sample_id"]
    other_project_id = setup_db_auth["other_project_id"]
    other_sample_id = setup_db_auth["other_sample_id"]

    await add_member(client, admin_token, other_project_id, "bob", "annotator")
    bob_token = await get_auth_token(client, "bob", "bob_pass")
    await put_annotations(
        client, other_project_id, other_sample_id, bob_token, "other_project_ann"
    )
    foreign = (
        await get_annotations(client, other_project_id, other_sample_id, admin_token)
    )[0]

    await add_member(client, admin_token, project_id, "alice", "annotator")
    alice_token = await get_auth_token(client, "alice", "alice_pass")

    # Alice tries to edit it through her own project's save endpoint.
    resp = await client.put(
        f"/projects/{project_id}/samples/{sample_id}/annotations",
        json=[{**foreign, "label": "hijacked"}],
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 200

    # ...and to delete it.
    del_resp = await client.delete(
        f"/projects/{project_id}/samples/{sample_id}/annotations/{foreign['_id']}",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert del_resp.status_code == 404

    untouched = await get_annotations(
        client, other_project_id, other_sample_id, admin_token
    )
    assert len(untouched) == 1
    assert untouched[0]["label"] == "other_project_ann"
    assert untouched[0]["created_by"] == "bob"
