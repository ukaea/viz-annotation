"""
Tests for model × auth interactions:
  1. Internal API token lets Ray-worker callbacks bypass the annotator guard.
  2. Unauthenticated sender is rejected when auth is required.
  3. Non-admin bulk import enforces created_by = current user.
  4. Global admin bulk import allows arbitrary created_by.
  5. Usernames with reserved prefixes ("model::", "__") are rejected.
  6. A user whose username matches a model-type string cannot corrupt
     "model::<type>" prefixed predictions.
"""

import pytest

from tests.api.auth.conftest import get_auth_token
from toktagger.api.auth.core import get_internal_token


async def create_project_and_sample(client, token):
    proj = await client.post(
        "/projects",
        json={
            "name": "model_auth_test",
            "task": "time-series",
            "query_strategy": "sequential",
            "data_loader": "tabular",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert proj.status_code == 200, proj.text
    project_id = proj.json()["_id"]

    sample = await client.post(
        f"/projects/{project_id}/samples",
        json=[{"shot_id": 1, "data": {"file_name": "t.csv", "type": "csv"}}],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert sample.status_code == 200, sample.text
    sample_id = sample.json()[0]
    return project_id, sample_id


def annotation_payload(
    label: str = "lbl", created_by: str = "placeholder", shot_id: int = 1
):
    """Payload suitable for bulk import (PUT /projects/{id}/annotations).
    shot_id must match an existing sample — the default matches the sample
    created by create_project_and_sample (shot_id=1).
    """
    return [
        {
            "label": label,
            "time_min": 0.0,
            "time_max": 1.0,
            "type": "time_region",
            "validated": False,
            "created_by": created_by,
            "shot_id": shot_id,
        }
    ]


@pytest.mark.asyncio
async def test_internal_token_accepted_for_import(auth_setup):
    """PUT /annotations with the server-internal token should be accepted as admin."""
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)

    internal_token = get_internal_token()
    resp = await client.put(
        f"/projects/{project_id}/annotations",
        json=annotation_payload(created_by="alice"),
        headers={"Authorization": f"Bearer {internal_token}"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_no_token_rejected_for_import_in_auth_mode(auth_setup):
    """PUT /annotations with no token must be rejected when auth is required."""
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, _ = await create_project_and_sample(client, admin_token)

    resp = await client.put(
        f"/projects/{project_id}/annotations",
        json=annotation_payload(),
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_import_non_admin_created_by_overwritten(auth_setup):
    """An annotator importing with a spoofed created_by should have it replaced."""
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)

    await client.post(
        f"/projects/{project_id}/members",
        json={"username": "alice", "role": "annotator"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    alice_token = await get_auth_token(client, "alice", "alice_pass")

    resp = await client.put(
        f"/projects/{project_id}/annotations",
        json=annotation_payload(created_by="bob"),
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert resp.status_code == 200

    # Verify: annotation stored as alice, not bob
    get_resp = await client.get(
        f"/projects/{project_id}/annotations",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert get_resp.status_code == 200
    annotations = get_resp.json()
    assert len(annotations) == 1
    assert annotations[0]["created_by"] == "alice"


@pytest.mark.asyncio
async def test_import_admin_can_set_arbitrary_created_by(auth_setup):
    """A global admin may import annotations attributed to any user."""
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)

    resp = await client.put(
        f"/projects/{project_id}/annotations",
        json=annotation_payload(created_by="alice"),
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200

    get_resp = await client.get(
        f"/projects/{project_id}/annotations",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    annotations = get_resp.json()
    assert len(annotations) == 1
    assert annotations[0]["created_by"] == "alice"


@pytest.mark.asyncio
async def test_internal_token_preserves_arbitrary_created_by(auth_setup):
    """The internal token (Ray worker) can import with model:: prefixed created_by."""
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)

    internal_token = get_internal_token()
    resp = await client.put(
        f"/projects/{project_id}/annotations",
        json=annotation_payload(created_by="model::disruption_cnn"),
        headers={"Authorization": f"Bearer {internal_token}"},
    )
    assert resp.status_code == 200

    get_resp = await client.get(
        f"/projects/{project_id}/annotations",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    annotations = get_resp.json()
    assert len(annotations) == 1
    assert annotations[0]["created_by"] == "model::disruption_cnn"


@pytest.mark.asyncio
async def test_username_with_model_prefix_rejected(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    resp = await client.post(
        "/users",
        json={
            "username": "model::disruption_cnn",
            "password": "pass123",
            "email": "",
            "global_role": "user",
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_username_with_dunder_prefix_rejected(auth_setup):
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    resp = await client.post(
        "/users",
        json={
            "username": "__internal__",
            "password": "pass123",
            "email": "",
            "global_role": "user",
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_user_save_does_not_corrupt_model_prefixed_predictions(auth_setup):
    """A human user named 'disruption_cnn' saving annotations must NOT delete
    model predictions stored as 'model::disruption_cnn'. The prefix is the separator.
    """
    client = auth_setup["client"]
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)

    # Create a human user whose name matches a model type (the collision scenario).
    create_resp = await client.post(
        "/users",
        json={
            "username": "disruption_cnn",
            "password": "pass123",
            "global_role": "user",
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert create_resp.status_code == 200

    # Insert a model prediction via the internal token.
    internal_token = get_internal_token()
    await client.put(
        f"/projects/{project_id}/annotations",
        json=annotation_payload(label="model_pred", created_by="model::disruption_cnn"),
        headers={"Authorization": f"Bearer {internal_token}"},
    )

    # The human user saves their own annotation for the same sample.
    await client.post(
        f"/projects/{project_id}/members",
        json={"username": "disruption_cnn", "role": "annotator"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    human_token = await get_auth_token(client, "disruption_cnn", "pass123")
    save_resp = await client.put(
        f"/projects/{project_id}/samples/{sample_id}/annotations",
        json=[
            {
                "label": "human_ann",
                "time_min": 0.0,
                "time_max": 1.0,
                "type": "time_region",
                "validated": True,
                "created_by": "placeholder",
            }
        ],
        headers={"Authorization": f"Bearer {human_token}"},
    )
    assert save_resp.status_code == 200

    # Both the model prediction and human annotation must survive — the model::
    # prefix provides complete namespace separation.
    get_resp = await client.get(
        f"/projects/{project_id}/annotations",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    annotations = get_resp.json()
    labels_by_author = {a["created_by"]: a["label"] for a in annotations}
    assert labels_by_author.get("model::disruption_cnn") == "model_pred"
    assert labels_by_author.get("disruption_cnn") == "human_ann"
