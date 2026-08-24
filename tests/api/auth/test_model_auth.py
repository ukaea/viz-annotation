"""
Tests for model × auth interactions:
  1. Internal API token lets Ray-worker callbacks bypass the annotator guard.
  2. Unauthenticated sender is rejected when auth is required.
  3. Non-admin bulk import enforces created_by = current user.
  4. Global admin bulk import allows arbitrary created_by.
  5. Usernames with reserved prefixes ("model::", "annotators::", "__") are rejected.
  6. A user whose username matches a model-type string cannot corrupt
     "model::<type>" prefixed predictions.
"""

import pytest

from tests.api.auth.conftest import get_auth_token
from toktagger.api.auth.core import get_internal_token


def annotation_payload(
    label: str = "lbl", created_by: str = "placeholder", shot_id: int = 1
):
    """Payload suitable for bulk import (PUT /projects/{id}/annotations).
    shot_id must match an existing sample — the default matches the sample
    created by setup (shot_id=1).
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
async def test_internal_token_accepted_for_import(
    setup_db_auth, unauthenticated_api_client
):
    """PUT /annotations with the server-internal token should be accepted as admin."""
    client = unauthenticated_api_client
    project_id = setup_db_auth["project_id"]

    internal_token = get_internal_token()
    resp = await client.put(
        f"/projects/{project_id}/annotations",
        json=annotation_payload(created_by="alice"),
        headers={"Authorization": f"Bearer {internal_token}"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_no_token_rejected_for_import_in_auth_mode(
    setup_db_auth, unauthenticated_api_client
):
    """PUT /annotations with no token must be rejected when auth is required."""
    client = unauthenticated_api_client
    project_id = setup_db_auth["project_id"]

    resp = await client.put(
        f"/projects/{project_id}/annotations",
        json=annotation_payload(),
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_import_non_admin_created_by_overwritten(
    setup_db_auth, unauthenticated_api_client
):
    """An annotator importing with a spoofed created_by should have it replaced."""
    client = unauthenticated_api_client
    admin_token = await get_auth_token(
        unauthenticated_api_client, "admin", "admin_pass"
    )
    project_id = setup_db_auth["project_id"]

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
async def test_import_admin_is_attributed_as_self(
    setup_db_auth, unauthenticated_api_client
):
    """A global admin importing annotations is recorded as the author; a
    supplied created_by is ignored so authorship stays auditable."""
    client = unauthenticated_api_client
    admin_token = await get_auth_token(
        unauthenticated_api_client, "admin", "admin_pass"
    )
    project_id = setup_db_auth["project_id"]

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
    assert annotations[0]["created_by"] == "admin"


@pytest.mark.asyncio
async def test_internal_token_preserves_arbitrary_created_by(
    setup_db_auth, unauthenticated_api_client
):
    """The internal token (Ray worker) can import with model:: prefixed created_by."""
    client = unauthenticated_api_client
    admin_token = await get_auth_token(
        unauthenticated_api_client, "admin", "admin_pass"
    )
    project_id = setup_db_auth["project_id"]

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
async def test_username_with_model_prefix_rejected(
    unauthenticated_api_client, setup_db_auth
):
    client = unauthenticated_api_client
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    resp = await client.post(
        "/users",
        json={
            "username": "model::disruption_cnn",
            "password": "pass123",
            "global_role": "user",
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_username_with_annotators_prefix_rejected(
    unauthenticated_api_client, setup_db_auth
):
    client = unauthenticated_api_client
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    resp = await client.post(
        "/users",
        json={
            "username": "annotators::peak_detection",
            "password": "pass123",
            "global_role": "user",
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_username_with_dunder_prefix_rejected(
    unauthenticated_api_client, setup_db_auth
):
    client = unauthenticated_api_client
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    resp = await client.post(
        "/users",
        json={
            "username": "__internal__",
            "password": "pass123",
            "global_role": "user",
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_user_save_does_not_corrupt_model_prefixed_predictions(
    setup_db_auth, unauthenticated_api_client
):
    """A human user named 'disruption_cnn' saving annotations must NOT delete
    model predictions stored as 'model::disruption_cnn'. The prefix is the separator.

    This hand-crafts the "model::" annotation via a direct PUT with the internal
    token, so it only proves the import endpoint's exemption logic — not that the
    real /predict pipeline actually produces that prefix. For the real end-to-end
    version (actual /predict call, real Ray worker, real per-user JWTs), see
    test_predict_endpoint_survives_same_named_human_save in
    tests/api/routers/test_models.py.
    """
    client = unauthenticated_api_client
    admin_token = await get_auth_token(
        unauthenticated_api_client, "admin", "admin_pass"
    )
    project_id = setup_db_auth["project_id"]
    sample_id = setup_db_auth["sample_id"]

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

    # Insert a model prediction via the internal tokens.
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
