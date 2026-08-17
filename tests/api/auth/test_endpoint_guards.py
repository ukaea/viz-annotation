"""
Integration tests verifying that membership guards are enforced across all resource
endpoints (samples, project-level annotations, sample-level annotation delete, data).

Permission matrix (role -> expected outcome), asserted explicitly for every action
below rather than relying on weaker roles being implied by stronger ones:
  - Unauthenticated       -> 401 on everything (no token at all)
  - Non-member             -> 403 on everything (authenticated, not a project member)
  - Viewer                 -> 200 on reads, 403 on every write and delete
  - Annotator               -> 200 on reads, on annotation writes and deletes, on
                              marking samples validated, and on editing/deleting the
                              project itself; 403 on adding or deleting samples, on
                              member management and on deleting a trained model
                              artifact
  - Project admin (member role="admin") -> 200 on everything
  - Global admin            -> 200 on everything
"""

import typing

import pytest

from tests.api.auth.conftest import add_member, get_auth_token

ROLES = [
    "unauthenticated",
    "non_member",
    "viewer",
    "annotator",
    "project_admin",
    "global_admin",
]


async def _get_role_token(client, admin_token, project_id, role):
    """Return a bearer token for `role` on `project_id` (None for unauthenticated)."""
    if role == "unauthenticated":
        return None
    if role == "global_admin":
        return admin_token
    if role == "non_member":
        return await get_auth_token(client, "bob", "bob_pass")
    if role in ("viewer", "annotator"):
        await add_member(client, admin_token, project_id, "alice", role)
    elif role == "project_admin":
        await add_member(client, admin_token, project_id, "alice", "admin")
    else:
        raise ValueError(f"Unknown role: {role}")
    return await get_auth_token(client, "alice", "alice_pass")


def _status(expected):
    return lambda code: code == expected


def _not_forbidden():
    """The endpoint may fail for other reasons (e.g. missing data file), but must
    not be blocked by auth/membership."""
    return lambda code: code not in (401, 403)


class Action(typing.NamedTuple):
    method: str
    path: str
    expected: dict[str, typing.Callable[[int], bool]]
    body: typing.Callable[[str, str], object] | None = None


# Every action this file guards, and the expected outcome for each role.
ACTIONS: dict[str, Action] = {
    "list_samples": Action(
        method="GET",
        path="/projects/{project_id}/samples",
        expected={
            "unauthenticated": _status(401),
            "non_member": _status(403),
            "viewer": _status(200),
            "annotator": _status(200),
            "project_admin": _status(200),
            "global_admin": _status(200),
        },
    ),
    # Which samples a project holds is project configuration, not annotation work, so
    # the three actions below are project-admin only.
    "add_samples": Action(
        method="POST",
        path="/projects/{project_id}/samples",
        body=lambda *_: [
            {"shot_id": 99, "data": {"file_name": "x.csv", "type": "csv"}}
        ],
        expected={
            "unauthenticated": _status(401),
            "non_member": _status(403),
            "viewer": _status(403),
            "annotator": _status(403),
            "project_admin": _status(200),
            "global_admin": _status(200),
        },
    ),
    "delete_sample": Action(
        method="DELETE",
        path="/projects/{project_id}/samples/{sample_id}",
        expected={
            "unauthenticated": _status(401),
            "non_member": _status(403),
            "viewer": _status(403),
            "annotator": _status(403),
            "project_admin": _status(200),
            "global_admin": _status(200),
        },
    ),
    "delete_all_samples": Action(
        method="DELETE",
        path="/projects/{project_id}/samples",
        expected={
            "unauthenticated": _status(401),
            "non_member": _status(403),
            "viewer": _status(403),
            "annotator": _status(403),
            "project_admin": _status(200),
            "global_admin": _status(200),
        },
    ),
    # Annotator-level, unlike the sample actions above: SampleUpdate carries only
    # validated_annotations, which every annotation save and clear sets.
    "update_samples": Action(
        method="PUT",
        path="/projects/{project_id}/samples",
        body=lambda _, sample_id: [
            {"_id": sample_id, "updates": {"validated_annotations": True}}
        ],
        expected={
            "unauthenticated": _status(401),
            "non_member": _status(403),
            "viewer": _status(403),
            "annotator": _status(200),
            "project_admin": _status(200),
            "global_admin": _status(200),
        },
    ),
    "get_project_annotations": Action(
        method="GET",
        path="/projects/{project_id}/annotations",
        expected={
            "unauthenticated": _status(401),
            "non_member": _status(403),
            "viewer": _status(200),
            "annotator": _status(200),
            "project_admin": _status(200),
            "global_admin": _status(200),
        },
    ),
    "import_annotations": Action(
        method="PUT",
        path="/projects/{project_id}/annotations",
        body=lambda project_id, sample_id: [
            {
                "label": "lbl",
                "time_min": 0.0,
                "time_max": 1.0,
                "type": "time_region",
                "validated": False,
                "created_by": "placeholder",  # server overwrites from JWT
                "shot_id": 1,
                "sample_id": sample_id,
                "project_id": project_id,
            }
        ],
        expected={
            "unauthenticated": _status(401),
            "non_member": _status(403),
            "viewer": _status(403),
            "annotator": _status(200),
            "project_admin": _status(200),
            "global_admin": _status(200),
        },
    ),
    "delete_project_annotations": Action(
        method="DELETE",
        path="/projects/{project_id}/annotations",
        expected={
            "unauthenticated": _status(401),
            "non_member": _status(403),
            "viewer": _status(403),
            "annotator": _status(200),
            "project_admin": _status(200),
            "global_admin": _status(200),
        },
    ),
    "delete_sample_annotations": Action(
        method="DELETE",
        path="/projects/{project_id}/samples/{sample_id}/annotations",
        expected={
            "unauthenticated": _status(401),
            "non_member": _status(403),
            "viewer": _status(403),
            "annotator": _status(200),
            "project_admin": _status(200),
            "global_admin": _status(200),
        },
    ),
    "get_data": Action(
        method="POST",
        path="/projects/{project_id}/samples/{sample_id}/data",
        body=lambda *_: {},
        expected={
            "unauthenticated": _status(401),
            "non_member": _status(403),
            "viewer": _not_forbidden(),
            "annotator": _not_forbidden(),
            "project_admin": _not_forbidden(),
            "global_admin": _not_forbidden(),
        },
    ),
    "update_project": Action(
        method="PUT",
        path="/projects/{project_id}",
        body=lambda project_id, _: {
            "_id": project_id,
            "name": "renamed_project",
            "task": "time-series",
            "query_strategy": "sequential",
            "data_loader": "tabular",
        },
        expected={
            "unauthenticated": _status(401),
            "non_member": _status(403),
            "viewer": _status(403),
            "annotator": _status(200),
            "project_admin": _status(200),
            "global_admin": _status(200),
        },
    ),
    "delete_project": Action(
        method="DELETE",
        path="/projects/{project_id}",
        expected={
            "unauthenticated": _status(401),
            "non_member": _status(403),
            "viewer": _status(403),
            "annotator": _status(200),
            "project_admin": _status(200),
            "global_admin": _status(200),
        },
    ),
    # A well-formed but absent annotation ID: an authorised caller gets past the guard
    # and finds nothing (404), so a 403 here can only come from the permission check.
    "delete_single_annotation": Action(
        method="DELETE",
        path="/projects/{project_id}/samples/{sample_id}/annotations/{missing_id}",
        expected={
            "unauthenticated": _status(401),
            "non_member": _status(403),
            "viewer": _status(403),
            "annotator": _status(404),
            "project_admin": _status(404),
            "global_admin": _status(404),
        },
    ),
    # Deletes prediction annotations, so annotators may run it. May 503 when the ML
    # extras aren't installed, hence _not_forbidden for the roles that are allowed.
    "delete_predictions": Action(
        method="DELETE",
        path="/projects/{project_id}/models/dummy/predict",
        expected={
            "unauthenticated": _status(401),
            "non_member": _status(403),
            "viewer": _status(403),
            "annotator": _not_forbidden(),
            "project_admin": _not_forbidden(),
            "global_admin": _not_forbidden(),
        },
    ),
    # A trained model artifact is not an annotation, so this stays project-admin only.
    "delete_model": Action(
        method="DELETE",
        path="/projects/{project_id}/models/dummy",
        expected={
            "unauthenticated": _status(401),
            "non_member": _status(403),
            "viewer": _status(403),
            "annotator": _status(403),
            "project_admin": _not_forbidden(),
            "global_admin": _not_forbidden(),
        },
    ),
    "add_member": Action(
        method="POST",
        path="/projects/{project_id}/members",
        body=lambda *_: {"username": "bob", "role": "viewer"},
        expected={
            "unauthenticated": _status(401),
            "non_member": _status(403),
            "viewer": _status(403),
            "annotator": _status(403),
            "project_admin": _status(200),
            "global_admin": _status(200),
        },
    ),
    "remove_member": Action(
        method="DELETE",
        path="/projects/{project_id}/members/{bob_id}",
        expected={
            "unauthenticated": _status(401),
            "non_member": _status(403),
            "viewer": _status(403),
            "annotator": _status(403),
            "project_admin": _not_forbidden(),
            "global_admin": _not_forbidden(),
        },
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("action_name", list(ACTIONS))
async def test_permission_matrix(project_setup, action_name, role):
    client = project_setup["client"]
    admin_token = project_setup["admin_token"]
    project_id = project_setup["project_id"]
    sample_id = project_setup["sample_id"]
    action = ACTIONS[action_name]

    token = await _get_role_token(client, admin_token, project_id, role)

    path = action.path.format(
        project_id=project_id,
        sample_id=sample_id,
        bob_id=project_setup["bob_id"],
        # Valid ObjectId shape, guaranteed not to exist.
        missing_id="0" * 24,
    )
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    kwargs = {"headers": headers}
    if action.body is not None:
        kwargs["json"] = action.body(project_id, sample_id)

    resp = await client.request(action.method, path, **kwargs)

    check = action.expected[role]
    assert check(resp.status_code), (
        f"{role} -> {action_name} {action.method} {path}: unexpected status {resp.status_code} ({resp.text})"
    )
