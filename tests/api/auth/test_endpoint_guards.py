"""
Integration tests verifying that membership guards are enforced across all resource
endpoints (samples, project-level annotations, sample-level annotation delete, data).

Permission matrix (role -> expected outcome), asserted explicitly for every action
below rather than relying on weaker roles being implied by stronger ones:
  - Unauthenticated       -> 401 on everything (no token at all)
  - Non-member             -> 403 on everything (authenticated, not a project member)
  - Viewer                 -> 200 on reads, 403 on writes/deletes
  - Annotator               -> 200 on reads and writes, 403 on destructive deletes
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
            "annotator": _status(200),
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
            "annotator": _status(403),
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
            "annotator": _status(403),
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

    path = action.path.format(project_id=project_id, sample_id=sample_id)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    kwargs = {"headers": headers}
    if action.body is not None:
        kwargs["json"] = action.body(project_id, sample_id)

    resp = await client.request(action.method, path, **kwargs)

    check = action.expected[role]
    assert check(resp.status_code), (
        f"{role} -> {action_name} {action.method} {path}: unexpected status {resp.status_code} ({resp.text})"
    )
