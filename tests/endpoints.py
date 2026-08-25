import pathlib

import requests

from tests import db_definitions
from toktagger.api.schemas.annotations import TimeRegion

# Shared session for all e2e helper requests below, so auth only needs to be
# configured once (see set_auth_token) rather than threaded through every
# helper function's signature and every call site across the e2e test files.
session = requests.Session()


def set_auth_token(token: str) -> None:
    """Authenticate all subsequent tests.endpoints.* requests as this user."""
    session.headers.update({"Authorization": f"Bearer {token}"})


def create_user(
    username: str,
    password: str,
    role: str = "user",
    must_change_password: bool = False,
) -> str:
    response = session.post(
        "http://localhost:8002/users",
        json={
            "username": username,
            "password": password,
            "global_role": role,
        },
    )
    assert response.status_code == 200, response.text
    user_id = response.json()["_id"]

    # POST /users always forces a password change. Test-created accounts should be
    # usable immediately rather than stuck behind that redirect, so clear it here;
    # pass must_change_password=True to test the redirect itself.
    if not must_change_password:
        response = session.put(
            f"http://localhost:8002/users/{user_id}",
            json={"must_change_password": False},
        )
        assert response.status_code == 200, response.text

    return user_id


def get_user(user_id: str) -> dict:
    response = session.get(f"http://localhost:8002/users/{user_id}")
    assert response.status_code == 200, response.text
    return response.json()


def add_project_member(project_id: str, username: str, role: str = "annotator"):
    response = session.post(
        f"http://localhost:8002/projects/{project_id}/members",
        json={"username": username, "role": role},
    )
    assert response.status_code == 200, response.text


def get_project_members(project_id: str) -> list[dict]:
    response = session.get(f"http://localhost:8002/projects/{project_id}/members")
    assert response.status_code == 200, response.text
    return response.json()


def create_project(
    name: str, task: str, data_loader: str, query_strategy: str = "random"
) -> str:
    project = {
        "name": name,
        "task": task,
        "query_strategy": query_strategy,
        "data_loader": data_loader,
    }

    response = session.post(
        "http://localhost:8002/projects",
        json=project,
    )
    assert response.status_code == 200

    project_id = response.json()["_id"]
    return project_id


def create_local_samples(
    project_id: str,
    shot_ids: list[int],
    base_path: str,
    columns: list[str] | None = None,
    file_names: list[str] | None = None,
):
    samples = []
    if not file_names:
        file_names = [f"{shot_id}.parquet" for shot_id in shot_ids]

    base_path = pathlib.Path(base_path)
    for file_name, shot_id in zip(file_names, shot_ids):
        sample = {
            "project_id": project_id,
            "shot_id": shot_id,
            "data": {
                "file_name": str(base_path / file_name),
                "type": "parquet",
                "protocol": "file",
                "signal_names": columns,
            },
        }
        samples.append(sample)

    response = session.post(
        f"http://localhost:8002/projects/{project_id}/samples", json=samples
    )
    assert response.status_code == 200
    return response.json()


def create_image_samples(
    project_id: str,
    shot_id: int,
    base_path: str,
    file_type: str,
):
    samples = []

    sample = {
        "project_id": project_id,
        "shot_id": shot_id,
        "data": {
            "file_name": str(base_path),
            "type": file_type,
            "protocol": "file",
        },
    }
    samples.append(sample)

    response = session.post(
        f"http://localhost:8002/projects/{project_id}/samples", json=samples
    )
    assert response.status_code == 200
    return response.json()


def create_uda_samples(
    project_id: str,
    shot_ids: list[int],
    signal_names: list[str] | None = None,
):
    if signal_names is None:
        signal_names = ["ip", "ANE_DENSITY"]
    samples = []
    for shot_id in shot_ids:
        sample = {
            "project_id": project_id,
            "shot_id": shot_id,
            "data": {
                "signal_names": signal_names,
                "protocol": "uda",
            },
        }
        samples.append(sample)

    response = session.post(
        f"http://localhost:8002/projects/{project_id}/samples", json=samples
    )

    assert response.status_code == 200
    return response.json()


def create_model_samples(setup_model_samples):
    response = session.post(
        "http://localhost:8002/projects",
        json=db_definitions.PROJECT_2.model_dump(mode="json"),
    )
    assert response.status_code == 200

    project_id = response.json()["_id"]

    response = session.post(
        f"http://localhost:8002/projects/{project_id}/samples",
        json=[sample.model_dump(mode="json") for sample in setup_model_samples],
    )
    assert response.status_code == 200
    sample_ids = response.json()

    return project_id, sample_ids


def create_query_strategy_samples(query_strategy: str):
    # Create project
    project_id = create_project(
        "Test Project", "time-series", "tabular", query_strategy=query_strategy
    )
    # And create samples,
    # but create them in reverse order of shot ID so that sorting by timestamp gives you opposite order
    sample_ids = create_local_samples(
        project_id,
        list(range(10004, 9999, -1)),
        pathlib.Path(__file__).parent,
        ["Ip"],
        ["10000.parquet"] * 5,
    )
    sample_ids.reverse()

    # Samples 10000, 10001 will have validated annotations
    for sample_id in sample_ids[:2]:
        flat_top = TimeRegion(
            label="Flat Top",
            created_by="manual",
            time_min=10,
            time_max=20,
            validated=True,
            uncertainty=0,
        )
        response = session.put(
            f"http://localhost:8002/projects/{project_id}/samples/{sample_id}/annotations",
            json=[flat_top.model_dump(mode="json")],
        )
        assert response.status_code == 200

    # Sample 10002 will have middle uncertain annotation
    flat_top = TimeRegion(
        label="Flat Top",
        created_by="peak_detection",
        time_min=10,
        time_max=20,
        validated=False,
        uncertainty=0.5,
    )
    response = session.put(
        f"http://localhost:8002/projects/{project_id}/samples/{sample_ids[2]}/annotations",
        json=[flat_top.model_dump(mode="json")],
    )
    assert response.status_code == 200

    # Sample 10003 will have no annotations

    # Sample 10004 will have most uncertain annotation, and least uncertain annotation
    # Should use most uncertain in query strategy
    ramp_up = TimeRegion(
        label="Flat Top",
        created_by="peak_detection",
        time_min=30,
        time_max=40,
        validated=False,
        uncertainty=0.9,
    )
    flat_top = TimeRegion(
        label="Ramp Up",
        created_by="peak_detection",
        time_min=10,
        time_max=20,
        validated=False,
        uncertainty=0.1,
    )
    response = session.put(
        f"http://localhost:8002/projects/{project_id}/samples/{sample_ids[4]}/annotations",
        json=[ramp_up.model_dump(mode="json"), flat_top.model_dump(mode="json")],
    )
    assert response.status_code == 200

    return project_id, sample_ids
