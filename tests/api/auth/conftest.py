"""Conftest for auth tests — uses mongita (no Docker required)."""

import pytest_asyncio
from httpx import AsyncClient


async def get_auth_token(client: AsyncClient, username: str, password: str) -> str:
    """Obtain a JWT access token for the given user."""
    resp = await client.post(
        "/auth/token",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 200, f"Login failed ({username}): {resp.text}"
    return resp.json()["access_token"]


async def create_project(
    client: AsyncClient, token: str, name: str = "test_project"
) -> str:
    """Create a project as the given token; return the project_id."""
    resp = await client.post(
        "/projects",
        json={
            "name": name,
            "task": "time-series",
            "query_strategy": "sequential",
            "data_loader": "tabular",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["_id"]


async def add_member(
    client: AsyncClient, token: str, project_id: str, username: str, role: str
) -> None:
    """Add `username` to `project_id` with `role`, authenticated as `token`."""
    resp = await client.post(
        f"/projects/{project_id}/members",
        json={"username": username, "role": role},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text


async def create_project_and_sample(
    client: AsyncClient, token: str, name: str = "test_project"
) -> tuple[str, str]:
    """Create a project then add one sample; return (project_id, sample_id)."""
    project_id = await create_project(client, token, name=name)
    sample_resp = await client.post(
        f"/projects/{project_id}/samples",
        json=[{"shot_id": 1, "data": {"file_name": "t.csv", "type": "csv"}}],
        headers={"Authorization": f"Bearer {token}"},
    )
    assert sample_resp.status_code == 200, sample_resp.text
    sample_id = sample_resp.json()[0]
    return project_id, sample_id


@pytest_asyncio.fixture(scope="function")
async def project_setup(unauthenticated_api_client, setup_db_auth):
    """setup_db_auth, plus an admin-created project with one sample."""
    client = unauthenticated_api_client
    admin_token = await get_auth_token(client, "admin", "admin_pass")
    project_id, sample_id = await create_project_and_sample(client, admin_token)
    return {
        **setup_db_auth,
        "admin_token": admin_token,
        "project_id": project_id,
        "sample_id": sample_id,
    }
