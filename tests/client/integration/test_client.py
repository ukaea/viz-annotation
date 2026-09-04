import pytest
from toktagger.api.schemas.projects import Project as ProjectSchema
from toktagger.client.client import (
    Project,
    TokTaggerClient,
)
from toktagger.client.exceptions import (
    NotFoundError,
    TokTaggerAPIError,
    TokTaggerClientError,
)

# Valid ObjectId format but absent from the server database
MISSING_OBJECT_ID = "0" * 24


def test_health(client):
    health = client.health()
    assert health["name"] == "TokTagger"
    assert health["db_connected"] is True
    assert health["testing_mode"] is True


def test_404_raises_not_found_error(client):
    with pytest.raises(NotFoundError) as exc_info:
        client._request("GET", f"/projects/{MISSING_OBJECT_ID}")
    assert exc_info.value.status_code == 404
    assert "Project not found" in exc_info.value.detail


def test_invalid_id_raises_api_error(client):
    # Invalid ObjectId format -> 400 from the server
    with pytest.raises(TokTaggerAPIError) as exc_info:
        client._request("GET", "/projects/not-a-valid-id")
    assert not isinstance(exc_info.value, NotFoundError)
    assert exc_info.value.status_code == 400


def test_bootstrapped_project_from_live_server(client, seeded_project):
    raw = client._request("GET", f"/projects/{seeded_project}").json()
    project = client._bootstrap_project(raw)

    assert isinstance(project, Project)
    assert isinstance(project, ProjectSchema)
    assert project.id == seeded_project
    assert project._client is client
    # Bootstrapping must not change serialization output
    assert project.model_dump() == ProjectSchema.model_validate(raw).model_dump()


def test_connection_error_is_wrapped():
    # Nothing is listening on this port
    client = TokTaggerClient(base_url="http://127.0.0.1:1", timeout=1)
    with pytest.raises(TokTaggerClientError):
        client.health()
