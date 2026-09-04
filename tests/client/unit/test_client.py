import httpx
import pytest
import tests.db_definitions as db_definitions
from toktagger.api.schemas.projects import Project as ProjectSchema
from toktagger.api.schemas.samples import Sample as SampleSchema
from toktagger.client.client import (
    Project,
    Sample,
    TokTaggerClient,
)
from toktagger.client.exceptions import (
    MultipleResultsFound,
    NotFoundError,
    TokTaggerAPIError,
    TokTaggerClientError,
)

PROJECT_ID = "0" * 24
SAMPLE_ID = "1" * 24


def make_client(handler, **kwargs) -> TokTaggerClient:
    return TokTaggerClient(transport=httpx.MockTransport(handler), **kwargs)


def make_raw_project() -> dict:
    # Simulate the JSON the server returns for GET /projects/{id}
    raw = db_definitions.PROJECT_1.model_dump(mode="json", by_alias=True)
    raw["_id"] = PROJECT_ID
    return raw


def make_raw_sample() -> dict:
    # Simulate the JSON the server returns for GET /projects/{pid}/samples/{id}
    raw = db_definitions.SAMPLE_1.model_dump(mode="json", by_alias=True)
    raw["_id"] = SAMPLE_ID
    raw["project_id"] = PROJECT_ID
    raw["validated_annotations"] = False
    return raw


def test_health_parses_json():
    client = make_client(
        lambda request: httpx.Response(200, json={"name": "TokTagger"})
    )
    assert client.health() == {"name": "TokTagger"}


def test_404_raises_not_found_error():
    client = make_client(
        lambda request: httpx.Response(404, json={"detail": "Project not found."})
    )
    with pytest.raises(NotFoundError) as exc_info:
        client._request("GET", f"/projects/{PROJECT_ID}")
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Project not found."


def test_non_404_error_raises_api_error():
    client = make_client(lambda request: httpx.Response(500, json={"detail": "boom"}))
    with pytest.raises(TokTaggerAPIError) as exc_info:
        client._request("GET", "/projects")
    assert not isinstance(exc_info.value, NotFoundError)
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "boom"


def test_non_json_error_body_uses_text():
    client = make_client(lambda request: httpx.Response(400, text="Bad Request"))
    with pytest.raises(TokTaggerAPIError) as exc_info:
        client._request("GET", "/projects")
    assert exc_info.value.detail == "Bad Request"


def test_list_error_detail_is_stringified():
    # FastAPI validation errors return detail as a list
    client = make_client(
        lambda request: httpx.Response(
            422, json={"detail": [{"msg": "field required"}]}
        )
    )
    with pytest.raises(TokTaggerAPIError) as exc_info:
        client._request("POST", "/projects")
    assert "field required" in exc_info.value.detail


def test_transport_error_wrapped():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = make_client(handler)
    with pytest.raises(TokTaggerClientError):
        client.health()


def test_connection_error_is_wrapped():
    # Nothing is listening on this port
    client = TokTaggerClient(base_url="http://127.0.0.1:1", timeout=1)
    with pytest.raises(TokTaggerClientError):
        client.health()


def test_base_url_trailing_slash_removed():
    client = make_client(
        lambda request: httpx.Response(200, json={}), base_url="http://localhost:8002/"
    )
    assert client._base_url == "http://localhost:8002"


def test_headers_are_passed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"token": request.headers.get("x-token")})

    client = make_client(handler, headers={"X-Token": "secret"})
    assert client.health()["token"] == "secret"


def test_context_manager_closes_client():
    client = make_client(lambda request: httpx.Response(200, json={}))
    with client:
        assert not client._http.is_closed
    assert client._http.is_closed


def test_bootstrap_project():
    client = TokTaggerClient()
    raw = make_raw_project()
    project = client._bootstrap_project(raw)

    assert isinstance(project, Project)
    assert isinstance(project, ProjectSchema)
    assert project.id == PROJECT_ID
    assert project.name == db_definitions.PROJECT_1.name
    assert project._client is client
    # Bootstrapping must not change serialization output
    assert project.model_dump() == ProjectSchema.model_validate(raw).model_dump()


def test_bootstrap_sample():
    client = TokTaggerClient()
    raw = make_raw_sample()
    sample = client._bootstrap_sample(raw)

    assert isinstance(sample, Sample)
    assert isinstance(sample, SampleSchema)
    assert sample.id == SAMPLE_ID
    assert sample.project_id == PROJECT_ID
    assert sample._client is client
    assert sample.model_dump() == SampleSchema.model_validate(raw).model_dump()


def test_hand_constructed_model_has_no_client():
    project = Project.model_validate(make_raw_project())
    assert project._client is None
    sample = Sample.model_validate(make_raw_sample())
    assert sample._client is None


def test_error_hierarchy():
    # Client-side lookup errors: no status code, not API errors
    multiple = MultipleResultsFound("2 projects matched name 'tokamak'")
    assert isinstance(multiple, TokTaggerClientError)
    assert not isinstance(multiple, TokTaggerAPIError)
    not_found = NotFoundError(None, "no match")
    assert isinstance(not_found, TokTaggerAPIError)
    assert not_found.status_code is None
