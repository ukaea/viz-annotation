import httpx
import pytest
from toktagger.client.client import Project
from toktagger.client.exceptions import (
    MultipleResultsFoundError,
    NotFoundError,
    TokTaggerAPIError,
    TokTaggerClientError,
)
from tests.client.conftest import (
    make_client,
)


def test_404_raises_not_found_error():
    client = make_client(
        lambda request: httpx.Response(404, json={"detail": "Project not found."})
    )
    with pytest.raises(NotFoundError) as exc_info:
        client._request("GET", f"/projects/{'0' * 24}")
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


def test_error_hierarchy():
    # Client-side lookup errors: no status code, not API errors
    multiple = MultipleResultsFoundError("2 projects matched name 'tokamak'")
    assert isinstance(multiple, TokTaggerClientError)
    assert not isinstance(multiple, TokTaggerAPIError)
    not_found = NotFoundError(None, "no match")
    assert isinstance(not_found, TokTaggerAPIError)
    assert not_found.status_code is None


def test_project_method_without_client_raises(project_1, sample_1):
    project = Project.model_validate(project_1)
    assert project._client is None
    with pytest.raises(TokTaggerClientError, match="no client attached"):
        project.list_samples()
    with pytest.raises(TokTaggerClientError):
        project.get_sample(sample_1["_id"])
    with pytest.raises(TokTaggerClientError):
        project.get_sample_by_shot_id(1)
    with pytest.raises(TokTaggerClientError):
        project.get_samples_summary()
