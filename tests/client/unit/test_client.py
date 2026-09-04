import httpx
import pytest
import tests.db_definitions as db_definitions
from toktagger.api.schemas.projects import Project as ProjectSchema
from toktagger.api.schemas.samples import Sample as SampleSchema, SampleSummary
from toktagger.client.client import (
    Project,
    Sample,
    TokTaggerClient,
)
from toktagger.client.exceptions import (
    MultipleResultsFoundError,
    NotFoundError,
)
from tests.client.conftest import (
    make_client,
    PROJECT_1_ID,
    PROJECT_2_ID,
    SAMPLE_1_ID,
    SAMPLE_2_ID,
)


def test_health_parses_json():
    client = make_client(
        lambda request: httpx.Response(200, json={"name": "TokTagger"})
    )
    assert client.health() == {"name": "TokTagger"}


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


def test_bootstrap_project(project_1):
    client = TokTaggerClient()
    project = client._bootstrap_project(project_1)

    assert isinstance(project, Project)
    assert isinstance(project, ProjectSchema)
    assert project.id == PROJECT_1_ID
    assert project.name == db_definitions.PROJECT_1.name
    assert project._client is client
    # Bootstrapping must not change serialization output
    assert project.model_dump() == ProjectSchema.model_validate(project_1).model_dump()


def test_bootstrap_sample(sample_1):
    client = TokTaggerClient()
    sample = client._bootstrap_sample(sample_1)

    assert isinstance(sample, Sample)
    assert isinstance(sample, SampleSchema)
    assert sample.id == SAMPLE_1_ID
    assert sample.project_id == PROJECT_1_ID
    assert sample._client is client
    assert sample.model_dump() == SampleSchema.model_validate(sample_1).model_dump()


def test_list_projects_returns_bootstrapped(project_1, project_2):
    requests_seen = []

    def handler(request):
        requests_seen.append(request)
        return httpx.Response(200, json=[project_1, project_2])

    client = make_client(handler)
    projects = client.list_projects()

    assert [p.id for p in projects] == [PROJECT_1_ID, PROJECT_2_ID]
    assert all(isinstance(p, Project) for p in projects)
    assert all(p._client is client for p in projects)
    assert requests_seen[0].url.path == "/projects"
    # defaults are sent explicitly
    params = requests_seen[0].url.params
    assert params["sort_by"] == "_id"
    assert params["sort_direction"] == "descending"
    assert params["start"] == "0"
    # the optional name filter is omitted when None
    assert "name" not in params


def test_list_projects_forwards_filters(project_1, project_2):
    requests_seen = []

    def handler(request):
        requests_seen.append(request)
        return httpx.Response(200, json=[project_1, project_2])

    client = make_client(handler)
    client.list_projects(
        name="test", start=5, count=3, sort_by="name", sort_direction="ascending"
    )

    params = requests_seen[0].url.params
    assert params["name"] == "test"
    assert params["start"] == "5"
    assert params["count"] == "3"
    assert params["sort_by"] == "name"
    assert params["sort_direction"] == "ascending"


def test_get_project_returns_bootstrapped(project_1):
    client = make_client(lambda request: httpx.Response(200, json=project_1))
    project = client.get_project(PROJECT_1_ID)
    assert isinstance(project, Project)
    assert project.id == PROJECT_1_ID
    assert project._client is client


def test_get_project_by_name_single_match(project_1):
    client = make_client(lambda request: httpx.Response(200, json=[project_1]))
    project = client.get_project_by_name("test_project_0")
    assert project.id == PROJECT_1_ID
    assert project._client is client


def test_get_project_by_name_no_match():
    client = make_client(lambda request: httpx.Response(200, json=[]))
    with pytest.raises(NotFoundError) as exc_info:
        client.get_project_by_name("does not exist")
    # client-side lookup error: no HTTP status code
    assert exc_info.value.status_code is None


def test_get_project_by_name_multiple_match(project_1, project_2):
    client = make_client(
        lambda request: httpx.Response(200, json=[project_1, project_2])
    )
    with pytest.raises(MultipleResultsFoundError) as exc_info:
        client.get_project_by_name("project")
    assert "2 projects" in str(exc_info.value)


# --- list_samples / get_sample / get_sample_by_shot_id / get_samples_summary ---


def test_list_samples_returns_bootstrapped(sample_1, sample_2):
    requests_seen = []

    def handler(request):
        requests_seen.append(request)
        return httpx.Response(200, json=[sample_1, sample_2])

    client = make_client(handler)
    samples = client.list_samples(PROJECT_1_ID)

    assert [s.id for s in samples] == [SAMPLE_1_ID, SAMPLE_2_ID]
    assert all(isinstance(s, Sample) for s in samples)
    assert all(s._client is client for s in samples)
    assert requests_seen[0].url.path == f"/projects/{PROJECT_1_ID}/samples"
    # Default None not forwarded
    assert "shot_id" not in requests_seen[0].url.params


def test_list_samples_forwards_shot_id_and_count(sample_1, sample_2):
    requests_seen = []

    def handler(request):
        requests_seen.append(request)
        return httpx.Response(200, json=[sample_1, sample_2])

    client = make_client(handler)
    client.list_samples(PROJECT_1_ID, shot_id=1, count=2)

    params = requests_seen[0].url.params
    assert params["shot_id"] == "1"
    assert params["count"] == "2"


def test_get_sample_returns_bootstrapped(sample_1):
    client = make_client(lambda request: httpx.Response(200, json=sample_1))
    sample = client.get_sample(PROJECT_1_ID, SAMPLE_1_ID)
    assert isinstance(sample, Sample)
    assert sample.id == SAMPLE_1_ID
    assert sample.project_id == PROJECT_1_ID
    assert sample._client is client


def test_get_sample_by_shot_id_single_match(sample_1):
    client = make_client(lambda request: httpx.Response(200, json=[sample_1]))
    sample = client.get_sample_by_shot_id(PROJECT_1_ID, 1)
    assert sample.id == SAMPLE_1_ID
    assert sample._client is client


def test_get_sample_by_shot_id_no_match():
    client = make_client(lambda request: httpx.Response(200, json=[]))
    with pytest.raises(NotFoundError) as exc_info:
        client.get_sample_by_shot_id(PROJECT_1_ID, 999)
    assert exc_info.value.status_code is None


def test_get_sample_by_shot_id_multiple_match(sample_1, sample_2):
    client = make_client(lambda request: httpx.Response(200, json=[sample_1, sample_2]))
    with pytest.raises(MultipleResultsFoundError) as exc_info:
        client.get_sample_by_shot_id(PROJECT_1_ID, 1)
    assert "2 samples" in str(exc_info.value)


# --- bootstrapped Project methods (delegation + chaining) ---


def test_project_list_samples_delegates(project_1, sample_1, sample_2):
    requests_seen = []

    def handler(request):
        requests_seen.append(request)
        if request.url.path == f"/projects/{PROJECT_1_ID}":
            return httpx.Response(200, json=project_1)
        return httpx.Response(200, json=[sample_1, sample_2])

    client = make_client(handler)
    project = client.get_project(PROJECT_1_ID)
    samples = project.list_samples(shot_id=1)

    assert [s.id for s in samples] == [SAMPLE_1_ID, SAMPLE_2_ID]
    assert all(isinstance(s, Sample) for s in samples)
    assert all(s._client is client for s in samples)
    # delegated to the samples endpoint with the project's own id
    assert requests_seen[-1].url.path == f"/projects/{PROJECT_1_ID}/samples"
    assert requests_seen[-1].url.params["shot_id"] == "1"


def test_project_get_sample_delegates(project_1, sample_1):
    def handler(request):
        if request.url.path == f"/projects/{PROJECT_1_ID}":
            return httpx.Response(200, json=project_1)
        return httpx.Response(200, json=sample_1)

    client = make_client(handler)
    project = client.get_project(PROJECT_1_ID)
    sample = project.get_sample(SAMPLE_1_ID)
    assert sample.id == SAMPLE_1_ID
    assert sample._client is client


def test_project_get_sample_by_shot_id_delegates(project_1, sample_1):
    def handler(request):
        if request.url.path == f"/projects/{PROJECT_1_ID}":
            return httpx.Response(200, json=project_1)
        return httpx.Response(200, json=[sample_1])

    client = make_client(handler)
    project = client.get_project(PROJECT_1_ID)
    sample = project.get_sample_by_shot_id(1)
    assert sample.id == SAMPLE_1_ID


def test_project_get_samples_summary_delegates(project_1):
    def handler(request):
        if request.url.path == f"/projects/{PROJECT_1_ID}":
            return httpx.Response(200, json=project_1)
        return httpx.Response(200, json={"total": 0})

    client = make_client(handler)
    project = client.get_project(PROJECT_1_ID)
    summary = project.get_samples_summary()
    assert isinstance(summary, SampleSummary)
    assert summary.total == 0


def test_chained_retrieval_style(project_1, sample_1):
    def handler(request):
        path = request.url.path
        if path == f"/projects/{PROJECT_1_ID}":
            return httpx.Response(200, json=project_1)
        if path == f"/projects/{PROJECT_1_ID}/samples/{SAMPLE_1_ID}":
            return httpx.Response(200, json=sample_1)
        return httpx.Response(404, json={"detail": f"unexpected {path}"})

    client = make_client(handler)
    sample = client.get_project(PROJECT_1_ID).get_sample(SAMPLE_1_ID)
    assert isinstance(sample, Sample)
    assert sample.id == SAMPLE_1_ID
    assert sample.project_id == PROJECT_1_ID
    assert sample._client is client
