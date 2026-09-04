import pytest
from toktagger.api.schemas.samples import SampleSummary
from toktagger.client.client import (
    Project,
    Sample,
)
from toktagger.client.exceptions import (
    NotFoundError,
    MultipleResultsFoundError,
)

# Valid ObjectId format but absent from the server database
MISSING_OBJECT_ID = "0" * 24


def test_health(client):
    health = client.health()
    assert health["name"] == "TokTagger"
    assert health["db_connected"] is True
    assert health["testing_mode"] is True


def test_list_projects(client, seeded_project, seeded_project_with_samples):
    projects = client.list_projects()
    ids = [p.id for p in projects]
    assert seeded_project_with_samples[0] in ids
    assert seeded_project[0] in ids
    assert all(isinstance(p, Project) for p in projects)
    assert all(p._client is client for p in projects)


def test_list_projects_name_filter(client, seeded_project, seeded_project_with_samples):
    projects = client.list_projects(name="seeded_project_with_samples")
    ids = [p.id for p in projects]
    assert seeded_project_with_samples[0] in ids
    assert seeded_project[0] not in ids


def test_get_project(client, seeded_project):
    project = client.get_project(seeded_project[0])
    assert isinstance(project, Project)
    assert project.id == seeded_project[0]
    assert project._client is client


def test_get_project_missing_raises_not_found(client):
    with pytest.raises(NotFoundError) as exc_info:
        client.get_project(MISSING_OBJECT_ID)
    assert exc_info.value.status_code == 404


def test_get_project_by_name(client, seeded_project_with_samples):
    project_id, _ = seeded_project_with_samples
    project = client.get_project_by_name("seeded_project_with_samples")
    assert project.id == project_id
    assert project._client is client


def test_get_project_by_name_missing(client):
    with pytest.raises(NotFoundError) as exc_info:
        client.get_project_by_name("toktagger-does-not-exist-xyz")
    # Client-side lookup error: no HTTP status code
    assert exc_info.value.status_code is None


def test_get_project_by_name_multiple(
    client, seeded_project, seeded_project_with_samples
):
    with pytest.raises(MultipleResultsFoundError):
        client.get_project_by_name("seeded")


def test_list_samples(client, seeded_project_with_samples):
    project_id, sample_ids = seeded_project_with_samples
    samples = client.list_samples(project_id)
    assert sorted(s.id for s in samples) == sorted(sample_ids)
    assert all(isinstance(s, Sample) for s in samples)
    assert all(s._client is client for s in samples)


def test_list_samples_shot_id_filter(client, seeded_project_with_samples):
    project_id, _ = seeded_project_with_samples
    samples = client.list_samples(project_id, shot_id=10000)
    assert len(samples) == 1
    assert samples[0].shot_id == 10000


def test_get_sample(client, seeded_project_with_samples):
    project_id, sample_ids = seeded_project_with_samples
    sample = client.get_sample(project_id, sample_ids[0])
    assert isinstance(sample, Sample)
    assert sample.id == sample_ids[0]
    assert sample.project_id == project_id
    assert sample._client is client


def test_get_sample_by_shot_id(client, seeded_project_with_samples):
    project_id, _ = seeded_project_with_samples
    sample = client.get_sample_by_shot_id(project_id, 10001)
    assert sample.shot_id == 10001
    assert sample.project_id == project_id
    assert sample._client is client


def test_get_sample_by_shot_id_missing(client, seeded_project_with_samples):
    project_id, _ = seeded_project_with_samples
    with pytest.raises(NotFoundError) as exc_info:
        client.get_sample_by_shot_id(project_id, 99999)
    assert exc_info.value.status_code is None


def test_get_samples_summary(client, seeded_project_with_samples):
    project_id, _ = seeded_project_with_samples
    summary = client.get_samples_summary(project_id)
    assert isinstance(summary, SampleSummary)
    assert summary.total == 2
    assert summary.shot_min == 10000
    assert summary.shot_max == 10001
