import pytest
from toktagger.api.schemas.samples import SampleSummary
from toktagger.client.client import (
    Sample,
)
from toktagger.client.exceptions import (
    NotFoundError,
)


def test_list_samples(client, seeded_project_with_samples):
    project_id, sample_ids = seeded_project_with_samples
    project = client.get_project(project_id)
    samples = project.list_samples()
    assert sorted(s.id for s in samples) == sorted(sample_ids)
    assert all(isinstance(s, Sample) for s in samples)
    assert all(s._client is client for s in samples)


def test_list_samples_shot_id_filter(client, seeded_project_with_samples):
    project_id, _ = seeded_project_with_samples
    project = client.get_project(project_id)
    samples = project.list_samples(shot_id=10000)
    assert len(samples) == 1
    assert samples[0].shot_id == 10000


def test_get_sample(client, seeded_project_with_samples):
    project_id, sample_ids = seeded_project_with_samples
    project = client.get_project(project_id)
    sample = project.get_sample(sample_ids[0])
    assert isinstance(sample, Sample)
    assert sample.id == sample_ids[0]
    assert sample.project_id == project_id
    assert sample._client is client


def test_get_sample_by_shot_id(client, seeded_project_with_samples):
    project_id, _ = seeded_project_with_samples
    project = client.get_project(project_id)
    sample = project.get_sample_by_shot_id(10001)
    assert sample.shot_id == 10001
    assert sample.project_id == project_id
    assert sample._client is client


def test_get_sample_by_shot_id_missing(client, seeded_project_with_samples):
    project_id, _ = seeded_project_with_samples
    project = client.get_project(project_id)
    with pytest.raises(NotFoundError) as exc_info:
        project.get_sample_by_shot_id(99999)
    assert exc_info.value.status_code is None


def test_get_samples_summary(client, seeded_project_with_samples):
    project_id, _ = seeded_project_with_samples
    project = client.get_project(project_id)
    summary = project.get_samples_summary()
    assert isinstance(summary, SampleSummary)
    assert summary.total == 2
    assert summary.shot_min == 10000
    assert summary.shot_max == 10001
