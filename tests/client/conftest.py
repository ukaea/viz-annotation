# Integration tests for the TokTagger client: the real server (run_server from
# tests/conftest.py) in a subprocess on :8002, seeded through tests/endpoints.py
# like the end-to-end suite

import pathlib
import httpx

import pytest
import requests
import tests.endpoints as endpoints
from toktagger.client import TokTaggerClient
import tests.db_definitions as db_definitions

BASE_URL = "http://localhost:8002"


PROJECT_1_ID = "1" * 24
SAMPLE_1_ID = "2" * 24
PROJECT_2_ID = "3" * 24
SAMPLE_2_ID = "4" * 24


def make_client(handler, **kwargs) -> TokTaggerClient:
    return TokTaggerClient(transport=httpx.MockTransport(handler), **kwargs)


def raw_project(project_id: str, definition) -> dict:
    # Simulate the JSON the server returns for GET /projects/{id}
    raw = definition.model_dump(mode="json", by_alias=True)
    raw["_id"] = project_id
    return raw


def raw_sample(sample_id: str, project_id: str, definition) -> dict:
    # Simulate the JSON the server returns for GET /projects/{pid}/samples/{id}
    raw = definition.model_dump(mode="json", by_alias=True)
    raw["_id"] = sample_id
    raw["project_id"] = project_id
    raw["validated_annotations"] = False
    return raw


@pytest.fixture
def project_1() -> dict:
    return raw_project(PROJECT_1_ID, db_definitions.PROJECT_1)


@pytest.fixture
def sample_1() -> dict:
    return raw_sample(SAMPLE_1_ID, PROJECT_1_ID, db_definitions.SAMPLE_1)


@pytest.fixture
def project_2() -> dict:
    return raw_project(PROJECT_2_ID, db_definitions.PROJECT_2)


@pytest.fixture
def sample_2() -> dict:
    return raw_sample(SAMPLE_2_ID, PROJECT_1_ID, db_definitions.SAMPLE_2)


@pytest.fixture(scope="package")
def client(start_server):
    return TokTaggerClient(base_url=BASE_URL)


@pytest.fixture(scope="package", autouse=True)
def seeded_project(client):
    project_id = endpoints.create_project("seeded_project", "time-series", "tabular")
    yield project_id, []
    requests.delete(f"{BASE_URL}/projects/{project_id}")


@pytest.fixture(scope="package", autouse=True)
def seeded_project_with_samples(client):
    # A project with two samples so shot-id lookups and summaries are exercised.
    # The parquet files under tests/ back these local samples (see e2e suite).
    project_id = endpoints.create_project(
        "seeded_project_with_samples", "time-series", "tabular"
    )
    sample_ids = endpoints.create_local_samples(
        project_id, [10000, 10001], pathlib.Path(__file__).parents[2], ["Ip"]
    )
    yield project_id, sample_ids
    requests.delete(f"{BASE_URL}/projects/{project_id}")
