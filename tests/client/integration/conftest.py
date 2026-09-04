# Integration tests for the TokTagger client: the real server (run_server from
# tests/conftest.py) in a subprocess on :8002, seeded through tests/endpoints.py
# like the end-to-end suite

import uuid

import pytest
import requests
import tests.endpoints as endpoints
from toktagger.client import TokTaggerClient

BASE_URL = "http://localhost:8002"


@pytest.fixture(scope="package")
def client(start_server):
    return TokTaggerClient(base_url=BASE_URL)


@pytest.fixture(scope="package")
def seeded_project():
    project_id = endpoints.create_project(
        f"toktagger-client-test-{uuid.uuid4().hex[:8]}", "time-series", "tabular"
    )
    yield project_id
    requests.delete(f"{BASE_URL}/projects/{project_id}")
