import warnings

import pytest

from toktagger.api.main import Server


def test_openapi_schema_has_no_duplicate_operation_ids():
    """The SPA catch-all routes must be excluded from the OpenAPI schema.

    They are registered as single multi-method routes, so FastAPI would otherwise
    reuse one operation ID across every method and warn about duplicates when the
    schema is generated.
    """
    server = Server()
    server._setup_app()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        schema = server.app.openapi()

    duplicate_warnings = [
        str(w.message) for w in caught if "Duplicate Operation ID" in str(w.message)
    ]
    assert duplicate_warnings == []

    # Catch-all SPA routes are excluded from the schema; real operations remain.
    paths = schema["paths"]
    assert "/" not in paths
    assert "/{full_path}" not in paths
    assert "/health" in paths


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_health_models_enabled(models_api_client, setup_db):
    response = await models_api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "TokTagger"
    assert data.get("version")  # Won't check its contents here
    assert data.get("db_connected")
    assert data.get("models_enabled")
    assert data.get("gpu_available")  # Forced to be 1 GPUs in conftest setup


@pytest.mark.asyncio
@pytest.mark.models_disabled
async def test_health_models_disabled(api_client, setup_db):
    response = await api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "TokTagger"
    assert data.get("version")  # Won't check its contents here
    assert data.get("db_connected")
    assert data.get("models_enabled") is False
    assert data.get("gpu_available") is False
