import pytest


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
