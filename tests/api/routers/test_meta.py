import pytest
from toktagger.api.schemas.samples import ShotData
import toktagger.api.config as config


@pytest.mark.asyncio
async def test_get_data_loaders(api_client, setup_db):
    response = await api_client.get("/meta/dataloader")
    assert response.status_code == 200
    data = response.json()
    assert all(item in data for item in ("uda", "image", "tabular", "sal", "fair_mast"))


@pytest.mark.asyncio
async def test_get_data_schema(api_client, setup_db):
    response = await api_client.get("/meta/dataloader/uda")
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "ShotData"
    assert data == ShotData.model_json_schema()


@pytest.mark.asyncio
@pytest.mark.models_enabled
@pytest.mark.parametrize("task", ["time-series", "video"])
async def test_get_model_types(api_client, setup_db, task):
    response = await api_client.get(f"/meta/models?task={task}")
    assert response.status_code == 200

    data = response.json()
    models_present = (
        item in data
        for item in [
            "mock_timeseries_cnn",
            "mock_params_timeseries_cnn",
            "mock_disruption_cnn",
        ]
    )
    if task == "time-series":
        assert all(models_present)
    else:
        assert not any(models_present)


@pytest.mark.asyncio
@pytest.mark.models_enabled
@pytest.mark.parametrize("local", [True, False])
async def test_get_model_load_methods(api_client, setup_db, local):
    if not local:
        config.settings.models.local_load_enabled = False
    response = await api_client.get("/meta/models/load")
    config.settings.models.local_load_enabled = True  # Restore default setting
    assert response.status_code == 200
    data = response.json()
    if local:
        assert data == ["local"]
    else:
        assert data == []


@pytest.mark.asyncio
@pytest.mark.models_enabled
@pytest.mark.parametrize(
    "model_name", ["mock_timeseries_cnn", "mock_params_timeseries_cnn"]
)
@pytest.mark.parametrize("method", ["train", "predict"])
async def test_get_model_schema(api_client, setup_db, model_name, method):
    response = await api_client.get(f"/meta/models/{model_name}/{method}")

    assert response.status_code == 200
    data = response.json()

    if model_name == "mock_timeseries_cnn":
        assert not data
    else:
        assert data["title"] == "TimeSeriesCNNParams"
        assert data["properties"]["final_score"]["type"] == "integer"
        assert data["properties"]["final_score"]["minimum"] == 50
        assert data["properties"]["final_score"]["exclusiveMaximum"] == 100
        assert data["properties"]["test_string"]["type"] == "string"
        assert data["properties"]["test_bool"]["type"] == "boolean"
        assert data["properties"]["test_bool"]["default"]  # == True
        assert data["properties"]["test_selection"]["enum"] == [
            "selection_1",
            "selection_2",
        ]


@pytest.mark.asyncio
@pytest.mark.models_disabled
@pytest.mark.parametrize("task", ["time-series", "video"])
async def test_get_model_types_disabled(api_client, setup_db, task):
    response = await api_client.get(f"/meta/models?task={task}")
    assert response.status_code == 503
    data = response.json()
    assert (
        "ML model features are disabled (optional dependencies missing)"
        in data["detail"]
    )


@pytest.mark.asyncio
@pytest.mark.models_disabled
@pytest.mark.parametrize("local", [True, False])
async def test_get_model_load_methods_disabled(api_client, setup_db, local):
    if not local:
        config.settings.models.local_load_enabled = False
    response = await api_client.get("/meta/models/load")
    config.settings.models.local_load_enabled = True
    assert response.status_code == 503
    data = response.json()
    assert (
        "ML model features are disabled (optional dependencies missing)"
        in data["detail"]
    )


@pytest.mark.asyncio
@pytest.mark.models_disabled
@pytest.mark.parametrize(
    "model_name", ["mock_timeseries_cnn", "mock_params_timeseries_cnn"]
)
@pytest.mark.parametrize("method", ["train", "predict"])
async def test_get_model_schema_disabled(api_client, setup_db, model_name, method):
    response = await api_client.get(f"/meta/models/{model_name}/{method}")

    assert response.status_code == 503
    data = response.json()
    assert (
        "ML model features are disabled (optional dependencies missing)"
        in data["detail"]
    )
