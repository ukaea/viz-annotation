import pathlib

import pytest
from bson import ObjectId

from toktagger.api.schemas.samples import ShotData, SampleIn, TimeSeriesFileData
import tests.db_definitions as db_definitions
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
@pytest.mark.parametrize("gitlab", [True, False])
@pytest.mark.parametrize("hugging_face", [True, False])
async def test_get_model_load_methods(
    api_client, setup_db, local, gitlab, hugging_face
):
    expected_methods = ["local", "gitlab", "hugging_face"]
    if not local:
        config.settings.models.local_load_enabled = False
        expected_methods.remove("local")
    if not gitlab:
        config.settings.models.gitlab_load_enabled = False
        expected_methods.remove("gitlab")
    if not hugging_face:
        config.settings.models.huggingface_load_enabled = False
        expected_methods.remove("hugging_face")

    response = await api_client.get("/meta/models/load")
    config.settings.models.local_load_enabled = True  # Restore default settings
    config.settings.models.huggingface_load_enabled = True
    config.settings.models.gitlab_load_enabled = True

    assert response.status_code == 200
    data = response.json()
    assert data == expected_methods


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
@pytest.mark.models_enabled
@pytest.mark.parametrize("model_name", ["minirocket", "shapelet_transform"])
async def test_get_model_train_schema_injects_class_label_enum(
    api_client, setup_db, model_name
):
    response = await api_client.get(
        f"/meta/models/{model_name}/train?project_id={setup_db['project_id_2']}"
    )
    assert response.status_code == 200
    data = response.json()
    assert (
        data["properties"]["class_label"]["enum"]
        == db_definitions.PROJECT_2.time_region_labels
    )


@pytest.mark.asyncio
@pytest.mark.models_enabled
@pytest.mark.parametrize("model_name", ["minirocket", "shapelet_transform"])
async def test_get_model_train_schema_without_project_id_has_no_enum(
    api_client, setup_db, model_name
):
    response = await api_client.get(f"/meta/models/{model_name}/train")
    assert response.status_code == 200
    data = response.json()
    assert "enum" not in data["properties"]["class_label"]


@pytest.mark.asyncio
@pytest.mark.models_enabled
@pytest.mark.parametrize("model_name", ["dtw_motif", "stumpy_motif"])
async def test_get_model_train_schema_injects_class_label_enum_with_blank_option(
    api_client, setup_db, model_name
):
    """class_label is optional (a filter) on template-matching models, so an
    unselected/blank value must remain a valid enum choice alongside the
    project's real labels."""
    response = await api_client.get(
        f"/meta/models/{model_name}/train?project_id={setup_db['project_id_2']}"
    )
    assert response.status_code == 200
    data = response.json()
    assert (
        data["properties"]["class_label"]["enum"]
        == [""] + db_definitions.PROJECT_2.time_region_labels
    )


@pytest.mark.asyncio
@pytest.mark.models_enabled
@pytest.mark.parametrize(
    "model_name", ["dtw_motif", "stumpy_motif", "minirocket", "shapelet_transform"]
)
async def test_get_model_train_schema_injects_signal_names_enum(
    api_client, setup_db, model_name
):
    """Signal names come from the project's samples, so the user picks from the
    signals which are really there instead of typing them out."""
    response = await api_client.get(
        f"/meta/models/{model_name}/train?project_id={setup_db['project_id_2']}"
    )
    assert response.status_code == 200
    data = response.json()
    assert (
        data["properties"]["signal_names"]["items"]["enum"]
        == db_definitions.SAMPLE_3.data.signal_names
    )


@pytest.mark.asyncio
@pytest.mark.models_enabled
@pytest.mark.parametrize(
    "model_name", ["dtw_motif", "stumpy_motif", "minirocket", "shapelet_transform"]
)
async def test_get_model_train_schema_without_project_id_has_no_signal_enum(
    api_client, setup_db, model_name
):
    response = await api_client.get(f"/meta/models/{model_name}/train")
    assert response.status_code == 200
    data = response.json()
    assert "enum" not in data["properties"]["signal_names"]["items"]


@pytest.mark.asyncio
@pytest.mark.models_enabled
@pytest.mark.parametrize("model_name", ["dtw_motif", "minirocket"])
async def test_get_model_train_schema_reads_signal_names_from_data(
    api_client, db_client, setup_db, model_name
):
    """File-based loaders treat signal names as an optional column filter, so a
    sample often does not name them. Read them from the data itself instead.

    The loader makes every column a signal and takes the time from the index, so
    a file with its own time column offers that column too, exactly as the plots
    for that sample do."""
    sample = SampleIn(
        shot_id=5,
        data=TimeSeriesFileData(
            file_name=str(
                pathlib.Path(__file__).parents[2].joinpath("10000.parquet").absolute()
            ),
            type="parquet",
            protocol="file",
            signal_names=None,
        ),
        annotations=None,
    )
    await db_client.insert(
        "samples",
        sample,
        ids={"project_id": ObjectId(setup_db["project_id_2"])},
    )

    response = await api_client.get(
        f"/meta/models/{model_name}/train?project_id={setup_db['project_id_2']}"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["properties"]["signal_names"]["items"]["enum"] == [
        "time",
        "Ip",
        "dalpha",
    ]


@pytest.mark.asyncio
@pytest.mark.models_enabled
@pytest.mark.parametrize("model_name", ["dtw_motif", "minirocket"])
async def test_get_model_train_schema_no_samples_has_no_signal_enum(
    api_client, setup_db, model_name
):
    """A project with no samples has no signals to offer, and an empty enum would
    leave the field impossible to fill, so it stays a free-text input."""
    response = await api_client.get(
        f"/meta/models/{model_name}/train?project_id={setup_db['project_id_3']}"
    )
    assert response.status_code == 200
    data = response.json()
    assert "enum" not in data["properties"]["signal_names"]["items"]


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
