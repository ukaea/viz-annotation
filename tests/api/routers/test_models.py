import pytest

pytest.importorskip("ray")

import pathlib
from toktagger.api.schemas.models import ModelUpdate
from toktagger.api.models.base import ActorRegistry
from toktagger.api.core.sender import (
    send_batch_samples,
    send_batch_annotations,
    send_model_updates,
)
import ray
from unittest.mock import patch
from bson import ObjectId
import tempfile
import asyncio
import toktagger.api.config as config


def wait_for_results(task_registry: ActorRegistry, task_id: str):
    task = task_registry.get(task_id)
    results = ray.get(task, timeout=30)
    return results


async def collect_predict_results(models_api_client, task_id):
    results = wait_for_results(models_api_client.app.state.task_registry, task_id)
    with patch("requests.put", models_api_client.put):
        response = await send_batch_annotations(
            results["project_id"], results["annotations_batch"]
        )
        assert response.status_code == 200
        await send_batch_samples(results["project_id"], results["samples_batch"])
        assert response.status_code == 200


async def collect_train_results(models_api_client, task_id):
    results = wait_for_results(models_api_client.app.state.task_registry, task_id)
    update = ModelUpdate(
        training_status="completed", progress=100, score=results["score"]
    )
    with patch("requests.put", models_api_client.put):
        response = await send_model_updates(
            results["project_id"], results["model_id"], update
        )
        assert response.status_code == 200


KILL_COUNT = 0


def kill(*args, **kwargs):
    print("killed")


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_batch_predict_num_predictions(
    models_api_client, db_client, setup_model_db
):
    response = await models_api_client.post(
        f"/projects/{setup_model_db['project_id']}/models/mock_disruption_cnn/predict?num_predictions=5"
    )

    await collect_predict_results(models_api_client, response.json()["task_id"])

    # Get annotations from the database, there should be 5 non validated ones
    annotations = await db_client.get_filtered_documents(
        collection="annotations", filters={"validated": False}
    )

    assert len(annotations) == 5

    # Check latest version of model has been used by default (annotation label set to model ID in Mock)
    assert all(ann["label"] == setup_model_db["model_id_2"] for ann in annotations)


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_batch_predict_num_predictions_params(
    models_api_client, db_client, setup_model_db
):
    response = await models_api_client.post(
        f"/projects/{setup_model_db['project_id']}/models/mock_params_timeseries_cnn/predict?num_predictions=5",
        json={
            "params": {
                "final_score": 50,
                "test_string": "testing",
                "test_bool": True,
                "test_selection": "selection_1",
            }
        },
    )

    await collect_predict_results(models_api_client, response.json()["task_id"])

    # Get annotations from the database, there should be 5 x 3 non validated ones
    annotations = await db_client.get_filtered_documents(
        collection="annotations", filters={"validated": False}
    )

    assert len(annotations) == 15

    # Check disruption annotations have used params correctly to give time = params.final_score + 1
    assert all(ann["time"] == 51 for ann in annotations if ann["label"] == "disruption")


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_batch_predict_samples(
    models_api_client, db_client, setup_model_db
):
    query_string = "&".join(
        f"sample_ids={id}" for id in setup_model_db["sample_ids"][:2]
    )
    response = await models_api_client.post(
        f"/projects/{setup_model_db['project_id']}/models/mock_disruption_cnn/predict?{query_string}"
    )

    await collect_predict_results(models_api_client, response.json()["task_id"])

    # Get annotations from the database, there should be 2 non validated ones
    annotations = await db_client.get_filtered_documents(
        collection="annotations", filters={"validated": False}
    )
    assert len(annotations) == 2

    # Check annotations assocaited with correct samples were updated
    assert sorted(
        [str(annotation["sample_id"]) for annotation in annotations]
    ) == sorted(setup_model_db["sample_ids"][:2])

    # Check latest version of model has been used by default (annotation label set to model ID in Mock)
    assert all(ann["label"] == setup_model_db["model_id_2"] for ann in annotations)


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_batch_predict_version(
    models_api_client, db_client, setup_model_db
):
    response = await models_api_client.post(
        f"/projects/{setup_model_db['project_id']}/models/mock_disruption_cnn/predict?num_predictions=5&version=1"
    )

    await collect_predict_results(models_api_client, response.json()["task_id"])

    # Get annotations from the database, there should be 5 non validated ones
    annotations = await db_client.get_filtered_documents(
        collection="annotations", filters={"validated": False}
    )
    assert len(annotations) == 5

    # Check version 1 of model has been used (annotation label set to model ID in Mock)
    assert all(ann["label"] == setup_model_db["model_id_1"] for ann in annotations)


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_predict_missing_weights(
    models_api_client, db_client, setup_model_db
):
    # Delete weights
    config.settings.models.cache_dir.joinpath(
        f"{setup_model_db['model_id_1']}.model"
    ).unlink()
    response = await models_api_client.post(
        f"/projects/{setup_model_db['project_id']}/models/mock_disruption_cnn/predict?num_predictions=5&version=1"
    )
    with pytest.raises(RuntimeError) as e:
        wait_for_results(
            models_api_client.app.state.task_registry, response.json()["task_id"]
        )
        assert "Cannot make predictions using an untrained model!" in str(e)


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_sample_predict_params(
    models_api_client, db_client, setup_model_db
):
    response = await models_api_client.post(
        f"/projects/{setup_model_db['project_id']}/samples/{setup_model_db['sample_ids'][-1]}/models/mock_params_timeseries_cnn/predict",
        json={
            "params": {
                "final_score": 50,
                "test_string": "testing",
                "test_bool": True,
                "test_selection": "selection_1",
            }
        },
    )

    await collect_predict_results(models_api_client, response.json()["task_id"])

    # Get annotations from the database, there should be 3 non validated one
    annotations = await db_client.get_filtered_documents(
        collection="annotations", filters={"validated": False}
    )
    assert len(annotations) == 3

    # Check it corresponds to sample ID we asked for predictions on
    assert str(annotations[0]["sample_id"]) == setup_model_db["sample_ids"][-1]

    # Check disruption has time corresponding to final_score in params + 1
    ann = next(ann for ann in annotations if ann["label"] == "Disruption")
    assert ann["time"] == 51


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_sample_predict(models_api_client, db_client, setup_model_db):
    response = await models_api_client.post(
        f"/projects/{setup_model_db['project_id']}/samples/{setup_model_db['sample_ids'][-1]}/models/mock_disruption_cnn/predict"
    )

    await collect_predict_results(models_api_client, response.json()["task_id"])

    # Get annotations from the database, there should be 1 non validated one
    annotations = await db_client.get_filtered_documents(
        collection="annotations", filters={"validated": False}
    )
    assert len(annotations) == 1

    # Check latest version of model has been used (annotation label set to model ID in Mock)
    assert annotations[0]["label"] == setup_model_db["model_id_2"]

    # Check it corresponds to sample ID we asked for predictions on
    assert str(annotations[0]["sample_id"]) == setup_model_db["sample_ids"][-1]


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_get_sample_prediction(
    models_api_client, db_client, setup_model_db
):
    response = await models_api_client.post(
        f"/projects/{setup_model_db['project_id']}/samples/{setup_model_db['sample_ids'][-1]}/models/mock_disruption_cnn/predict"
    )

    task_id = response.json()["task_id"]

    # Poll the endpoint until results arrive
    t = 0
    while t < 30:
        get_response = await models_api_client.get(
            f"/projects/{setup_model_db['project_id']}/samples/{setup_model_db['sample_ids'][-1]}/models/mock_disruption_cnn/predict/{task_id}"
        )
        if get_response.status_code == 200:
            break
        elif get_response.status_code != 202:
            raise ValueError(f"Got response {get_response.status_code}!")
        await asyncio.sleep(1)
        t += 1

    assert get_response.status_code == 200

    annotation = get_response.json()[0]

    # Check it corresponds to sample ID we asked for predictions on
    assert str(annotation["sample_id"]) == setup_model_db["sample_ids"][-1]


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_get_sample_prediction_invalid_task(
    models_api_client, db_client, setup_model_db
):
    # Ask for predictions from a task which doesn't exist
    get_response = await models_api_client.get(
        f"/projects/{setup_model_db['project_id']}/samples/{setup_model_db['sample_ids'][-1]}/models/mock_disruption_cnn/predict/invalid_id"
    )

    # Check it returns 404 with appropriate message
    assert get_response.status_code == 404
    assert "Predict task not found with that ID!" in get_response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_get_sample_prediction_wrong_sample(
    models_api_client, db_client, setup_model_db
):
    response = await models_api_client.post(
        f"/projects/{setup_model_db['project_id']}/samples/{setup_model_db['sample_ids'][-1]}/models/mock_disruption_cnn/predict"
    )

    task_id = response.json()["task_id"]

    # Ask for predictions from this task for a sample which we did not predict on
    t = 0
    while t < 30:
        get_response = await models_api_client.get(
            f"/projects/{setup_model_db['project_id']}/samples/{setup_model_db['sample_ids'][-2]}/models/mock_disruption_cnn/predict/{task_id}"
        )
        if get_response.status_code == 202:
            await asyncio.sleep(1)
            t += 1
            continue
        else:
            break

    # Check it returned 404 with appropriate message
    assert get_response.status_code == 404
    assert (
        "This task does not have results for the specified sample!"
        in get_response.json()["detail"]
    )


def mock_wait(*args, **kwargs):
    return [], ["waiting"]


@pytest.mark.asyncio
@pytest.mark.models_enabled
@patch("ray.wait", mock_wait)
async def test_model_get_sample_prediction_in_progress(
    models_api_client, db_client, setup_model_db
):
    response = await models_api_client.post(
        f"/projects/{setup_model_db['project_id']}/samples/{setup_model_db['sample_ids'][-1]}/models/mock_disruption_cnn/predict"
    )

    task_id = response.json()["task_id"]

    # Ask for predictions from this task for a sample while it is still in progress
    get_response = await models_api_client.get(
        f"/projects/{setup_model_db['project_id']}/samples/{setup_model_db['sample_ids'][-1]}/models/mock_disruption_cnn/predict/{task_id}"
    )

    # Check it returns 202 with appropriate message
    assert get_response.status_code == 202
    assert "Predict task in the queue!" in get_response.json()["message"]


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_update(models_api_client, db_client, setup_model_db):
    model_updates = ModelUpdate(training_status="started", progress=50, score=20)
    response = await models_api_client.put(
        f"/projects/{setup_model_db['project_id']}/models/{setup_model_db['model_id_1']}",
        json=model_updates.model_dump(mode="json"),
    )
    assert response.status_code == 200

    # Get model from the database
    model = await db_client.get_document_by_id(
        collection="models", object_id=ObjectId(setup_model_db["model_id_1"])
    )
    assert model["training_status"] == "started"
    assert model["progress"] == 50
    assert model["score"] == 20


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_start_training_no_params(
    models_api_client, db_client, setup_model_db
):
    response = await models_api_client.put(
        f"/projects/{setup_model_db['project_id']}/models/mock_disruption_cnn/train"
    )

    await collect_train_results(models_api_client, response.json()["task_id"])
    model_id = response.json()["model_id"]

    model = await db_client.get_document_by_id(
        collection="models", object_id=ObjectId(model_id)
    )

    # Check model has been set to completed, with 100% completion
    assert model["training_status"] == "completed"
    assert model["progress"] == 100
    assert model["score"] == 60  # value returned by train method

    # Check model has been saved after completion
    assert config.settings.models.cache_dir.joinpath(f"{model_id}.model").exists()


@pytest.mark.asyncio
@pytest.mark.models_enabled
@pytest.mark.parametrize("method", ["train", "predict", "sample"])
async def test_model_wrong_params(models_api_client, db_client, setup_model_db, method):
    if method == "sample":
        url = f"/projects/{setup_model_db['project_id']}/samples/{setup_model_db['sample_ids'][-2]}/models/mock_params_timeseries_cnn/predict"
    else:
        url = f"/projects/{setup_model_db['project_id']}/models/mock_params_timeseries_cnn/{method}"

    json = {
        "params": {
            "final_score": 20,
            "test_bool": 5,
            "test_selection": "wrong_selection",
        }
    }

    if method == "train":
        response = await models_api_client.put(url, json=json)
    else:
        response = await models_api_client.post(url, json=json)

    assert response.status_code == 422
    error = response.json()["detail"]
    assert "'final_score': Input should be greater than or equal to 50" in error
    assert "'test_string': Field required" in error
    assert "'test_bool': Input should be a valid boolean" in error
    assert "'test_selection': Input should be 'selection_1' or 'selection_2'" in error


@pytest.mark.asyncio
@pytest.mark.models_enabled
@pytest.mark.parametrize("method", ["train", "predict", "sample"])
async def test_model_missing_params(
    models_api_client, db_client, setup_model_db, method
):
    if method == "sample":
        url = f"/projects/{setup_model_db['project_id']}/samples/{setup_model_db['sample_ids'][-2]}/models/mock_params_timeseries_cnn/predict"
    else:
        url = f"/projects/{setup_model_db['project_id']}/models/mock_params_timeseries_cnn/{method}"

    if method == "train":
        response = await models_api_client.put(url)
    else:
        response = await models_api_client.post(url)

    assert response.status_code == 422
    assert (
        "Model training parameters are missing! Requires 'TimeSeriesCNNParams' parameters."
        in response.json()["detail"]
    )


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_start_training_params(
    models_api_client, db_client, setup_model_db
):
    response = await models_api_client.put(
        f"/projects/{setup_model_db['project_id']}/models/mock_params_timeseries_cnn/train",
        json={
            "params": {
                "final_score": 50,
                "test_string": "testing",
                "test_bool": True,
                "test_selection": "selection_1",
            }
        },
    )

    await collect_train_results(models_api_client, response.json()["task_id"])
    model_id = response.json()["model_id"]

    model = await db_client.get_document_by_id(
        collection="models", object_id=ObjectId(model_id)
    )

    # Check model has been set to completed, with 100% completion
    assert model["training_status"] == "completed"
    assert model["progress"] == 100
    assert model["score"] == 50  # value returned from params

    # Check model has been saved after completion
    assert config.settings.models.cache_dir.joinpath(f"{model_id}.model").exists()


# Test delete model
@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_delete_type(models_api_client, db_client, setup_model_db):
    response = await models_api_client.delete(
        f"/projects/{setup_model_db['project_id']}/models/mock_disruption_cnn"
    )
    assert response.status_code == 200

    # Check there are two model left in the database
    models = await db_client.get_all_documents("models")
    assert len(models) == 2

    # Check they are not of type 'mock_disruption_cnn'
    assert all(model["type"] != "mock_disruption_cnn" for model in models)

    # Check for models 1 and 2, their file no longer exists
    assert not config.settings.models.cache_dir.joinpath(
        f"{setup_model_db['model_id_1']}.model"
    ).exists()
    assert not config.settings.models.cache_dir.joinpath(
        f"{setup_model_db['model_id_2']}.model"
    ).exists()
    # And for model 3 it does still exist
    assert config.settings.models.cache_dir.joinpath(
        f"{setup_model_db['model_id_3']}.model"
    ).exists()
    assert config.settings.models.cache_dir.joinpath(
        f"{setup_model_db['model_id_3']}.model"
    ).exists()


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_delete_type_version(models_api_client, db_client, setup_model_db):
    response = await models_api_client.delete(
        f"/projects/{setup_model_db['project_id']}/models/mock_disruption_cnn?version=2"
    )
    assert response.status_code == 200

    # Check there are three models left in the database
    models = await db_client.get_all_documents("models")
    assert len(models) == 3
    # Check model version 1 of mock_disruption_cnn still exists
    # Check the others are not of type 'mock_disruption_cnn'
    assert all(
        (model["type"] == "mock_disruption_cnn" and model["version"] == 1)
        or (model["type"] != "mock_disruption_cnn")
        for model in models
    )

    # Check for model 2, their file no longer exists
    assert not config.settings.models.cache_dir.joinpath(
        f"{setup_model_db['model_id_2']}.model"
    ).exists()
    # And for models 1, 3 and 4 it does still exist
    assert all(
        (
            config.settings.models.cache_dir.joinpath(
                f"{setup_model_db[model_id]}.model"
            ).exists()
        )
        for model_id in ("model_id_1", "model_id_3", "model_id_4")
    )


@pytest.mark.asyncio
@pytest.mark.models_enabled
@patch("ray.cancel")
async def test_model_stop_training(
    mock_func, models_api_client, db_client, setup_model_db
):
    response = await models_api_client.delete(
        f"/projects/{setup_model_db['project_id']}/models/disruption_cnn/train"
    )
    assert response.status_code == 200
    assert len(response.json()) == 1
    deleted_id = response.json()[0]
    assert deleted_id == setup_model_db["model_id_3"]

    # Check it is aborted in database
    models = await db_client.get_filtered_documents(
        "models", {"type": "disruption_cnn"}
    )
    model = models[0]
    assert model["training_status"] == "aborted"

    assert mock_func.call_count > 0


@pytest.mark.asyncio
@pytest.mark.models_enabled
@patch("ray.kill")
async def test_model_stop_training_not_in_progress(
    mock_func, models_api_client, db_client, setup_model_db
):
    response = await models_api_client.delete(
        f"/projects/{setup_model_db['project_id']}/models/mock_disruption_cnn/train?version=1"
    )  # Version 1 model is 'completed' so nothing to do
    assert response.status_code == 409
    assert (
        response.json()["detail"] == "Model training is not in progress for this model!"
    )

    # Check no models show as aborted
    models = await db_client.get_all_documents("models")
    assert all(model["training_status"] != "aborted" for model in models)

    assert mock_func.call_count == 0


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_delete_predictions(models_api_client, db_client, setup_model_db):
    await models_api_client.delete(
        f"/projects/{setup_model_db['project_id']}/models/disruption_cnn/predict"
    )

    # Should be 5 annotations remaining since half were created by 'manual'
    annotations = await db_client.get_all_documents(collection="annotations")
    assert len(annotations) == 5
    assert all(annotation["created_by"] == "manual" for annotation in annotations)


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_delete_no_predictions(
    models_api_client, db_client, setup_model_db
):
    response = await models_api_client.delete(
        f"/projects/{setup_model_db['project_id']}/models/mock_disruption_cnn/predict"
    )

    # Nothing created by this model, so should return 404 and not delete anything
    assert response.status_code == 404
    assert (
        response.json()["detail"]
        == "No annotations produced by mock_disruption_cnn could be found for this Project."
    )
    annotations = await db_client.get_all_documents(collection="annotations")
    assert len(annotations) == 10


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_load_local(models_api_client, db_client, setup_model_db):
    # Create tempfile
    with tempfile.NamedTemporaryFile(suffix=".model", mode="w") as tempf:
        tempf.write("Model Weights")
        tempf.flush()
        response = await models_api_client.post(
            f"/projects/{setup_model_db['project_id']}/models/mock_disruption_cnn/load?method=local&weights_path={str(tempf.name)}"
        )
        assert response.status_code == 200
        model_id = response.json()["model_id"]
        task_id = response.json()["task_id"]

        t = 0
        while t < 30:
            response = await models_api_client.get(
                f"/projects/{setup_model_db['project_id']}/models/mock_disruption_cnn/load/{task_id}"
            )
            if response.status_code == 202:
                await asyncio.sleep(1)
                t += 1
                continue
            elif response.status_code == 200:
                break
            else:
                raise AssertionError(f"Response code {response.status_code} invalid")

        assert response.status_code == 200

        model = await db_client.get_document_by_id(
            collection="models", object_id=ObjectId(model_id)
        )

        # Check model has been set to completed, with 100% completion
        assert model["training_status"] == "completed"
        assert model["progress"] == 100

        # Check model has been saved after completion
        model_path = pathlib.Path(config.settings.models.cache_dir).joinpath(
            f"{model_id}.model"
        )
        assert model_path.exists()

        # Open the file, check contents are there
        assert model_path.read_text() == "Model Weights"


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_load_local_missing_file(
    models_api_client, db_client, setup_model_db
):
    # Try loading nonexistent file
    response = await models_api_client.post(
        f"/projects/{setup_model_db['project_id']}/models/mock_disruption_cnn/load?method=local&weights_path=non_existant_path.model"
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Weights file not found at specified path!"


@pytest.mark.asyncio
async def test_model_load_local_disabled(models_api_client, db_client, setup_model_db):
    # Try loading  file with local load disabled
    config.settings.models.local_load_enabled = False
    with tempfile.NamedTemporaryFile(suffix=".model", mode="w") as tempf:
        tempf.write("Model Weights")
        tempf.flush()
        response = await models_api_client.post(
            f"/projects/{setup_model_db['project_id']}/models/mock_disruption_cnn/load?method=local&weights_path={str(tempf.name)}"
        )
    config.settings.models.local_load_enabled = True
    assert response.status_code == 403
    assert response.json()["detail"] == "Loading from local weights is disabled."


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_model_load_local_failed(models_api_client, db_client, setup_model_db):
    # Create tempfile
    with tempfile.NamedTemporaryFile(suffix=".model") as tempf:
        # Write invalid bytes which will fail to be read by the model
        tempf.write(b"\xff")
        tempf.flush()
        response = await models_api_client.post(
            f"/projects/{setup_model_db['project_id']}/models/mock_disruption_cnn/load?method=local&weights_path={str(tempf.name)}"
        )
        # Successfully submits the load job
        assert response.status_code == 200

        model_id = response.json()["model_id"]
        task_id = response.json()["task_id"]
        # Get results from endpoint polling
        t = 0
        while t < 30:
            response = await models_api_client.get(
                f"/projects/{setup_model_db['project_id']}/models/mock_disruption_cnn/load/{task_id}"
            )
            if response.status_code == 202:
                await asyncio.sleep(1)
                t += 1
                continue
            elif response.status_code == 500:
                break
            else:
                raise AssertionError(f"Response code {response.status_code} invalid")
        assert response.status_code == 500

        # Check message is as expected
        print(response.json())
        assert "Load task failed - Failed to load weights" in response.json()["detail"]

        model = await db_client.get_document_by_id(
            collection="models", object_id=ObjectId(model_id)
        )

        # Check model has been set to failed
        assert model["training_status"] == "failed"

        # Check model has not been saved after completion
        assert (
            not pathlib.Path(config.settings.models.cache_dir)
            .joinpath(f"{model_id}.model")
            .exists()
        )
