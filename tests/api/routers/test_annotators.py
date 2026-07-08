from toktagger.api.schemas.annotators import AnnotatorTypes
import pytest


@pytest.mark.asyncio
async def test_annotators_invalid_type(api_client, setup_db):
    params = {
        "annotator_params": {"signal_name": "Ip", "prominence": 10, "distance": 5},
        "data_params": {"name": "identity"},
    }
    response = await api_client.post(
        f"/projects/{setup_db['project_id_1']}/samples/{setup_db['sample_id_2']}/annotator/wrong_type",
        json=params,
    )
    # Should throw a validation error since annotator of this type doesnt exist
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_annotators(api_client, setup_db):
    params = {
        "annotator_params": {"signal_name": "Ip", "prominence": 1, "distance": 1},
        "data_params": {"name": "identity"},
    }
    response = await api_client.post(
        f"/projects/{setup_db['project_id_2']}/samples/{setup_db['sample_id_4']}/annotator/peak_detection",
        json=params,
    )

    returned_annotations = response.json()
    # Check it found 4 ELMs since we added 4 spikes to the data at 20, 40, 60, 80
    assert len(returned_annotations) == 4
    for annotation in returned_annotations:
        assert annotation["label"] == "Unknown"
        assert (
            annotation["created_by"]
            == f"annotators::{AnnotatorTypes.PEAK_DETECTION.value}"
        )
        assert "time_min" in annotation
        assert "time_max" in annotation


@pytest.mark.asyncio
async def test_annotators_not_supported_for_project(api_client, setup_db):
    params = {
        "annotator_params": {"signal_name": "Ip", "prominence": 10, "distance": 5},
        "data_params": {"name": "identity"},
    }
    response = await api_client.post(
        f"/projects/{setup_db['project_id_3']}/samples/{setup_db['sample_id_3']}/annotator/peak_detection",
        json=params,
    )
    # Should throw a 409 conflict error since Disruption project does not support Find Peaks annotator
    assert response.status_code == 409
    assert "The selected annotator cannot be used" in response.json().get("detail")
