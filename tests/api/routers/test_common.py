"""Tests which are common to multiple endpoints which can be parametrized."""

import pytest
from bson.objectid import ObjectId
from tests.db_definitions import SAMPLE_1, ANNOTATION_1


def create_endpoint(
    endpoint: str, test_component: str, db_ids: dict, invalid_obj_id: bool
):
    """Create valid endpoint by inserting either valid or invalid IDs into their relevant positions."""
    # Split endpoint up
    components = endpoint.split("/")[1:]
    # Loop through each item in endpoint, check if it needs an ID replacing
    for i in range(0, len(components) - 1, 2):
        if components[i + 1] != "id":
            continue
        # If this is the component being tested, insert incorrect object ID
        if components[i] == test_component + "s":
            components[i + 1] = "invalid_id" if invalid_obj_id else str(ObjectId())
        # Otherwise, insert valid ID
        else:
            components[i + 1] = db_ids[components[i]]

    return "/".join(components)


async def make_request(api_client, request_method, endpoint, request_body):
    """Make the request using the method defined in the parametrization, passing in request body if post/put."""
    if request_method == "get":
        response = await api_client.get(endpoint)
    elif request_method == "put":
        response = await api_client.put(endpoint, json=request_body)
    elif request_method == "post":
        response = await api_client.post(endpoint, json=request_body)
    elif request_method == "delete":
        response = await api_client.delete(endpoint)
    else:
        response = None
    return response


# To add to this test, provide the endpoint you want to be testing
# Anywhere that you want an id inserted, put 'id'
# Then enter the type of object you want tested, eg 'project'
# The code will then figure out where to place correct and incorrect IDs
# Also define the method ('get', 'post', 'put', 'delete') to use,
# and if required provide a valid request body as a dict
@pytest.mark.parametrize(
    "endpoint, test_component, request_method, request_body",
    [
        ("/projects/id/annotations", "project", "get", {}),
        ("/projects/id/annotations", "project", "delete", {}),
        ("/projects/id/samples/id/annotations", "project", "get", {}),
        ("/projects/id/samples/id/annotations", "sample", "get", {}),
        ("/projects/id/samples/id/annotations", "project", "delete", {}),
        ("/projects/id/samples/id/annotations", "sample", "delete", {}),
        (
            "/projects/id/samples/id/annotations",
            "project",
            "put",
            [
                ANNOTATION_1.model_dump(mode="json"),
            ],
        ),
        (
            "/projects/id/samples/id/annotations",
            "sample",
            "put",
            [
                ANNOTATION_1.model_dump(mode="json"),
            ],
        ),
        ("/projects/id/samples/id/data", "project", "post", {}),
        ("/projects/id/samples/id/data", "sample", "post", {}),
        ("/projects/id", "project", "get", {}),
        ("/projects/id/samples", "project", "get", {}),
        ("/projects/id/samples/summary", "project", "get", {}),
        (
            "/projects/id/samples",
            "project",
            "post",
            [
                SAMPLE_1.model_dump(mode="json"),
            ],
        ),
        ("/projects/id/samples/next", "project", "get", {}),
        ("/projects/id/samples/id", "project", "get", {}),
        ("/projects/id/samples/id", "sample", "get", {}),
        ("/projects/id/samples/id", "project", "delete", {}),
        ("/projects/id/samples/id", "sample", "delete", {}),
    ],
    ids=[
        "get_annotations-wrong_project_id",
        "delete_annotations-wrong_project_id",
        "get_sample_annotations-wrong_project_id",
        "get_sample_annotations-wrong_sample_id",
        "delete_sample_annotations-wrong_project_id",
        "delete_sample_annotations-wrong_sample_id",
        "put_sample_annotations-wrong_project_id",
        "put_sample_annotations-wrong_sample_id",
        "get_data-wrong_project_id",
        "get_data-wrong_sample_id",
        "get_project-wrong_project_id",
        "get_samples-wrong_project_id",
        "get_samples_summary-wrong_project_id",
        "put_samples-wrong_project_id",
        "get_next_sample-wrong_project_id",
        "get_sample-wrong_project_id",
        "get_sample-wrong_sample_id",
        "delete_sample-wrong_project_id",
        "delete_sample-wrong_sample_id",
    ],
)
@pytest.mark.parametrize(
    "invalid_obj_id", (True, False), ids=["invalid_obj_id", "valid_obj_id"]
)
@pytest.mark.asyncio
async def test_invalid_id(
    api_client,
    setup_db_small,
    endpoint,
    test_component,
    request_method,
    request_body,
    invalid_obj_id,
):
    endpoint_with_ids = create_endpoint(
        endpoint, test_component, setup_db_small, invalid_obj_id
    )
    response = await make_request(
        api_client, request_method, endpoint_with_ids, request_body
    )
    if not response:
        pytest.fail("Test setup failed: invalid request method.")

    if invalid_obj_id:
        assert response.status_code == 400
        assert f"{test_component.title()} ID is not valid" in response.json().get(
            "detail"
        )
    else:
        assert response.status_code == 404
        assert f"{test_component.title()} not found" in response.json().get("detail")


@pytest.mark.models_disabled
@pytest.mark.parametrize(
    ("endpoint", "request_method"),
    [
        ["/projects/id/models", "get"],
        ["/projects/id/models/mock_disruption_cnn", "get"],
        ["/projects/id/models/mock_disruption_cnn", "delete"],
        ["/projects/id/models/mock_disruption_cnn/train", "get"],
        ["/projects/id/models/mock_disruption_cnn/train", "put"],
        ["/projects/id/models/mock_disruption_cnn/train", "delete"],
        ["/projects/id/models/mock_disruption_cnn/load/local", "post"],
        ["/projects/id/models/mock_disruption_cnn/load/gitlab", "post"],
        ["/projects/id/models/mock_disruption_cnn/load/hugging_face", "post"],
        ["/projects/id/models/mock_disruption_cnn/load/abc123", "get"],
        ["/projects/id/models/mock_disruption_cnn/predict", "post"],
        ["/projects/id/models/mock_disruption_cnn/predict", "delete"],
        ["/projects/id/models/abc123", "put"],
    ],
)
@pytest.mark.asyncio
async def test_model_endpoints_disabled(
    api_client, setup_db_small, endpoint, request_method
):
    endpoint_with_ids = endpoint.replace("id", setup_db_small["projects"])
    response = await make_request(api_client, request_method, endpoint_with_ids, {})

    # Should return 503 error since models endpoints not enabled
    assert response.status_code == 503
    data = response.json()
    assert (
        "ML model features are disabled (optional dependencies missing)"
        in data["detail"]
    )
