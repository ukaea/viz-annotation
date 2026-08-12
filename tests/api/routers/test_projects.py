import pytest

from toktagger.api.schemas.projects import Task


@pytest.mark.asyncio
async def test_get_all_projects(api_client, setup_db):
    response = await api_client.get("/projects?sort_direction=ascending")
    assert response.status_code == 200
    returned_projects = response.json()
    assert [project["name"] for project in returned_projects] == [
        "test_project_0",
        "test_project_1",
        "project_2",
    ]
    assert [project["_id"] for project in returned_projects] == [
        setup_db["project_id_1"],
        setup_db["project_id_2"],
        setup_db["project_id_3"],
    ]


@pytest.mark.asyncio
async def test_update_project(api_client, setup_db):
    response = await api_client.get(f"/projects/{setup_db['project_id_1']}")
    assert response.status_code == 200

    project = response.json()
    project["name"] = "updated_project_name"

    response = await api_client.put(
        f"/projects/{setup_db['project_id_1']}", json=project
    )
    assert response.status_code == 200

    result = await api_client.get(f"/projects/{setup_db['project_id_1']}")
    assert result.status_code == 200
    assert result.json().get("name") == "updated_project_name"


@pytest.mark.asyncio
async def test_get_all_projects_sortby(api_client, setup_db):
    response = await api_client.get("/projects?sort_by=task")
    # Should sort alphabetically by task
    # Sorts by case first (all uppers before any lowers)
    # So time series (project 1), then video (project 2), then profile2d (project 0)
    # Default sort direction is descending, so will return the opposite of this: 0, 2, 1
    assert response.status_code == 200
    returned_projects = response.json()
    assert [project["name"] for project in returned_projects] == [
        "project_2",
        "test_project_1",
        "test_project_0",
    ]


@pytest.mark.asyncio
async def test_get_projects_start(api_client, setup_db):
    response = await api_client.get("/projects?sort_direction=ascending&start=1")
    # Should return 2 projects
    assert response.status_code == 200
    returned_projects = response.json()
    assert len(returned_projects) == 2
    assert [project["name"] for project in returned_projects] == [
        "test_project_1",
        "project_2",
    ]


@pytest.mark.asyncio
async def test_get_projects_count(api_client, setup_db):
    response = await api_client.get("/projects?sort_direction=ascending&count=2")
    # Should return 2 projects
    assert response.status_code == 200
    returned_projects = response.json()
    assert len(returned_projects) == 2
    assert [project["name"] for project in returned_projects] == [
        "test_project_0",
        "test_project_1",
    ]


@pytest.mark.asyncio
async def test_get_projects_start_count(api_client, setup_db):
    response = await api_client.get(
        "/projects?sort_direction=ascending&start=1&count=1"
    )
    # Should return 1 project
    assert response.status_code == 200
    returned_projects = response.json()
    assert len(returned_projects) == 1
    assert [project["name"] for project in returned_projects] == ["test_project_1"]


@pytest.mark.asyncio
async def test_get_projects_invalid_start(api_client, setup_db):
    response = await api_client.get("/projects?start=10")
    # Should return 0 projects
    assert response.status_code == 200
    returned_projects = response.json()
    assert len(returned_projects) == 0


@pytest.mark.asyncio
async def test_get_projects_name(api_client, setup_db):
    response = await api_client.get("/projects?name=test_project_1")
    # Should return 1 project
    assert response.status_code == 200
    returned_projects = response.json()
    assert len(returned_projects) == 1
    assert returned_projects[0]["name"] == "test_project_1"


@pytest.mark.asyncio
async def test_get_project_id(api_client, setup_db):
    response = await api_client.get(f"/projects/{setup_db['project_id_1']}")
    assert response.status_code == 200
    returned_project = response.json()
    # Check info matches what we created the entry with
    assert returned_project.get("name") == "test_project_0"
    assert returned_project.get("task") == Task.PROFILE_2D
    assert returned_project.get("query_strategy") == "sequential"
    assert returned_project.get("data_loader") == "uda"

    # Then also check ID and timestamp are returned - should have been added automatically
    assert returned_project.get("_id") == setup_db["project_id_1"]
    assert returned_project.get("timestamp")


@pytest.mark.asyncio
async def test_delete_project(api_client, setup_db, db_client):
    response = await api_client.delete(f"/projects/{setup_db['project_id_2']}")
    assert response.status_code == 200

    # Check there are two projects left in the database
    projects = await db_client.get_all_documents("projects")
    assert len(projects) == 2
    # Check project with above ID no longer in database
    assert setup_db["project_id_2"] not in [project.get("_id") for project in projects]

    # Check samples associated with this project have been deleted
    samples = await db_client.get_all_documents("samples")
    assert len(samples) == 2  # Samples associated with project 1 still exist
    # Check sample associated with above project no longer in database
    assert setup_db["sample_id_4"] not in [sample.get("_id") for sample in samples]

    # Check annotations associated with this project have been deleted
    annotations = await db_client.get_all_documents("annotations")
    assert len(annotations) == 4  # Annotations associated with project 1 still exist
    # Check annotation associated with above project no longer in database
    assert setup_db["annotation_id_4"] not in [
        annotation.get("_id") for annotation in annotations
    ]


@pytest.mark.asyncio
async def test_create_project(api_client, db_client):
    in_project = {
        "name": "test_project",
        "task": Task.VIDEO,
        "query_strategy": "random",
        "data_loader": "image",
    }
    response = await api_client.post("/projects", json=in_project)
    assert response.status_code == 200
    assert (_id := response.json().get("_id"))

    # Check it has been added to database
    projects = await db_client.get_all_documents("projects")
    assert len(projects) == 1

    for key, value in in_project.items():
        assert projects[0][key] == value

    assert str(projects[0]["_id"]) == _id
    assert projects[0].get("timestamp")


@pytest.mark.asyncio
async def test_create_project_invalid(api_client, db_client):
    in_project = {
        "name": "test_project",
        "task": "invalid_task",
        "data_loader": "files",
        # missing: query_strategy
    }
    response = await api_client.post("/projects", json=in_project)
    assert response.status_code == 422
    errors = response.json().get("detail", [])
    # Should flag that task and data_loader are invalid options, and query_strategy is missing...
    assert len(errors) == 3

    # Check it has not been added to database
    projects = await db_client.get_all_documents("projects")
    assert len(projects) == 0
