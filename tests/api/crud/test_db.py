import asyncio

import pytest
import pytest_asyncio
from bson.objectid import ObjectId

from tests.db_definitions import PROJECT_1, PROJECT_2, SAMPLE_1, SAMPLE_2
from toktagger.api.crud import utils
from toktagger.api.crud.db import MongoDBClient


@pytest_asyncio.fixture(scope="function")
async def setup_db_for_filtering(db_client):
    await db_client.db["projects"].insert_one({"idx": 1, "number": 10, "local": True})
    await asyncio.sleep(0.1)
    await db_client.db["projects"].insert_one({"idx": 2, "number": 30, "local": False})
    await asyncio.sleep(0.1)
    await db_client.db["projects"].insert_one({"idx": 3, "number": 20, "local": True})


@pytest.mark.asyncio
async def test_insert(db_client):
    project_id = await db_client.insert(collection="projects", model=PROJECT_1)

    # Check project has been inserted
    retrieved_project = await db_client.db["projects"].find_one(
        {"_id": ObjectId(project_id)}
    )
    assert retrieved_project["name"] == PROJECT_1.name


@pytest.mark.asyncio
async def test_insert_ids(db_client):
    project_id = await db_client.insert(collection="projects", model=PROJECT_1)
    sample_id = await db_client.insert(
        collection="samples", model=SAMPLE_1, ids={"project_id": ObjectId(project_id)}
    )

    # Check sample has been inserted with project ID attached
    retrieved_sample = await db_client.db["samples"].find_one(
        {"_id": ObjectId(sample_id)}
    )

    assert retrieved_sample["shot_id"] == SAMPLE_1.shot_id
    assert retrieved_sample["project_id"] == ObjectId(project_id)


@pytest.mark.asyncio
async def test_insert_many(db_client):
    project_ids = await db_client.insert_many(
        collection="projects", models=[PROJECT_1, PROJECT_2]
    )

    # Check projects has been inserted
    retrieved_project_1 = await db_client.db["projects"].find_one(
        {"_id": ObjectId(project_ids[0])}
    )
    assert retrieved_project_1["name"] == PROJECT_1.name

    retrieved_project_2 = await db_client.db["projects"].find_one(
        {"_id": ObjectId(project_ids[1])}
    )
    assert retrieved_project_2["name"] == PROJECT_2.name


@pytest.mark.asyncio
async def test_insert_many_same_ids(db_client):
    project_id = await db_client.insert(collection="projects", model=PROJECT_1)
    sample_ids = await db_client.insert_many(
        collection="samples",
        models=[SAMPLE_1, SAMPLE_2],
        ids={"project_id": ObjectId(project_id)},
    )

    # Check samples has been inserted with project ID attached
    retrieved_sample_1 = await db_client.db["samples"].find_one(
        {"_id": ObjectId(sample_ids[0])}
    )

    assert retrieved_sample_1["shot_id"] == SAMPLE_1.shot_id
    assert retrieved_sample_1["project_id"] == ObjectId(project_id)

    retrieved_sample_2 = await db_client.db["samples"].find_one(
        {"_id": ObjectId(sample_ids[1])}
    )

    assert retrieved_sample_2["shot_id"] == SAMPLE_2.shot_id
    assert retrieved_sample_2["project_id"] == ObjectId(project_id)


@pytest.mark.asyncio
async def test_update(db_client: MongoDBClient):
    project_id = await db_client.insert(collection="projects", model=PROJECT_1)
    project = await utils.get_project(db_client, project_id)
    project.name = "New Project Name"

    await db_client.update("projects", project, ObjectId(project_id))

    # Check project has been updated
    retrieved_project = await db_client.db["projects"].find_one(
        {"_id": ObjectId(project_id)}
    )
    assert retrieved_project["name"] == "New Project Name"


@pytest.mark.asyncio
async def test_insert_many_different_ids(db_client):
    project_ids = await db_client.insert_many(
        collection="projects", models=[PROJECT_1, PROJECT_2]
    )
    sample_ids = await db_client.insert_many(
        collection="samples",
        models=[SAMPLE_1, SAMPLE_2],
        ids=[
            {"project_id": ObjectId(project_ids[0])},
            {"project_id": ObjectId(project_ids[1])},
        ],
    )

    # Check samples has been inserted with correct project ID attached
    retrieved_sample_1 = await db_client.db["samples"].find_one(
        {"_id": ObjectId(sample_ids[0])}
    )

    assert retrieved_sample_1["shot_id"] == SAMPLE_1.shot_id
    assert retrieved_sample_1["project_id"] == ObjectId(project_ids[0])

    retrieved_sample_2 = await db_client.db["samples"].find_one(
        {"_id": ObjectId(sample_ids[1])}
    )

    assert retrieved_sample_2["shot_id"] == SAMPLE_2.shot_id
    assert retrieved_sample_2["project_id"] == ObjectId(project_ids[1])


@pytest.mark.asyncio
async def test_get_document_by_id(db_client):
    result = await db_client.db["projects"].insert_one({"name": "Test Project"})
    returned_item = await db_client.get_document_by_id(
        collection="projects", object_id=result.inserted_id
    )
    assert returned_item["name"] == "Test Project"
    assert returned_item["_id"] == result.inserted_id


@pytest.mark.asyncio
async def test_get_document_by_id_doesnt_exist(db_client):
    returned_item = await db_client.get_document_by_id(
        collection="projects", object_id=ObjectId()
    )
    assert returned_item is None


@pytest.mark.asyncio
async def test_get_all_documents(db_client):
    await db_client.db["projects"].insert_one({"name": "Test Project 1"})
    await db_client.db["projects"].insert_one({"name": "Test Project 2"})
    await db_client.db["samples"].insert_one(
        {"name": "Test Sample"}
    )  # <--- shouldn't include this
    returned_items = await db_client.get_all_documents(collection="projects")
    assert len(returned_items) == 2
    assert returned_items[0]["name"] == "Test Project 1"
    assert returned_items[1]["name"] == "Test Project 2"


@pytest.mark.asyncio
async def test_get_filtered_documents_defaults(db_client, setup_db_for_filtering):
    # Defaults to timestamp, descending
    results = await db_client.get_filtered_documents("projects")
    assert [result["idx"] for result in results] == [3, 2, 1]


@pytest.mark.asyncio
async def test_get_filtered_documents_ascending(db_client, setup_db_for_filtering):
    # Defaults to timestamp sorting
    results = await db_client.get_filtered_documents(
        "projects", sort_direction="ascending"
    )
    assert [result["idx"] for result in results] == [1, 2, 3]


@pytest.mark.asyncio
async def test_get_filtered_documents_sortby(db_client, setup_db_for_filtering):
    # Defaults to descending
    results = await db_client.get_filtered_documents("projects", sort_by="number")
    assert [result["number"] for result in results] == [30, 20, 10]
    assert [result["idx"] for result in results] == [2, 3, 1]


@pytest.mark.asyncio
async def test_get_filtered_documents_start(db_client, setup_db_for_filtering):
    # Defaults to timestamp, descending
    results = await db_client.get_filtered_documents("projects", start=1)
    assert [result["idx"] for result in results] == [2, 1]


@pytest.mark.asyncio
async def test_get_filtered_documents_limit(db_client, setup_db_for_filtering):
    # Defaults to timestamp, descending - 2, 3, 1
    results = await db_client.get_filtered_documents("projects", limit=2)
    assert [result["idx"] for result in results] == [3, 2]


@pytest.mark.asyncio
async def test_get_filtered_documents_start_limit(db_client, setup_db_for_filtering):
    # Defaults to timestamp, descending - 2, 3, 1
    results = await db_client.get_filtered_documents("projects", start=1, limit=1)
    assert [result["idx"] for result in results] == [2]


@pytest.mark.asyncio
async def test_get_filtered_documents_high_start(db_client, setup_db_for_filtering):
    results = await db_client.get_filtered_documents("projects", start=10)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_get_filtered_documents_high_limit(db_client, setup_db_for_filtering):
    results = await db_client.get_filtered_documents("projects", limit=10)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_get_filtered_documents_filter(db_client, setup_db_for_filtering):
    results = await db_client.get_filtered_documents(
        "projects", filters={"local": True}
    )
    assert len(results) == 2
    assert [result["idx"] for result in results] == [3, 1]


@pytest.mark.asyncio
async def test_delete_filtered_documents_all(db_client, setup_db_for_filtering):
    results = await db_client.delete_filtered_documents("projects")

    assert results.deleted_count == 3

    cursor = db_client.db["projects"].find()
    projects = await cursor.to_list()
    assert len(projects) == 0


@pytest.mark.asyncio
async def test_delete_filtered_documents_filters(db_client, setup_db_for_filtering):
    results = await db_client.delete_filtered_documents(
        "projects", filters={"local": True}
    )

    assert results.deleted_count == 2

    cursor = db_client.db["projects"].find()
    projects = await cursor.to_list()

    assert len(projects) == 1
    assert projects[0]["idx"] == 2
