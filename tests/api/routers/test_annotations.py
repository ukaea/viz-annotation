import pytest
from bson.objectid import ObjectId


@pytest.mark.asyncio
async def test_get_all_annotations(api_client, setup_db):
    response = await api_client.get(
        f"/projects/{setup_db['project_id_1']}/samples/{setup_db['sample_id_1']}/annotations?sort_direction=ascending"
    )
    assert response.status_code == 200
    returned_annotations = response.json()
    assert [annotation["_id"] for annotation in returned_annotations] == [
        setup_db["annotation_id_1"],
        setup_db["annotation_id_2"],
        setup_db["annotation_id_3"],
    ]


@pytest.mark.asyncio
async def test_get_all_annotations_sortby(api_client, setup_db):
    response = await api_client.get(
        f"/projects/{setup_db['project_id_1']}/samples/{setup_db['sample_id_1']}/annotations?sort_by=label"
    )
    # Should sort by label
    # So 'annotation' (annotation 1), then 'disruption' (annotation 3) then 'ramp_up' (annotation 2)
    # Default sort direction is descending, so will return the opposite of this: 2, 3, 1
    assert response.status_code == 200
    returned_annotations = response.json()
    assert [annotation["label"] for annotation in returned_annotations] == [
        "ramp_up",
        "disruption",
        "annotation",
    ]
    assert [annotation["_id"] for annotation in returned_annotations] == [
        setup_db["annotation_id_2"],
        setup_db["annotation_id_3"],
        setup_db["annotation_id_1"],
    ]


@pytest.mark.asyncio
async def test_get_annotations_start(api_client, setup_db):
    response = await api_client.get(
        f"/projects/{setup_db['project_id_1']}/samples/{setup_db['sample_id_1']}/annotations?start=1&sort_direction=ascending"
    )
    # Should return 2 annotations
    assert response.status_code == 200
    returned_annotations = response.json()
    assert len(returned_annotations) == 2
    assert [sample["_id"] for sample in returned_annotations] == [
        setup_db["annotation_id_2"],
        setup_db["annotation_id_3"],
    ]


@pytest.mark.asyncio
async def test_get_annotations_count(api_client, setup_db):
    response = await api_client.get(
        f"/projects/{setup_db['project_id_1']}/samples/{setup_db['sample_id_1']}/annotations?count=2&sort_direction=ascending"
    )
    # Should return 2 samples
    assert response.status_code == 200
    returned_annotations = response.json()
    assert len(returned_annotations) == 2
    assert [annotation["_id"] for annotation in returned_annotations] == [
        setup_db["annotation_id_1"],
        setup_db["annotation_id_2"],
    ]


@pytest.mark.asyncio
async def test_get_annotations_start_count(api_client, setup_db):
    response = await api_client.get(
        f"/projects/{setup_db['project_id_1']}/samples/{setup_db['sample_id_1']}/annotations?start=1&count=1&sort_direction=ascending"
    )
    # Should return 1 annotation
    assert response.status_code == 200
    returned_annotations = response.json()
    assert len(returned_annotations) == 1
    assert [annotation["_id"] for annotation in returned_annotations] == [
        setup_db["annotation_id_2"]
    ]


@pytest.mark.asyncio
async def test_get_annotations_invalid_start(api_client, setup_db):
    response = await api_client.get(
        f"/projects/{setup_db['project_id_1']}/samples/{setup_db['sample_id_1']}/annotations?start=10"
    )
    # Should return 0 projects
    assert response.status_code == 200
    returned_annotations = response.json()
    assert len(returned_annotations) == 0


@pytest.mark.asyncio
async def test_get_annotations_validated(api_client, setup_db):
    response = await api_client.get(
        f"/projects/{setup_db['project_id_1']}/samples/{setup_db['sample_id_1']}/annotations?validated=true&sort_direction=ascending"
    )
    # Should return 2 annotations
    assert response.status_code == 200
    returned_annotations = response.json()
    assert len(returned_annotations) == 2
    assert [annotation["_id"] for annotation in returned_annotations] == [
        setup_db["annotation_id_1"],
        setup_db["annotation_id_2"],
    ]
    assert all([annotation["validated"] for annotation in returned_annotations])


@pytest.mark.asyncio
async def test_get_annotations_unvalidated(api_client, setup_db):
    response = await api_client.get(
        f"/projects/{setup_db['project_id_1']}/samples/{setup_db['sample_id_1']}/annotations?validated=false&sort_direction=ascending"
    )
    # Should return 1 annotation
    assert response.status_code == 200
    returned_annotations = response.json()
    assert len(returned_annotations) == 1
    assert returned_annotations[0]["_id"] == setup_db["annotation_id_3"]
    assert not returned_annotations[0]["validated"]


@pytest.mark.asyncio
async def test_get_annotation(api_client, setup_db):
    response = await api_client.get(
        f"/projects/{setup_db['project_id_1']}/samples/{setup_db['sample_id_1']}/annotations?start=1&count=1&sort_direction=ascending"
    )
    assert response.status_code == 200
    returned_annotation = response.json()[0]
    # Check info matches what we created the entry with
    assert returned_annotation.get("time_min") == 0.1
    assert returned_annotation.get("time_max") == 0.2
    assert returned_annotation.get("label") == "ramp_up"
    assert returned_annotation.get("validated")

    # Then also check ID and timestamp are returned - should have been added automatically
    assert returned_annotation.get("_id") == setup_db["annotation_id_2"]
    assert returned_annotation.get("project_id") == setup_db["project_id_1"]
    assert returned_annotation.get("sample_id") == setup_db["sample_id_1"]
    assert returned_annotation.get("timestamp")


@pytest.mark.asyncio
async def test_delete_all_annotations(api_client, setup_db, db_client):
    response = await api_client.delete(
        f"/projects/{setup_db['project_id_1']}/annotations"
    )
    assert response.status_code == 200

    # Check annotations for this project have ALL been deleted
    # But annotations for other samples and projects still exist
    annotations = await db_client.get_all_documents("annotations")
    assert len(annotations) == 1
    assert str(annotations[0]["project_id"]) == setup_db["project_id_2"]

    # Check that sample associated with these annotations has NOT been deleted
    samples = await db_client.get_all_documents("samples")
    assert len(samples) == 4
    # Check sample with above ID is in database
    assert setup_db["sample_id_1"] in [str(sample.get("_id")) for sample in samples]

    # Check that project assocaited with these annotations has NOT been deleted
    projects = await db_client.get_all_documents("projects")
    assert len(projects) == 3
    # Check project with above ID is in database
    assert setup_db["project_id_1"] in [str(project.get("_id")) for project in projects]


@pytest.mark.asyncio
async def test_delete_sample_annotations(api_client, setup_db, db_client):
    response = await api_client.delete(
        f"/projects/{setup_db['project_id_1']}/samples/{setup_db['sample_id_1']}/annotations"
    )
    assert response.status_code == 200

    # Check annotations for this sample have been deleted
    # But annotations for other samples and projects still exist
    annotations = await db_client.get_all_documents("annotations")
    assert len(annotations) == 2
    assert setup_db["sample_id_1"] not in [
        annotation.get("sample_id") for annotation in annotations
    ]

    # Check that sample associated with these annotations has NOT been deleted
    samples = await db_client.get_all_documents("samples")
    assert len(samples) == 4
    # Check sample with above ID is in database
    assert setup_db["sample_id_1"] in [str(sample.get("_id")) for sample in samples]

    # Check that project assocaited with these annotations has NOT been deleted
    projects = await db_client.get_all_documents("projects")
    assert len(projects) == 3
    # Check project with above ID is in database
    assert setup_db["project_id_1"] in [str(project.get("_id")) for project in projects]


@pytest.mark.asyncio
async def test_create_annotations(api_client, setup_db, db_client):
    in_annotations = [
        {
            "label": "ramp_up",
            "time_min": 0.1,
            "time_max": 0.2,
            "created_by": "manual",
            "type": "time_region",
            "validated": True,
        },
        {
            "label": "flat_top",
            "time_min": 0.2,
            "time_max": 0.5,
            "created_by": "manual",
            "type": "time_region",
            "validated": True,
        },
        {
            "label": "disruption",
            "time": 0.5,
            "created_by": "manual",
            "type": "time_point",
            "validated": True,
        },
    ]

    project_id = "project_id_1"
    sample_id = "sample_id_2"
    response = await api_client.put(
        f"/projects/{setup_db[project_id]}/samples/{setup_db[sample_id]}/annotations",
        json=in_annotations,
    )
    assert response.status_code == 200

    # Check they have been added to database (existing annotations from other users are preserved)
    annotations = await db_client.get_all_documents("annotations")
    assert len(annotations) == 8
    db_annotations = await db_client.get_filtered_documents(
        "annotations", filters={"sample_id": ObjectId(setup_db[sample_id])}
    )
    for in_annotation in in_annotations:
        # Find annotation with that label
        db_annotation = next(
            annotation
            for annotation in db_annotations
            if annotation["label"] == in_annotation["label"]
        )
        for key, value in in_annotation.items():
            if key == "created_by":
                continue  # server overwrites created_by with authenticated user
            assert db_annotation[key] == value

        assert db_annotation.get("timestamp")
        assert db_annotation.get("_id")
        assert str(db_annotation.get("project_id")) == setup_db[project_id]
        assert str(db_annotation.get("sample_id")) == setup_db[sample_id]


@pytest.mark.asyncio
async def test_create_annotation_invalid(api_client, setup_db, db_client):
    in_annotations = [
        {"time": 5.2, "validated": "validated"},
    ]
    response = await api_client.put(
        f"/projects/{setup_db['project_id_1']}/samples/{setup_db['sample_id_3']}/annotations",
        json=in_annotations,
    )
    assert response.status_code == 422
    errors = response.json().get("detail", [])
    # Should flag that label is missing, validated is wrong type
    # It will also flag validation errors from all other possible 'annotation' schemas since none validated correctly...
    assert len(errors) >= 2

    # Check it has not been added to database
    annotations = await db_client.get_all_documents("annotations")
    assert len(annotations) == 5
    assert setup_db["sample_id_3"] not in [
        annotation["sample_id"] for annotation in annotations
    ]


@pytest.mark.asyncio
async def test_export_annotations(api_client, setup_db):
    response = await api_client.get(f"/projects/{setup_db['project_id_1']}/annotations")
    assert response.status_code == 200
    exported_data = response.json()

    # Check that all annotations for the sample are exported
    assert len(exported_data) == 4
    # Verify annotation IDs match
    exported_ids = [annotation["_id"] for annotation in exported_data]
    assert setup_db["annotation_id_1"] in exported_ids
    assert setup_db["annotation_id_2"] in exported_ids
    assert setup_db["annotation_id_3"] in exported_ids


@pytest.mark.asyncio
async def test_import_annotations(api_client, setup_db, db_client):
    import_data = [
        {
            "project_id": setup_db["project_id_1"],
            "sample_id": setup_db["sample_id_2"],
            "shot_id": 2,
            "label": "new_annotation",
            "time_min": 0.3,
            "time_max": 0.4,
            "created_by": "manual",
            "type": "time_region",
            "validated": False,
        },
        {
            "project_id": setup_db["project_id_1"],
            "sample_id": setup_db["sample_id_2"],
            "shot_id": 2,
            "label": "another_annotation",
            "time": 0.6,
            "created_by": "manual",
            "type": "time_point",
            "validated": True,
        },
    ]

    response = await api_client.put(
        f"/projects/{setup_db['project_id_1']}/annotations",
        json=import_data,
    )
    assert response.status_code == 200

    # Check annotations were added to database
    annotations = await db_client.get_all_documents("annotations")
    assert len(annotations) == 7

    # Verify imported annotations
    db_annotations = await db_client.get_filtered_documents(
        "annotations", filters={"sample_id": ObjectId(setup_db["sample_id_2"])}
    )
    imported_annotations = [
        ann for ann in db_annotations if ann["created_by"] == "manual"
    ]
    assert len(imported_annotations) == 2


@pytest.mark.asyncio
async def test_batch_update_annotations(api_client, setup_db, db_client):
    """Test batch updating annotations by first deleting existing ones then importing new ones."""
    # First delete existing annotations for the samples we're updating
    await api_client.delete(
        f"/projects/{setup_db['project_id_1']}/samples/{setup_db['sample_id_1']}/annotations"
    )
    await api_client.delete(
        f"/projects/{setup_db['project_id_1']}/samples/{setup_db['sample_id_2']}/annotations"
    )

    annotations_batch = [
        {
            "project_id": setup_db["project_id_1"],
            "sample_id": setup_db["sample_id_1"],
            "shot_id": 1,
            "type": "time_point",
            "label": "TestAnnotation1",
            "time": 1,
            "created_by": "disruption_cnn",
        },
        {
            "project_id": setup_db["project_id_1"],
            "sample_id": setup_db["sample_id_1"],
            "shot_id": 1,
            "type": "time_point",
            "label": "TestAnnotation2",
            "time": 2,
            "created_by": "disruption_cnn",
        },
        {
            "project_id": setup_db["project_id_1"],
            "sample_id": setup_db["sample_id_2"],
            "shot_id": 2,
            "label": "TestAnnotation",
            "type": "time_region",
            "time_min": 1,
            "time_max": 2,
            "created_by": "disruption_cnn",
        },
    ]

    response = await api_client.put(
        f"/projects/{setup_db['project_id_1']}/annotations", json=annotations_batch
    )
    assert response.status_code == 200

    # Check annotations for sample 1 have been updated
    annotations_sample_1 = await db_client.get_filtered_documents(
        "annotations",
        filters={"sample_id": ObjectId(setup_db["sample_id_1"])},
        sort_by="time",
        sort_direction="ascending",
    )
    assert len(annotations_sample_1) == 2
    assert annotations_sample_1[0]["label"] == "TestAnnotation1"
    assert annotations_sample_1[0]["time"] == 1
    assert annotations_sample_1[1]["label"] == "TestAnnotation2"
    assert annotations_sample_1[1]["time"] == 2

    # Check annotation for sample 2 also updated
    annotations_sample_1 = await db_client.get_filtered_documents(
        "annotations", filters={"sample_id": ObjectId(setup_db["sample_id_2"])}
    )
    assert len(annotations_sample_1) == 1
    assert annotations_sample_1[0]["label"] == "TestAnnotation"
    assert annotations_sample_1[0]["time_min"] == 1
    assert annotations_sample_1[0]["time_max"] == 2
