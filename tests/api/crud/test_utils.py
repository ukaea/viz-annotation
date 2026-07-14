import pytest
from bson.objectid import ObjectId
from tests.db_definitions import PROJECT_1, SAMPLE_1, ANNOTATION_1, ANNOTATION_2
from toktagger.api.schemas.samples import SampleUpdate
from toktagger.api.schemas.models import ModelUpdate, ModelIn
import toktagger.api.crud.utils as utils
from fastapi import HTTPException
import tempfile
import os
from pathlib import Path


@pytest.mark.asyncio
async def test_get_projects(db_client, setup_db):
    projects = await utils.get_projects(db_client)
    # Check three projects returned
    assert len(projects) == 3
    # Check returned in correct order - reverse order of created
    assert [project["name"] for project in projects] == [
        "project_2",
        "test_project_1",
        "test_project_0",
    ]


@pytest.mark.asyncio
async def test_get_projects_by_name(db_client, setup_db):
    projects = await utils.get_projects(db_client, name="Test")
    # Should fuzzy search for any names including 'test'
    # Case insensitive
    assert len(projects) == 2
    assert [project["name"] for project in projects] == [
        "test_project_1",
        "test_project_0",
    ]


@pytest.mark.asyncio
async def test_get_project(db_client, setup_db):
    project = await utils.get_project(db_client, setup_db["project_id_1"])
    assert project.name == "test_project_0"
    assert project.id == setup_db["project_id_1"]


@pytest.mark.asyncio
async def test_get_project_wrong_id(db_client, setup_db):
    with pytest.raises(HTTPException, match="Project not found with that ID"):
        await utils.get_project(db_client, str(ObjectId()))


@pytest.mark.asyncio
async def test_delete_project(db_client, setup_db):
    await utils.delete_projects(db_client, setup_db["project_id_1"])
    projects = await db_client.get_filtered_documents("projects")
    assert len(projects) == 2
    assert setup_db["project_id_1"] not in [project["_id"] for project in projects]


@pytest.mark.asyncio
async def test_delete_projects(db_client, setup_db):
    await utils.delete_projects(db_client)
    projects = await db_client.get_filtered_documents("projects")
    assert len(projects) == 0


@pytest.mark.asyncio
async def test_delete_project_wrong_id(db_client, setup_db):
    with pytest.raises(HTTPException, match="Project not found with that ID"):
        await utils.delete_projects(db_client, str(ObjectId()))


@pytest.mark.asyncio
async def test_update_project(db_client, setup_db):
    project = await utils.get_project(db_client, setup_db["project_id_1"])
    project.name = "Updated Project Name"
    await utils.update_project(db_client, setup_db["project_id_1"], project=project)
    # Check project has been updated
    await db_client.db["projects"].find_one({"_id": ObjectId(setup_db["project_id_1"])})


@pytest.mark.asyncio
async def test_get_samples(db_client, setup_db):
    samples = await utils.get_samples(db_client, project_id=setup_db["project_id_1"])
    # Check three samples returned
    assert len(samples) == 2
    # Check returned in correct order - reverse order of created
    assert [sample.shot_id for sample in samples] == [2, 1]


@pytest.mark.asyncio
async def test_get_samples_by_shot_id(db_client, setup_db):
    samples = await utils.get_samples(
        db_client, project_id=setup_db["project_id_1"], shot_id=1
    )
    # Should do an exact search for this shot_id
    assert len(samples) == 1
    assert samples[0].shot_id == 1


@pytest.mark.asyncio
async def test_get_sample(db_client, setup_db):
    sample = await utils.get_sample(
        db_client, setup_db["project_id_1"], setup_db["sample_id_1"]
    )
    assert sample.shot_id == 1
    assert sample.id == setup_db["sample_id_1"]


@pytest.mark.asyncio
async def test_get_files(db_client, setup_db):
    # Create temporary directory and parquet files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create fake parquet files
        Path(os.path.join(temp_dir, "file1.parquet")).touch()
        Path(os.path.join(temp_dir, "file2.parquet")).touch()
        Path(os.path.join(temp_dir, "file3.txt")).touch()  # Non-parquet file

        file_names = await utils.get_files(dir_path=temp_dir, file_type="parquet")
        assert len(file_names) == 2
        assert all(file.endswith(".parquet") for file in file_names)


@pytest.mark.asyncio
async def test_get_sample_summary(db_client, setup_db):
    summary = await utils.get_sample_summary(
        db_client, project_id=setup_db["project_id_1"]
    )
    assert summary.total == 2
    assert summary.shot_min == 1
    assert summary.shot_max == 2
    assert summary.data is not None


@pytest.mark.asyncio
async def test_get_sample_wrong_id(db_client, setup_db):
    with pytest.raises(HTTPException, match="Sample not found with that ID"):
        await utils.get_sample(db_client, setup_db["project_id_1"], str(ObjectId()))


@pytest.mark.asyncio
async def test_delete_all_samples_in_project(db_client, setup_db):
    # Delete all samples belonging to a project
    await utils.delete_samples(db_client, project_id=setup_db["project_id_1"])
    # Should only be one remaining sample
    samples = await db_client.get_filtered_documents("samples")
    assert len(samples) == 2


@pytest.mark.asyncio
async def test_delete_specific_sample(db_client, setup_db):
    # Delete all samples belonging to a project
    await utils.delete_samples(
        db_client,
        project_id=setup_db["project_id_1"],
        sample_id=setup_db["sample_id_1"],
    )
    # Should be two remaining sample
    samples = await db_client.get_filtered_documents("samples")
    assert len(samples) == 3
    assert 1 not in [sample["shot_id"] for sample in samples]


@pytest.mark.asyncio
async def test_update_sample(db_client, setup_db):
    # Check Sample 1 has validated_annotations = False
    sample_1 = await db_client.get_document_by_id(
        "samples", ObjectId(setup_db["sample_id_1"])
    )
    assert not sample_1["validated_annotations"]

    # Update sample to set validated_annotations to True
    updates = SampleUpdate(validated_annotations=True)
    await utils.update_sample(
        db_client, sample_id=setup_db["sample_id_1"], updates=updates
    )

    # Check sample is updated
    sample_1_updated = await db_client.get_document_by_id(
        "samples", ObjectId(setup_db["sample_id_1"])
    )
    assert sample_1_updated["validated_annotations"]


@pytest.mark.asyncio
async def test_get_annotations_in_project(db_client, setup_db):
    annotations = await utils.get_annotations(
        db_client, project_id=setup_db["project_id_1"]
    )
    # Check four annotations returned
    assert len(annotations) == 4
    # Check returned in correct order - reverse order of created
    assert [str(ann.id) for ann in annotations] == [
        setup_db["annotation_id_4"],
        setup_db["annotation_id_3"],
        setup_db["annotation_id_2"],
        setup_db["annotation_id_1"],
    ]


@pytest.mark.asyncio
async def test_get_annotations_in_sample(db_client, setup_db):
    annotations = await utils.get_annotations(
        db_client,
        project_id=setup_db["project_id_1"],
        sample_id=setup_db["sample_id_1"],
    )
    # Check three annotations returned
    assert len(annotations) == 3
    # Check returned in correct order - reverse order of created
    assert [str(ann.id) for ann in annotations] == [
        setup_db["annotation_id_3"],
        setup_db["annotation_id_2"],
        setup_db["annotation_id_1"],
    ]


@pytest.mark.asyncio
async def test_get_annotations_validated(db_client, setup_db):
    annotations = await utils.get_annotations(
        db_client, project_id=setup_db["project_id_1"], validated=True
    )

    assert len(annotations) == 2
    assert [str(annotation.id) for annotation in annotations] == [
        setup_db["annotation_id_2"],
        setup_db["annotation_id_1"],
    ]


@pytest.mark.asyncio
async def test_add_annotations(db_client):
    project_id = await db_client.insert(collection="projects", model=PROJECT_1)
    sample_id = await db_client.insert(collection="samples", model=SAMPLE_1)
    annotation_ids = await utils.add_annotations(
        db_client,
        project_id=project_id,
        sample_id=sample_id,
        annotations=[ANNOTATION_1, ANNOTATION_2],
    )

    assert len(annotation_ids) == 2

    returned_annotations = await db_client.get_all_documents(collection="annotations")
    for annotation in returned_annotations:
        assert str(annotation["project_id"]) == project_id
        assert str(annotation["sample_id"]) == sample_id
        assert str(annotation["_id"]) in annotation_ids


@pytest.mark.asyncio
async def test_delete_all_annotations_in_project(db_client, setup_db):
    # Delete all annotations belonging to a project
    await utils.delete_annotations(db_client, project_id=setup_db["project_id_1"])
    # Should only be one remaining annotation
    annotations = await db_client.get_filtered_documents("annotations")
    assert len(annotations) == 1
    assert str(annotations[0]["_id"]) == setup_db["annotation_id_5"]


@pytest.mark.asyncio
async def test_delete_all_annotations_in_project_and_sample(db_client, setup_db):
    # Delete all annotations belonging to a project and sample
    await utils.delete_annotations(
        db_client,
        project_id=setup_db["project_id_1"],
        sample_id=setup_db["sample_id_1"],
    )
    # Should only be two remaining annotations
    annotations = await db_client.get_filtered_documents("annotations")
    assert len(annotations) == 2
    assert str(annotations[0]["_id"]) == setup_db["annotation_id_5"]
    assert str(annotations[1]["_id"]) == setup_db["annotation_id_4"]


@pytest.mark.asyncio
async def test_delete_specific_annotation(db_client, setup_db):
    # Delete specific annotation by ID
    await utils.delete_annotations(
        db_client,
        project_id=setup_db["project_id_1"],
        sample_id=setup_db["sample_id_1"],
        annotation_id=setup_db["annotation_id_1"],
    )
    # Should be four remaining annotations
    annotations = await db_client.get_filtered_documents("annotations")
    assert len(annotations) == 4
    assert setup_db["annotation_id_1"] not in [
        str(annotation["_id"] for annotation in annotations)
    ]


@pytest.mark.asyncio
async def test_import_annotations_success(db_client, setup_db):
    # Create new annotations with sample_id references
    new_annotations = [
        ANNOTATION_1.model_copy(update={"sample_id": setup_db["sample_id_1"]}),
        ANNOTATION_2.model_copy(update={"sample_id": setup_db["sample_id_2"]}),
    ]

    await utils.import_annotations(
        db_client,
        project_id=setup_db["project_id_1"],
        annotations=new_annotations,
    )

    # Verify annotations were added
    all_annotations = await db_client.get_filtered_documents("annotations")
    assert len(all_annotations) == 7  # 5 existing + 2 new


@pytest.mark.asyncio
async def test_import_annotations_empty_list(db_client, setup_db):
    # Should handle empty list gracefully
    await utils.import_annotations(
        db_client,
        project_id=setup_db["project_id_1"],
        annotations=[],
    )

    # Verify no changes
    all_annotations = await db_client.get_filtered_documents("annotations")
    assert len(all_annotations) == 5


@pytest.mark.asyncio
async def test_import_annotations_project_not_found(db_client, setup_db):
    new_annotations = [
        ANNOTATION_1.model_copy(update={"sample_id": setup_db["sample_id_1"]}),
    ]

    with pytest.raises(HTTPException, match="Project not found with that ID"):
        await utils.import_annotations(
            db_client,
            project_id=str(ObjectId()),
            annotations=new_annotations,
        )


@pytest.mark.asyncio
async def test_import_annotations_sample_not_found(db_client, setup_db):
    # Use invalid sample_id
    shot_id = 10
    new_annotations = [
        ANNOTATION_1.model_copy(update={"shot_id": shot_id}),
    ]

    with pytest.raises(HTTPException, match=f"Sample not found with shot ID {shot_id}"):
        await utils.import_annotations(
            db_client,
            project_id=setup_db["project_id_1"],
            annotations=new_annotations,
        )


@pytest.mark.asyncio
async def test_import_annotations_multiple_samples(db_client, setup_db):
    # Import annotations for multiple samples
    new_annotations = [
        ANNOTATION_1.model_copy(update={"shot_id": 1}),
        ANNOTATION_2.model_copy(update={"shot_id": 1}),
        ANNOTATION_1.model_copy(update={"shot_id": 2}),
    ]

    await utils.import_annotations(
        db_client,
        project_id=setup_db["project_id_1"],
        annotations=new_annotations,
    )

    # Verify annotations were added correctly
    sample1_annotations = await utils.get_annotations(
        db_client,
        project_id=setup_db["project_id_1"],
        sample_id=setup_db["sample_id_1"],
    )
    sample2_annotations = await utils.get_annotations(
        db_client,
        project_id=setup_db["project_id_1"],
        sample_id=setup_db["sample_id_2"],
    )

    assert len(sample1_annotations) == 5  # 3 existing + 2 new
    assert len(sample2_annotations) == 2  # 1 existing + 1 new


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_get_models(db_client, setup_model_db):
    models = await utils.get_models(db_client, project_id=setup_model_db["project_id"])
    # Check three models returned
    assert len(models) == 4
    # Check returned in correct order - reverse order of version
    assert [model.version for model in models] == [3, 2, 1, 1]


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_get_models_by_type(db_client, setup_model_db):
    models = await utils.get_models(
        db_client,
        project_id=setup_model_db["project_id"],
        model_type="mock_disruption_cnn",
    )
    # Check two models returned
    assert len(models) == 2
    # Check returned in correct order - reverse order of created, not version 3
    assert [model.version for model in models] == [2, 1]
    assert all(model.type == "mock_disruption_cnn" for model in models)


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_get_models_by_status(db_client, setup_model_db):
    models = await utils.get_models(
        db_client, project_id=setup_model_db["project_id"], status="started"
    )
    # Check one model returned
    assert len(models) == 1
    # Check this is the correct model
    assert models[0].version == 3
    assert models[0].type == "disruption_cnn"


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_get_model(db_client, setup_model_db):
    model = await utils.get_model(
        db_client,
        project_id=setup_model_db["project_id"],
        model_type="mock_disruption_cnn",
        version=2,
    )
    assert model.id == setup_model_db["model_id_2"]


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_get_model_doesnt_exist(db_client, setup_model_db):
    with pytest.raises(
        HTTPException,
        match="No models found of that type for this project!",
    ):
        await utils.get_model(
            db_client,
            project_id=setup_model_db["project_id"],
            model_type="mock_disruption_cnn",
            version=4,
        )


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_update_model(db_client, setup_model_db):
    model_updates = ModelUpdate(training_status="completed", progress=100, score=80)
    await utils.update_model(
        db_client, model_id=setup_model_db["model_id_3"], updates=model_updates
    )
    # Check model has been updated
    model_updated = await db_client.get_document_by_id(
        "models", ObjectId(setup_model_db["model_id_3"])
    )
    assert model_updated["training_status"] == "completed"
    assert model_updated["progress"] == 100
    assert model_updated["score"] == 80


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_add_model(db_client, setup_model_db):
    model = ModelIn(
        type="mock_disruption_cnn",
        version=3,
        training_status="queued",
        progress=0,
        score=0,
    )
    await utils.add_model(
        db_client, project_id=setup_model_db["project_id"], model=model
    )
    # Check a new model has been added
    models = await db_client.get_all_documents(collection="models")
    assert len(models) == 5


@pytest.mark.asyncio
@pytest.mark.models_enabled
async def test_delete_model(db_client, setup_model_db):
    await utils.delete_model(
        db_client, setup_model_db["project_id"], setup_model_db["model_id_1"]
    )
    models = await db_client.get_filtered_documents("models")
    assert len(models) == 3
    assert setup_model_db["model_id_1"] not in [model["_id"] for model in models]
