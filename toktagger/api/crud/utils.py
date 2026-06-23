from pathlib import Path
from collections import defaultdict
from typing import Optional, Literal
from fastapi import HTTPException
from pydantic import TypeAdapter
from toktagger.api.crud.db import MongoDBClient
from toktagger.api.schemas import convert_to_objectid
from toktagger.api.schemas.annotations import (
    AnnotationOutTypes,
    AnnotationOutTypeAdapter,
    AnnotationBatchTypes,
)
from toktagger.api.schemas.projects import Project
from toktagger.api.schemas.samples import FileData, Sample, SampleUpdate, SampleSummary
from toktagger.api.schemas.models import Model, ModelIn, ModelUpdate
from toktagger.api.auth.core import hash_password
from toktagger.api.schemas.users import (
    ProjectMember,
    ProjectMemberOut,
    ProjectMemberUpdate,
    UserIn,
    UserOut,
    UserUpdate,
)


async def get_projects(
    db_client: MongoDBClient,
    name: Optional[str] = None,
    sort_by: Optional[str] = "_id",
    sort_direction: Optional[Literal["ascending", "descending"]] = "descending",
    start: Optional[int] = 0,
    count: Optional[int] = None,
):
    filters = {}
    if name:
        # Search with regex, return any projects which start with the searched for string, case insensitive
        filters["name"] = {"$regex": f"{name}", "$options": "i"}

    # Return a list of all projects and info about them
    projects = await db_client.get_filtered_documents(
        collection="projects",
        filters=filters,
        sort_by=sort_by,
        sort_direction=sort_direction,
        start=start,
        limit=count if count is not None else 0,
    )

    return projects


async def get_project(db_client: MongoDBClient, project_id: str) -> Project:
    obj_id = convert_to_objectid(project_id, "projects")

    projects = await db_client.get_filtered_documents(
        collection="projects", filters={"_id": obj_id}
    )

    if len(projects) == 0:
        raise HTTPException(status_code=404, detail="Project not found with that ID.")

    return Project(**projects[0])


async def get_samples(
    db_client: MongoDBClient,
    project_id: str,
    validated: Optional[bool] = None,
    shot_id: Optional[int] = None,
    sort_by: str = "_id",
    sort_direction: Literal["ascending", "descending"] = "descending",
    start: int = 0,
    count: Optional[int] = None,
) -> list[Sample]:
    # Return a list of all samples for this project and info about them
    project_obj_id = convert_to_objectid(project_id, "projects")

    if not await db_client.get_document_by_id("projects", project_obj_id):
        raise HTTPException(status_code=404, detail="Project not found with that ID.")

    filters = {"project_id": project_obj_id}

    if validated is not None:
        filters["validated_annotations"] = validated

    if shot_id:
        filters["shot_id"] = shot_id

    samples = await db_client.get_filtered_documents(
        collection="samples",
        filters=filters,
        sort_by=sort_by,
        sort_direction=sort_direction,
        start=start,
        limit=count if count is not None else 0,
    )

    samples = [TypeAdapter(Sample).validate_python(s) for s in samples]
    return samples


async def update_sample(
    db_client: MongoDBClient, sample_id: str, updates: SampleUpdate
):
    sample_obj_id = convert_to_objectid(sample_id, "samples")

    # Check sample already exists
    if not await db_client.get_document_by_id(
        collection="samples", object_id=sample_obj_id
    ):
        raise HTTPException(
            status_code=404, detail="Tried to update a sample which does not exist!"
        )

    # Update sample
    result = await db_client.update(
        collection="samples", model=updates, object_id=sample_obj_id
    )

    if result.matched_count != 1:
        raise HTTPException(status_code=500, detail="Failed to update sample")


async def get_models(
    db_client: MongoDBClient,
    project_id: str,
    model_type: Optional[str] = None,
    status: Optional[
        Literal["queued", "started", "failed", "completed", "aborted"]
    ] = None,
    start: int = 0,
    end: Optional[int] = None,
) -> list[Model]:
    project_obj_id = convert_to_objectid(project_id, "projects")
    filters = {"project_id": project_obj_id}
    if model_type:
        filters["type"] = model_type
    if status:
        filters["training_status"] = status

    if not await db_client.get_document_by_id("projects", project_obj_id):
        raise HTTPException(status_code=404, detail="Project not found with that ID.")

    models = await db_client.get_filtered_documents(
        collection="models",
        filters=filters,
        sort_by="version",
        sort_direction=-1,
        start=start,
        limit=end - start + 1 if end is not None else 0,
    )
    return [Model(**model) for model in models]


async def get_model(
    db_client: MongoDBClient,
    project_id: str,
    model_type: str,
    version: int | None = None,
    status: Literal["queued", "started", "failed", "completed", "aborted"]
    | None = None,
    model_id: str | None = None,
    task_id: str | None = None,
) -> Model:
    project_obj_id = convert_to_objectid(project_id, "projects")
    filters = {"project_id": project_obj_id, "type": model_type}
    if version:
        filters["version"] = version
    if status:
        filters["training_status"] = status
    if model_id:
        filters["_id"] = convert_to_objectid(model_id, "models")
    if task_id:
        filters["task_id"] = task_id

    if not await db_client.get_document_by_id("projects", project_obj_id):
        raise HTTPException(status_code=404, detail="Project not found with that ID.")

    models = await db_client.get_filtered_documents(
        collection="models",
        filters=filters,
        sort_by="version",
        sort_direction=-1,
    )
    if not models:
        raise HTTPException(
            status_code=404,
            detail="No models found of that type for this project!",
        )

    return Model(**models[0])


async def update_model(db_client: MongoDBClient, model_id: str, updates: ModelUpdate):
    model_obj_id = convert_to_objectid(model_id, "models")

    # Check model already exists
    if not await db_client.get_document_by_id(
        collection="models", object_id=model_obj_id
    ):
        raise HTTPException(
            status_code=404, detail="Tried to update a model which does not exist!"
        )

    # Update model
    result = await db_client.update(
        collection="models", model=updates, object_id=model_obj_id
    )

    if result.matched_count != 1:
        raise HTTPException(status_code=500, detail="Failed to update model")


async def add_model(db_client: MongoDBClient, project_id: str, model: ModelIn):
    project_obj_id = convert_to_objectid(project_id, "projects")

    return await db_client.insert(
        collection="models", model=model, ids={"project_id": project_obj_id}
    )


async def delete_model(
    db_client: MongoDBClient, project_id: str, model_id: str
) -> None:
    project_obj_id = convert_to_objectid(project_id, "projects")
    model_obj_id = convert_to_objectid(model_id, "models")

    filters = {"project_id": project_obj_id, "_id": model_obj_id}

    result = await db_client.delete_filtered_documents(
        collection="models", filters=filters
    )

    if result.deleted_count == 0:
        raise HTTPException(
            status_code=404, detail="Model not found belonging to this Project."
        )


async def update_project(
    db_client: MongoDBClient, project_id: str, project: Project
) -> None:
    project_id = convert_to_objectid(project_id, "projects")

    result = await db_client.update("projects", project, project_id)
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Project not found with that ID.")


async def delete_projects(
    db_client: MongoDBClient, project_id: str | None = None
) -> None:
    if project_id:
        project_id = convert_to_objectid(project_id, "projects")

    # Clean up all associated samples
    await db_client.delete_filtered_documents(
        collection="samples", filters={"project_id": project_id} if project_id else {}
    )

    # Clean up all associated annotations
    await db_client.delete_filtered_documents(
        collection="annotations",
        filters={"project_id": project_id} if project_id else {},
    )

    # Clean up all associated models
    await db_client.delete_filtered_documents(
        collection="models",
        filters={"project_id": project_id} if project_id else {},
    )

    # Delete this specific project
    result = await db_client.delete_filtered_documents(
        collection="projects", filters={"_id": project_id} if project_id else {}
    )

    if project_id and result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Project not found with that ID.")


async def get_sample(
    db_client: MongoDBClient, project_id: str, sample_id: str
) -> Sample:
    # Convert project ID to ObhectID
    project_obj_id = convert_to_objectid(project_id, "projects")

    # Get sample with this ID
    sample_obj_id = convert_to_objectid(sample_id, "samples")

    samples = await db_client.get_filtered_documents(
        collection="samples",
        filters={"_id": sample_obj_id, "project_id": project_obj_id},
    )

    if len(samples) == 0:
        raise HTTPException(
            status_code=404,
            detail="Sample not found with that ID belonging to specified Project.",
        )

    return Sample(**samples[0])


async def delete_samples(
    db_client: MongoDBClient, project_id: str, sample_id: str = None
) -> None:
    project_obj_id = convert_to_objectid(project_id, "projects")
    filters = {"project_id": project_obj_id}

    if sample_id:
        sample_obj_id = convert_to_objectid(sample_id, "samples")
        filters["_id"] = sample_obj_id

    result = await db_client.delete_filtered_documents(
        collection="samples", filters=filters
    )

    if result.deleted_count == 0:
        raise HTTPException(
            status_code=404, detail="Sample not found belonging to this Project."
        )


async def get_annotations(
    db_client: MongoDBClient,
    project_id: str,
    sample_id: Optional[str] = None,
    validated: Optional[bool] = None,
    created_by: Optional[str] = None,
    sort_by: str = "_id",
    sort_direction: Literal["ascending", "descending"] = "descending",
    start: int = 0,
    count: Optional[int] = None,
) -> list[AnnotationOutTypes]:
    db_filters = {"project_id": convert_to_objectid(project_id, "projects")}

    if sample_id:
        db_filters["sample_id"] = convert_to_objectid(sample_id, "samples")

    if validated is not None:
        db_filters["validated"] = validated
    if created_by is not None:
        db_filters["created_by"] = created_by

    annotations = await db_client.get_filtered_documents(
        collection="annotations",
        filters=db_filters,
        sort_by=sort_by,
        sort_direction=sort_direction,
        start=start,
        limit=count if count is not None else 0,
    )

    return [AnnotationOutTypeAdapter.validate_python(a) for a in annotations]


async def add_annotations(
    db_client: MongoDBClient,
    project_id: str,
    sample_id: str,
    annotations: list[AnnotationBatchTypes],
) -> list[str]:
    db_ids = {
        "project_id": convert_to_objectid(project_id, "projects"),
        "sample_id": convert_to_objectid(sample_id, "samples"),
    }
    return await db_client.insert_many(
        collection="annotations", models=annotations, ids=db_ids
    )


async def delete_annotations(
    db_client: MongoDBClient,
    project_id: str,
    sample_id: Optional[str] = None,
    annotation_id: Optional[str] = None,
) -> None:
    project_obj_id = convert_to_objectid(project_id, "projects")
    filters = {"project_id": project_obj_id}

    if sample_id:
        sample_obj_id = convert_to_objectid(sample_id, "samples")
        filters["sample_id"] = sample_obj_id

    if annotation_id:
        annotation_obj_id = convert_to_objectid(annotation_id, "annotations")
        filters["_id"] = annotation_obj_id

    result = await db_client.delete_filtered_documents(
        collection="annotations", filters=filters
    )

    return result.deleted_count


async def update_annotations(
    db_client: MongoDBClient,
    project_id: str,
    sample_id: str,
    annotations: list[AnnotationBatchTypes],
    created_by: Optional[str] = None,
) -> list[str]:
    project_obj_id = convert_to_objectid(project_id, "projects")
    sample_obj_id = convert_to_objectid(sample_id, "samples")
    filters: dict = {"project_id": project_obj_id, "sample_id": sample_obj_id}
    if created_by is not None:
        # Scope the delete to only this user's annotations (concurrent-safe)
        filters["created_by"] = created_by
    try:
        await db_client.delete_filtered_documents("annotations", filters)
    except HTTPException:
        pass

    if len(annotations) == 0:
        return []

    return await add_annotations(
        db_client=db_client,
        project_id=project_id,
        sample_id=sample_id,
        annotations=annotations,
    )


async def get_files(dir_path: str, file_type: str) -> list[str]:
    file_names = Path(dir_path).glob(f"*.{file_type}")
    file_names = map(str, file_names)
    file_names = list(sorted(file_names))
    return file_names


async def get_directories(dir_path: str) -> list[str]:
    dir_names = Path(dir_path).glob("*/")
    dir_names = filter(lambda p: p.is_dir(), dir_names)
    dir_names = map(str, dir_names)
    dir_names = list(sorted(dir_names))
    return dir_names


async def filter_directories_by_file_type(
    dir_paths: list[str], file_type: str
) -> list[str]:
    filtered_dirs = []
    for dir_path in dir_paths:
        files = await get_files(dir_path, file_type)
        if files:
            filtered_dirs.append(dir_path)
    return filtered_dirs


async def get_sample_summary(
    db_client: MongoDBClient, project_id: str
) -> SampleSummary:
    samples = await get_samples(db_client, project_id)

    summary = SampleSummary(
        total=len(samples),
        shot_min=min(sample.shot_id for sample in samples) if samples else None,
        shot_max=max(sample.shot_id for sample in samples) if samples else None,
        data=samples[0].data if samples else None,
    )

    if isinstance(summary.data, FileData):
        summary.data.file_name = str(Path(summary.data.file_name).parent)

    return summary


async def import_annotations(
    db_client: MongoDBClient,
    project_id: str,
    annotations: list[AnnotationBatchTypes],
) -> None:
    ids = {
        "project_id": convert_to_objectid(project_id, "projects"),
    }

    if not await db_client.get_document_by_id("projects", ids["project_id"]):
        raise HTTPException(status_code=404, detail="Project not found with that ID.")

    if len(annotations) == 0:
        return

    sample_groups = defaultdict(list)
    for annotation in annotations:
        sample_groups[annotation.shot_id].append(annotation)

    shot_ids = list(sample_groups.keys())
    samples = await db_client.get_filtered_documents(
        collection="samples",
        filters={"project_id": ids["project_id"], "shot_id": {"$in": shot_ids}},
        sort_by="shot_id",
        sort_direction="ascending",
    )

    sample_shot_ids = [sample["shot_id"] for sample in samples]
    for shot_id in shot_ids:
        if shot_id not in sample_shot_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Sample not found with shot ID {shot_id}.",
            )

    for sample in samples:
        sample_id = str(sample["_id"])
        shot_id = sample["shot_id"]
        sample_obj_id = convert_to_objectid(sample_id, "samples")
        sample_annotations: list[AnnotationOutTypes] = sample_groups[shot_id]

        # Set shot_id for each annotation
        for annotation in sample_annotations:
            annotation.sample_id = sample_id
            annotation.shot_id = shot_id

        ids["sample_id"] = sample_obj_id
        await db_client.insert_many(
            collection="annotations", models=sample_annotations, ids=ids
        )

        # If all annotations are validated, mark sample as validated
        if all(ann.validated for ann in sample_annotations):
            await update_sample(
                db_client, sample_id, SampleUpdate(validated_annotations=True)
            )
        # Else mark as unvalidated (if there are any annotations)
        elif sample_annotations:
            await update_sample(
                db_client, sample_id, SampleUpdate(validated_annotations=False)
            )


# ---------------------------------------------------------------------------
# User helpers
# ---------------------------------------------------------------------------


async def get_user_by_username(
    db_client: MongoDBClient, username: str
) -> UserOut | None:
    docs = await db_client.get_filtered_documents(
        "users", filters={"username": username}
    )
    return UserOut.model_validate(docs[0]) if docs else None


async def get_user_by_id(db_client: MongoDBClient, user_id: str) -> UserOut | None:
    obj_id = convert_to_objectid(user_id, "users")
    doc = await db_client.get_document_by_id("users", obj_id)
    return UserOut.model_validate(doc) if doc else None


async def get_all_users(db_client: MongoDBClient) -> list[UserOut]:
    docs = await db_client.get_all_documents("users")
    return [UserOut.model_validate(d) for d in docs]


async def create_user(db_client: MongoDBClient, user: UserIn) -> str:
    existing = await get_user_by_username(db_client, user.username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists")
    return await db_client.insert("users", user)


async def update_user(
    db_client: MongoDBClient, user_id: str, updates: UserUpdate
) -> None:
    obj_id = convert_to_objectid(user_id, "users")
    user = await get_user_by_id(db_client, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    update_dict = updates.model_dump(exclude_none=True)
    if "password" in update_dict:
        update_dict["hashed_password"] = hash_password(update_dict.pop("password"))
    await db_client.db["users"].update_one({"_id": obj_id}, {"$set": update_dict})


async def delete_user(db_client: MongoDBClient, user_id: str) -> None:
    obj_id = convert_to_objectid(user_id, "users")
    result = await db_client.delete_filtered_documents("users", {"_id": obj_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    # Also remove their project memberships
    await db_client.delete_filtered_documents("project_members", {"user_id": obj_id})


# ---------------------------------------------------------------------------
# Project membership helpers
# ---------------------------------------------------------------------------


async def get_project_members(
    db_client: MongoDBClient, project_id: str
) -> list[ProjectMemberOut]:
    project_oid = convert_to_objectid(project_id, "projects")
    docs = await db_client.get_filtered_documents(
        "project_members", filters={"project_id": project_oid}
    )
    result = []
    for doc in docs:
        user = await get_user_by_id(db_client, str(doc["user_id"]))
        doc["username"] = user.username if user else "unknown"
        doc["user_id"] = str(doc["user_id"])
        result.append(ProjectMemberOut.model_validate(doc))
    return result


async def get_project_membership(
    db_client: MongoDBClient, project_id: str, user_id: str
) -> dict | None:
    project_oid = convert_to_objectid(project_id, "projects")
    user_oid = convert_to_objectid(user_id, "users")
    docs = await db_client.get_filtered_documents(
        "project_members",
        filters={"project_id": project_oid, "user_id": user_oid},
    )
    return docs[0] if docs else None


async def add_project_member(
    db_client: MongoDBClient,
    project_id: str,
    user_id: str,
    role: str = "annotator",
) -> str:
    project_oid = convert_to_objectid(project_id, "projects")
    user_oid = convert_to_objectid(user_id, "users")

    existing = await get_project_membership(db_client, project_id, user_id)
    if existing:
        raise HTTPException(
            status_code=409, detail="User is already a member of this project"
        )

    member = ProjectMember(
        project_id=str(project_oid),
        user_id=str(user_oid),
        role=role,
    )
    return await db_client.insert(
        "project_members",
        member,
        ids={"project_id": project_oid, "user_id": user_oid},
    )


async def update_project_member(
    db_client: MongoDBClient,
    project_id: str,
    user_id: str,
    updates: ProjectMemberUpdate,
) -> None:
    project_oid = convert_to_objectid(project_id, "projects")
    user_oid = convert_to_objectid(user_id, "users")

    docs = await db_client.get_filtered_documents(
        "project_members",
        filters={"project_id": project_oid, "user_id": user_oid},
    )
    if not docs:
        raise HTTPException(status_code=404, detail="Membership not found")

    member_oid = convert_to_objectid(str(docs[0]["_id"]), "project_members")
    await db_client.update("project_members", updates, member_oid)


async def remove_project_member(
    db_client: MongoDBClient, project_id: str, user_id: str
) -> None:
    project_oid = convert_to_objectid(project_id, "projects")
    user_oid = convert_to_objectid(user_id, "users")
    result = await db_client.delete_filtered_documents(
        "project_members",
        {"project_id": project_oid, "user_id": user_oid},
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Membership not found")


async def get_user_projects(
    db_client: MongoDBClient,
    user_id: str,
    global_role: str,
    name: Optional[str] = None,
    sort_by: str = "_id",
    sort_direction: str = "descending",
    start: int = 0,
    count: Optional[int] = None,
) -> list[Project]:
    if global_role == "admin":
        return await get_projects(
            db_client,
            name=name,
            sort_by=sort_by,
            sort_direction=sort_direction,
            start=start,
            count=count,
        )

    user_oid = convert_to_objectid(user_id, "users")
    memberships = await db_client.get_filtered_documents(
        "project_members", filters={"user_id": user_oid}
    )
    project_oids = [m["project_id"] for m in memberships]

    if not project_oids:
        return []

    filters: dict = {"_id": {"$in": project_oids}}
    if name:
        filters["name"] = {"$regex": f"{name}", "$options": "i"}

    import pymongo

    _direction = (
        pymongo.ASCENDING if sort_direction == "ascending" else pymongo.DESCENDING
    )
    docs = await db_client.get_filtered_documents(
        "projects",
        filters=filters,
        sort_by=sort_by,
        sort_direction=sort_direction,
        start=start,
        limit=count if count is not None else 0,
    )
    return [Project(**d) for d in docs]
