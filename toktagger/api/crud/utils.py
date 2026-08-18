from collections import defaultdict
from pathlib import Path
from typing import Literal

from bson import ObjectId
from fastapi import HTTPException
from pydantic import TypeAdapter

from toktagger.api.auth.core import hash_password
from toktagger.api.crud.db import MongoDBClient
from toktagger.api.schemas import convert_to_objectid
from toktagger.api.schemas.annotations import (
    AnnotationBatchTypes,
    AnnotationOutTypeAdapter,
    AnnotationOutTypes,
)
from toktagger.api.schemas.models import Model, ModelIn, ModelUpdate
from toktagger.api.schemas.projects import Project
from toktagger.api.schemas.samples import FileData, Sample, SampleSummary, SampleUpdate
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
    name: str | None = None,
    sort_by: str | None = "_id",
    sort_direction: Literal["ascending", "descending"] | None = "descending",
    start: int | None = 0,
    count: int | None = None,
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
    validated: bool | None = None,
    shot_id: int | None = None,
    sort_by: str = "_id",
    sort_direction: Literal["ascending", "descending"] = "descending",
    start: int = 0,
    count: int | None = None,
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
    model_type: str | None = None,
    status: (
        Literal["queued", "training", "loading", "failed", "completed", "aborted"]
        | None
    ) = None,
    start: int = 0,
    end: int | None = None,
) -> list[Model]:
    project_obj_id = convert_to_objectid(project_id, "projects")
    filters = {"project_id": project_obj_id}
    if model_type:
        filters["type"] = model_type
    if status:
        filters["status"] = status

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
    status: Literal["queued", "training", "loading", "failed", "completed", "aborted"]
    | None = None,
    model_id: str | None = None,
    task_id: str | None = None,
) -> Model:
    project_obj_id = convert_to_objectid(project_id, "projects")
    filters = {"project_id": project_obj_id, "type": model_type}
    if version:
        filters["version"] = version
    if status:
        filters["status"] = status
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

    # Clean up all associated project memberships
    await db_client.delete_filtered_documents(
        collection="project_members",
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
    db_client: MongoDBClient, project_id: str, sample_id: str | None = None
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
    sample_id: str | None = None,
    validated: bool | None = None,
    created_by: str | None = None,
    sort_by: str = "_id",
    sort_direction: Literal["ascending", "descending"] = "descending",
    start: int = 0,
    count: int | None = None,
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
    sample_id: str | None = None,
    annotation_id: str | None = None,
    created_by: str | None = None,
) -> int:
    """Delete annotations matching the given filters, returning how many were removed."""
    project_obj_id = convert_to_objectid(project_id, "projects")
    filters = {"project_id": project_obj_id}

    if sample_id:
        sample_obj_id = convert_to_objectid(sample_id, "samples")
        filters["sample_id"] = sample_obj_id

    if annotation_id:
        annotation_obj_id = convert_to_objectid(annotation_id, "annotations")
        filters["_id"] = annotation_obj_id

    if created_by is not None:
        # Scope the delete to only this user's annotations (concurrent-safe)
        filters["created_by"] = created_by

    result = await db_client.delete_filtered_documents(
        collection="annotations", filters=filters
    )

    return result.deleted_count


async def update_annotations(
    db_client: MongoDBClient,
    project_id: str,
    sample_id: str,
    annotations: list[AnnotationBatchTypes],
    created_by: str | None = None,
) -> list[str]:
    await delete_annotations(
        db_client=db_client,
        project_id=project_id,
        sample_id=sample_id,
        created_by=created_by,
    )

    if len(annotations) == 0:
        return []

    return await add_annotations(
        db_client=db_client,
        project_id=project_id,
        sample_id=sample_id,
        annotations=annotations,
    )


async def update_annotation_by_id(
    db_client: MongoDBClient,
    project_id: str,
    sample_id: str,
    annotation_id: str,
    annotation: AnnotationBatchTypes,
) -> bool:
    """Edit one existing annotation in place, keeping its original author.

    `update_annotations` above replaces annotations by delete-then-reinsert scoped to
    a single `created_by`, so it cannot touch annotations written by someone else (or
    by a model) without deleting work the caller may not even be able to see. An
    annotator editing a colleague's annotation therefore comes through here instead.

    The filter is scoped to project and sample, so an annotation ID belonging to
    another project cannot be reached. Returns False if nothing matched.
    """
    filters = {
        "_id": convert_to_objectid(annotation_id, "annotations"),
        "project_id": convert_to_objectid(project_id, "projects"),
        "sample_id": convert_to_objectid(sample_id, "samples"),
    }

    # Identity is never taken from the request body: `created_by` keeps whatever the
    # database already holds, and the id/project/sample triple is pinned by the filter.
    updates = annotation.model_dump(
        mode="python",
        exclude_unset=True,
        exclude_none=True,
        exclude={"id", "created_by", "project_id", "sample_id"},
    )
    if not updates:
        return True

    result = await db_client.db["annotations"].update_one(filters, {"$set": updates})
    return result.matched_count > 0


async def get_files(dir_path: str, file_type: str) -> list[str]:
    file_names = Path(dir_path).glob(f"*.{file_type}")
    file_names = map(str, file_names)
    file_names = sorted(file_names)
    return file_names


async def get_directories(dir_path: str) -> list[str]:
    dir_names = Path(dir_path).glob("*/")
    dir_names = filter(lambda p: p.is_dir(), dir_names)
    dir_names = map(str, dir_names)
    dir_names = sorted(dir_names)
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


async def get_user_doc_by_username(
    db_client: MongoDBClient, username: str
) -> dict | None:
    """Return the raw user document, including fields UserOut omits (e.g.
    hashed_password). Use this only where those fields are required, such as
    password verification during login."""
    docs = await db_client.get_filtered_documents(
        "users", filters={"username": username}
    )
    return docs[0] if docs else None


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


async def _get_project_membership_doc(
    db_client: MongoDBClient, project_oid: ObjectId, user_oid: ObjectId
) -> dict | None:
    docs = await db_client.get_filtered_documents(
        "project_members",
        filters={"project_id": project_oid, "user_id": user_oid},
    )
    return docs[0] if docs else None


async def _get_user_membership_docs(
    db_client: MongoDBClient, user_oid: ObjectId
) -> list[dict]:
    """Raw membership documents for one user, project_id left as an ObjectId."""
    return await db_client.get_filtered_documents(
        "project_members", filters={"user_id": user_oid}
    )


async def get_user_memberships(
    db_client: MongoDBClient, user_id: str
) -> list[ProjectMember]:
    """Every project membership held by one user.

    Lets a client learn its own role in all its projects with a single request,
    instead of one `get_project_members` call per project.
    """
    docs = await _get_user_membership_docs(
        db_client, convert_to_objectid(user_id, "users")
    )
    for doc in docs:
        doc["user_id"] = str(doc["user_id"])
    return [ProjectMember.model_validate(doc) for doc in docs]


async def get_project_membership(
    db_client: MongoDBClient, project_id: str, user_id: str
) -> ProjectMember | None:
    project_oid = convert_to_objectid(project_id, "projects")
    user_oid = convert_to_objectid(user_id, "users")
    doc = await _get_project_membership_doc(db_client, project_oid, user_oid)
    if doc is None:
        return None
    doc["user_id"] = str(doc["user_id"])
    return ProjectMember.model_validate(doc)


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

    doc = await _get_project_membership_doc(db_client, project_oid, user_oid)
    if doc is None:
        raise HTTPException(status_code=404, detail="Membership not found")

    member_oid = convert_to_objectid(str(doc["_id"]), "project_members")
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
    name: str | None = None,
    sort_by: str = "_id",
    sort_direction: str = "descending",
    start: int = 0,
    count: int | None = None,
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

    memberships = await _get_user_membership_docs(
        db_client, convert_to_objectid(user_id, "users")
    )
    project_oids = [m["project_id"] for m in memberships]

    if not project_oids:
        return []

    filters: dict = {"_id": {"$in": project_oids}}
    if name:
        filters["name"] = {"$regex": f"{name}", "$options": "i"}

    docs = await db_client.get_filtered_documents(
        "projects",
        filters=filters,
        sort_by=sort_by,
        sort_direction=sort_direction,
        start=start,
        limit=count if count is not None else 0,
    )
    return [Project(**d) for d in docs]
