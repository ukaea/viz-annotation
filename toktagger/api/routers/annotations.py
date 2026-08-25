from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request

from toktagger.api.auth.dependencies import (
    get_current_user,
    require_project_annotator,
    require_project_viewer,
)
from toktagger.api.crud import utils
from toktagger.api.crud.db import MongoDBClient
from toktagger.api.schemas.annotations import (
    RESERVED_CREATED_BY_PREFIXES,
    AnnotationBatchTypes,
    AnnotationOutTypes,
)
from toktagger.api.schemas.samples import SampleUpdate
from toktagger.api.schemas.users import UserOut

router = APIRouter(
    prefix="/projects/{project_id}",
    tags=["Annotations"],
    dependencies=[Depends(get_current_user)],
)


@router.get(
    "/annotations",
    response_model=list[AnnotationOutTypes],
    responses={
        200: {"description": "Annotations for this project returned successfully."},
        404: {"description": "Project not found with that ID."},
    },
)
async def get_all_annotations(
    request: Request,
    project_id: str = Path(
        description="The ID of the project to retrieve annotations for"
    ),
    sort_by: str = Query(
        "_id",
        description="Field to sort responses by, by default '_id' (equivalent to timestamp)",
    ),
    sort_direction: Literal["ascending", "descending"] = Query(
        "descending",
        description="Direction to sort responses, by default 'descending'",
    ),
    start: int = Query(
        0,
        description="Index of the first annotation you want returned when sorted by above parameter",
    ),
    count: int = Query(
        None,
        description="The number of annotations to return, leave blank to return all entries",
    ),
    validated: bool = Query(
        None,
        description="Whether to return only validated or unvalidated annotations, leave blank for all annotations",
    ),
    current_user: UserOut = Depends(require_project_viewer),
) -> list[AnnotationOutTypes]:
    """Retrieve all annotations for this project."""
    db_client: MongoDBClient = request.app.state.db_client
    await utils.get_project(db_client=db_client, project_id=project_id)

    annotations = await utils.get_annotations(
        db_client=db_client,
        project_id=project_id,
        validated=validated,
        sort_by=sort_by,
        sort_direction=sort_direction,
        start=start,
        count=count,
    )
    return annotations


@router.put(
    "/annotations",
    responses={
        200: {"description": "Annotations for this project updated successfully."},
        404: {"description": "Project not found with that ID."},
        422: {"description": "Invalid annotation data provided."},
    },
)
async def import_annotations(
    request: Request,
    annotations: list[AnnotationBatchTypes],
    project_id: str = Path(
        description="The ID of the project to update annotations for"
    ),
    current_user: UserOut = Depends(require_project_annotator),
) -> None:
    """Update or add annotations for this project."""
    db_client: MongoDBClient = request.app.state.db_client
    # Every human caller — admins included — is recorded as the author of the
    # annotations they import, so authorship is always auditable. Only the internal
    # Ray-worker user may supply its own created_by (e.g. "model::<type>" predictions).
    if current_user.username != "__internal__":
        for annotation in annotations:
            annotation.created_by = current_user.username
    await utils.import_annotations(db_client, project_id, annotations)


@router.delete(
    "/annotations",
    responses={
        200: {"description": "Annotations for this project deleted successfully."},
        404: {"description": "Project not found with that ID."},
    },
)
async def delete_all_annotations(
    request: Request,
    project_id: str = Path(
        description="The ID of the project to delete all annotations for"
    ),
    current_user: UserOut = Depends(require_project_annotator),
):
    """Delete ALL annotations for the given project."""
    db_client: MongoDBClient = request.app.state.db_client
    await utils.get_project(db_client=db_client, project_id=project_id)
    await utils.delete_annotations(db_client=db_client, project_id=project_id)


@router.get(
    "/samples/{sample_id}/annotations",
    response_model=list[AnnotationOutTypes],
    responses={
        200: {"description": "Annotations for this sample returned successfully."},
        404: {"description": "Project or Sample not found with that ID."},
    },
)
async def get_annotations(
    request: Request,
    project_id: str = Path(description="The ID of the project to get samples from."),
    sample_id: str = Path(description="The ID of the sample to get annotations from."),
    sort_by: str = Query(
        "_id",
        description="Field to sort responses by, by default '_id' (equivalent to timestamp)",
    ),
    sort_direction: Literal["ascending", "descending"] = Query(
        "descending",
        description="Direction to sort responses, by default 'descending'",
    ),
    start: int = Query(
        0,
        description="Index of the first annotation you want returned when sorted newest - oldest",
    ),
    count: int = Query(
        None,
        description="The number of annotations to return, leave blank to return all entries",
    ),
    validated: bool = Query(
        None,
        description="Whether to return only validated or unvalidated annotations, leave blank for all annotations",
    ),
    created_by: str = Query(
        None,
        description="Whether to only return annotations created by a specific model or by a human.",
    ),
    current_user: UserOut = Depends(require_project_viewer),
) -> list[AnnotationOutTypes]:
    db_client: MongoDBClient = request.app.state.db_client
    await utils.get_project(db_client=db_client, project_id=project_id)
    await utils.get_sample(
        db_client=db_client, project_id=project_id, sample_id=sample_id
    )

    # Membership already validated by require_project_viewer; re-fetch only to check
    # the per-user show_others_annotations preference. Fetched for admins too, since an
    # admin who is also an explicit project member has their own preference to honor;
    # non-member admins simply get None back and fall through to seeing everything.
    membership = await utils.get_project_membership(
        db_client, project_id, current_user.id
    )

    # Apply per-user annotation visibility filter
    effective_created_by = created_by
    if membership and not membership.show_others_annotations:
        # Only show the current user's own annotations
        effective_created_by = current_user.username

    annotations = await utils.get_annotations(
        db_client=db_client,
        project_id=project_id,
        sample_id=sample_id,
        validated=validated,
        created_by=effective_created_by,
        sort_by=sort_by,
        sort_direction=sort_direction,
        start=start,
        count=count,
    )
    return annotations


@router.put(
    "/samples/{sample_id}/annotations",
    responses={
        200: {"description": "Annotations for this sample updated successfully."},
        404: {"description": "Project or Sample not found with that ID."},
    },
)
async def update_annotations(
    request: Request,
    annotations: list[AnnotationBatchTypes],
    project_id: str = Path(
        description="The ID of the project to update annotations for."
    ),
    sample_id: str = Path(
        description="The ID of the sample to update annotations for."
    ),
    validated: bool = Query(
        None,
        description="Whether to set sample to validated (useful if no annotations present).",
    ),
    current_user: UserOut = Depends(require_project_annotator),
) -> list[str]:
    """Update the annotations for a sample.

    The caller's own annotations are replaced wholesale. Annotations belonging to
    another user or to a model are edited in place instead, keeping their author, so
    a client that cannot see them (show_others_annotations=false) can never delete them.
    """
    db_client: MongoDBClient = request.app.state.db_client

    await utils.get_project(db_client=db_client, project_id=project_id)
    sample = await utils.get_sample(
        db_client=db_client, project_id=project_id, sample_id=sample_id
    )

    is_internal = current_user.username == "__internal__"
    owned_annotations = []
    edited_ids = []
    for annotation in annotations:
        is_other_authors = (
            annotation.id is not None
            and not is_internal
            and annotation.created_by != current_user.username
        )
        if is_other_authors:
            # The replace step below is scoped to the caller's own created_by, so
            # re-saving another author's annotation there would duplicate it under
            # the caller's name.
            if await utils.update_annotation_by_id(
                db_client=db_client,
                project_id=project_id,
                sample_id=sample_id,
                annotation_id=annotation.id,
                annotation=annotation,
            ):
                edited_ids.append(annotation.id)
            continue

        if annotation.id is None and not is_internal:
            # A just-run model prediction or annotator suggestion keeps its synthetic
            # author; otherwise the server is authoritative for identity.
            if not (annotation.created_by or "").startswith(
                RESERVED_CREATED_BY_PREFIXES
            ):
                annotation.created_by = current_user.username

        annotation.shot_id = sample.shot_id
        owned_annotations.append(annotation)

    result = await utils.update_annotations(
        db_client,
        project_id,
        sample_id,
        owned_annotations,
        created_by=current_user.username,
    )
    result.extend(edited_ids)

    if validated or any(annotation.validated for annotation in annotations):
        await utils.update_sample(
            db_client=db_client,
            sample_id=sample_id,
            updates=SampleUpdate(validated_annotations=True),
        )

    return result


@router.delete(
    "/samples/{sample_id}/annotations",
    responses={
        200: {"description": "Annotations for this project deleted successfully."},
        404: {"description": "Project not found with that ID."},
    },
)
async def remove_annotations(
    request: Request,
    project_id: str = Path(description="The ID of the project to delete samples from."),
    sample_id: str = Path(
        description="The ID of the sample to delete annotations from."
    ),
    current_user: UserOut = Depends(require_project_annotator),
):
    """Delete ALL annotations for a given sample from a given project."""
    db_client: MongoDBClient = request.app.state.db_client
    await utils.get_project(db_client=db_client, project_id=project_id)
    await utils.get_sample(
        db_client=db_client, project_id=project_id, sample_id=sample_id
    )
    await utils.delete_annotations(
        db_client=db_client, project_id=project_id, sample_id=sample_id
    )


@router.delete(
    "/samples/{sample_id}/annotations/{annotation_id}",
    responses={
        200: {"description": "Annotation deleted successfully."},
        404: {"description": "Project, Sample or Annotation not found with that ID."},
    },
)
async def remove_annotation(
    request: Request,
    project_id: str = Path(description="The ID of the project to delete from."),
    sample_id: str = Path(
        description="The ID of the sample to delete an annotation from."
    ),
    annotation_id: str = Path(description="The ID of the annotation to delete."),
    current_user: UserOut = Depends(require_project_annotator),
):
    """Delete a single annotation, whoever created it.

    The batch save (PUT above) carries no "deleted ids" signal and its replace step is
    scoped to the caller's own annotations, so removing someone else's annotation --
    or a model's prediction -- has to be an explicit call.
    """
    db_client: MongoDBClient = request.app.state.db_client
    await utils.get_project(db_client=db_client, project_id=project_id)
    await utils.get_sample(
        db_client=db_client, project_id=project_id, sample_id=sample_id
    )
    deleted = await utils.delete_annotations(
        db_client=db_client,
        project_id=project_id,
        sample_id=sample_id,
        annotation_id=annotation_id,
    )
    if not deleted:
        raise HTTPException(
            status_code=404, detail="Annotation not found for that project and sample."
        )
