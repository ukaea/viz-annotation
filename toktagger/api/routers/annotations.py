from typing import Literal
from fastapi import APIRouter, Depends, Request, Path, Query
from toktagger.api.auth.dependencies import (
    get_current_user,
    require_project_annotator,
    require_project_viewer,
    require_project_admin_role,
)
from toktagger.api.crud import utils
from toktagger.api.schemas.samples import SampleUpdate
from toktagger.api.schemas.annotations import (
    AnnotationBatchTypes,
    AnnotationOutTypes,
)
from toktagger.api.schemas.users import UserOut

router = APIRouter(
    prefix="/projects/{project_id}",
    tags=["Annotations"],
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
    sort_by: str = Query("_id"),
    sort_direction: Literal["ascending", "descending"] = Query("descending"),
    start: int = Query(0),
    count: int = Query(None),
    validated: bool = Query(None),
    current_user: UserOut = Depends(require_project_viewer),
) -> list[AnnotationOutTypes]:
    """Retrieve all annotations for this project."""
    db_client = request.app.state.db_client
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
    db_client = request.app.state.db_client
    # Non-admin, non-internal callers must own all annotations they import.
    # Global admins and the internal Ray-worker token bypass this for data migration / predictions.
    if current_user.username != "__internal__" and current_user.global_role != "admin":
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
    current_user: UserOut = Depends(require_project_admin_role),
):
    """Delete ALL annotations for the given project."""
    db_client = request.app.state.db_client
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
    sort_by: str = Query("_id"),
    sort_direction: Literal["ascending", "descending"] = Query("descending"),
    start: int = Query(0),
    count: int = Query(None),
    validated: bool = Query(None),
    created_by: str = Query(None),
    current_user: UserOut = Depends(get_current_user),
) -> list[AnnotationOutTypes]:
    db_client = request.app.state.db_client
    await utils.get_project(db_client=db_client, project_id=project_id)
    await utils.get_sample(
        db_client=db_client, project_id=project_id, sample_id=sample_id
    )

    # Enforce project membership (global admins bypass this check)
    membership = None
    if current_user.global_role != "admin":
        membership = await utils.get_project_membership(
            db_client, project_id, current_user.id
        )
        if membership is None:
            from fastapi import HTTPException as _HTTPException

            raise _HTTPException(
                status_code=403, detail="You are not a member of this project"
            )

    # Apply per-user annotation visibility filter
    effective_created_by = created_by
    if membership and not membership.get("show_others_annotations", True):
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
    validated: bool = Query(None),
    current_user: UserOut = Depends(require_project_annotator),
):
    """Update the annotations for a sample. Replaces only the current user's annotations."""
    db_client = request.app.state.db_client

    await utils.get_project(db_client=db_client, project_id=project_id)
    sample = await utils.get_sample(
        db_client=db_client, project_id=project_id, sample_id=sample_id
    )

    # Server is authoritative for identity
    for annotation in annotations:
        annotation.created_by = current_user.username
        annotation.shot_id = sample.shot_id

    result = await utils.update_annotations(
        db_client,
        project_id,
        sample_id,
        annotations,
        created_by=current_user.username,
    )

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
    current_user: UserOut = Depends(require_project_admin_role),
):
    """Delete ALL annotations for a given sample from a given project."""
    db_client = request.app.state.db_client
    await utils.get_project(db_client=db_client, project_id=project_id)
    await utils.get_sample(
        db_client=db_client, project_id=project_id, sample_id=sample_id
    )
    await utils.delete_annotations(
        db_client=db_client, project_id=project_id, sample_id=sample_id
    )
