from typing import Literal
from fastapi import APIRouter, Request, Path, Query
from toktagger.api.crud import utils
from toktagger.api.schemas.samples import SampleUpdate
from toktagger.api.schemas.annotations import (
    AnnotationBatchTypes,
    AnnotationOutTypes,
)

router = APIRouter(
    prefix="/projects/{project_id}",
    tags=["Annotations"],
)


@router.get(
    "/annotations",
    operation_id="get_project_annotations",
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
) -> list[AnnotationOutTypes]:
    """
    Retrieve all annotations for this project, subject to specified filters.
    ------------------------------------------------------------------------
    """
    db_client = request.app.state.db_client
    # Check project exists
    await utils.get_project(db_client=db_client, project_id=project_id)

    # Get annotations
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
    operation_id="import_annotations",
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
) -> None:
    """
    Update or add annotations for this project.
    -------------------------------------------
    """
    db_client = request.app.state.db_client
    await utils.import_annotations(db_client, project_id, annotations)


@router.delete(
    "/annotations",
    operation_id="delete_all_annotations",
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
):
    """
    Delete ALL annotations for the given project.
    ---------------------------------------------
    """
    db_client = request.app.state.db_client
    # Check project exists
    await utils.get_project(db_client=db_client, project_id=project_id)
    # Delete all annotations for this project
    await utils.delete_annotations(db_client=db_client, project_id=project_id)


@router.get(
    "/samples/{sample_id}/annotations",
    operation_id="get_sample_annotations",
    response_model=list[AnnotationOutTypes],
    responses={
        200: {"description": "Annotations for this sample deleted successfully."},
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
) -> list[AnnotationOutTypes]:
    # Return annotations available for this project and sample, if any
    # Can filter by params, eg specific camera or frame being returned (or return all annotations for this sample at once and store client side?)
    # Should return whether these are validated as a boolean
    db_client = request.app.state.db_client
    # Check project and sample exist
    await utils.get_project(db_client=db_client, project_id=project_id)
    await utils.get_sample(
        db_client=db_client, project_id=project_id, sample_id=sample_id
    )

    # Get annotations
    annotations = await utils.get_annotations(
        db_client=db_client,
        project_id=project_id,
        sample_id=sample_id,
        validated=validated,
        created_by=created_by,
        sort_by=sort_by,
        sort_direction=sort_direction,
        start=start,
        count=count,
    )

    return annotations


@router.put(
    "/samples/{sample_id}/annotations",
    operation_id="update_sample_annotations",
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
):
    """
    Update the list of annotations to a given sample for a specified project. Will overwrite existing annotations.
    ---------------------------------------------------------------------
    """
    # Add human annotations to this project and sample
    # Again dont know what form this data will take so have set to a Request for now
    # This data could be for one or more events per task, ie multiple ELMs or UFOs per pulse
    # This should be added into the database, with validated=True
    # Delete predictions from model, if they exist, since they are being replaced by human validated ones
    db_client = request.app.state.db_client

    # Check project and sample exist
    await utils.get_project(db_client=db_client, project_id=project_id)
    sample = await utils.get_sample(
        db_client=db_client, project_id=project_id, sample_id=sample_id
    )

    # Set shot_id for each annotation
    for annotation in annotations:
        annotation.shot_id = sample.shot_id

    # Delete previous annotations, if they exist, and add new ones
    result = await utils.update_annotations(
        db_client, project_id, sample_id, annotations
    )

    # Update sample to show that annotations are validated
    if validated or any(annotation.validated for annotation in annotations):
        await utils.update_sample(
            db_client=db_client,
            sample_id=sample_id,
            updates=SampleUpdate(validated_annotations=True),
        )

    return result


@router.delete(
    "/samples/{sample_id}/annotations",
    operation_id="delete_sample_annotations",
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
):
    """
    Delete ALL annotations for a given sample from a given project.
    ---------------------------------------------------------------
    """
    # Remove annotations for this project and sample
    # Probably dont need to be able to specify params here, don't envisage how/why the UI would allow you to remove specific annotations

    db_client = request.app.state.db_client
    # Check project and sample exist
    await utils.get_project(db_client=db_client, project_id=project_id)
    await utils.get_sample(
        db_client=db_client, project_id=project_id, sample_id=sample_id
    )

    # Delete all annotations for this project and sample
    await utils.delete_annotations(
        db_client=db_client, project_id=project_id, sample_id=sample_id
    )
