from fastapi import APIRouter, Request, HTTPException, Query, Path, Body
from toktagger.api.core.query_strategy import QUERY_STRATEGIES
from toktagger.api.crud import utils
from toktagger.api.schemas.samples import (
    SampleIn,
    Sample,
    SampleSummary,
    SampleUpdateBatchItem,
)
from toktagger.api.schemas.annotations import Annotation
from toktagger.api.schemas import convert_to_objectid
from typing import Literal
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects/{project_id}/samples", tags=["Samples"])


@router.get(
    "",
    operation_id="get_samples",
    response_model=list[Sample],
    responses={
        200: {"description": "Samples have been retrieved successfully."},
        404: {"description": "Project not found with that ID."},
    },
)
async def get_samples(
    request: Request,
    project_id: str = Path(description="The ID of the project to get samples for."),
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
        description="Index of the first sample you want returned when sorted by above parameter",
    ),
    count: int = Query(
        None,
        description="The number of samples to return, leave blank to return all entries",
    ),
    shot_id: int | None = Query(
        None, description="The shot ID to search for, by default None"
    ),
) -> list[Sample]:
    """
    Get the full list of samples available for this project.
    --------------------------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Retrieve all samples for a project, with optional pagination, sorting, and shot ID filtering.

    Use When:
        - You need a complete inventory of samples in a project
        - You want to find a sample by its shot ID
        - You need sample metadata (shot_id, data, validated status) without fetching data content

    Do Not Use When:
        - You need the next sample to annotate - use get_next_sample instead
        - You need actual diagnostic signal/data values - use get_sample_data instead
        - You only need summary info - use get_sample_summary instead

    Returns:
        A list of Sample objects

    Example User Requests:
        - "Show me all samples in this project"
        - "Has shot 30421 been validated?"
        - "What is the sample ID for shot 30421?"

    """
    db_client = request.app.state.db_client
    samples = await utils.get_samples(
        db_client=db_client,
        project_id=project_id,
        shot_id=shot_id,
        sort_by=sort_by,
        sort_direction=sort_direction,
        start=start,
        count=count,
    )
    return samples


@router.post(
    "",
    operation_id="add_samples",
    responses={
        200: {
            "description": "Samples have been added successfully, and a list of their IDs has been returned."
        },
        404: {"description": "Project not found with that ID."},
    },
)
async def add_samples(
    request: Request,
    samples: list[SampleIn],
    project_id: str = Path(
        description="The project ID to associate these samples with."
    ),
):
    """
    Add a list of samples (with optional annotations) to this project.
    ------------------------------------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Insert one or more new samples into a project, optionally with initial annotations.

    Use When:
        - You are setting up a new project with its initial dataset
        - You are bulk-importing samples from a data source or external list
        - You want to add samples and pre-populate them with human annotations

    Do Not Use When:
        - You are updating existing samples - use update_samples instead
        - The project does not exist - verify with get_projects first

    Returns:
        A list of sample _id strings for the newly created samples

    Example User Requests:
        - "Add shots 30400 to 30500 from UDA to the project"
        - "Add samples from my local directory of files at /path/to/my/files"
    """
    # Add samples from the range specified to the project
    # I'm assuming these will be shot/pulse numbers, hence int, but could be unique ID strings instead
    # Depends if for us a 'sample' will always be a shot/pulse, or if it could be a subset eg a single frame of video
    # Do we also want to allow a single value, or list of specific value?
    project_obj_id = convert_to_objectid(project_id, "projects")
    if not await request.app.state.db_client.get_document_by_id(
        "projects", project_obj_id
    ):
        raise HTTPException(status_code=404, detail="Project not found with that ID.")

    # Remove annotations (if they exist), these will be added later
    all_annotations = [sample.annotations for sample in samples]

    # Insert new samples
    ids = await request.app.state.db_client.insert_many(
        collection="samples", models=samples, ids={"project_id": project_obj_id}
    )

    all_ids = [
        {
            "project_id": project_obj_id,
            "sample_id": convert_to_objectid(sample_id, "samples"),
        }
        for sample_id in ids
    ]

    annotations = []
    annotation_ids = []
    for _ann_list, _id in zip(all_annotations, all_ids):
        if _ann_list is not None:
            _ids = [_id for item in _ann_list]
            annotations.extend(_ann_list)
            annotation_ids.extend(_ids)

    # If there are any annotations provided, insert new annotations
    if annotations:
        await request.app.state.db_client.insert_many(
            collection="annotations", models=list(annotations), ids=list(annotation_ids)
        )

    # If a project has been set, update data pool
    if request.app.state.project:
        # Update the query strategy with the new list of samples that can be considered
        # Get all samples which can be considered - sort by shot ID
        samples = await request.app.state.db_client.get_filtered_documents(
            collection="samples",
            filters={"project_id": project_obj_id},
            sort_by="shot_id",
            sort_direction=1,
        )

        # Then get all non-validated annotations for these samples, sorted by uncertainty:
        non_validated_annotations = (
            await request.app.state.db_client.get_filtered_documents(
                collection="annotations",
                filters={"project_id": project_obj_id, "validated": False},
                sort_by="uncertainty",
                sort_direction=1,
            )
        )
        validated_annotations = (
            await request.app.state.db_client.get_filtered_documents(
                collection="annotations",
                filters={"project_id": project_obj_id, "validated": False},
            )
        )
        validated_sample_ids = [
            validated_annotation["sample_id"]
            for validated_annotation in validated_annotations
        ]

        # Update query strategy in the app state with these
        request.app.state.data_pool.query_strategy.samples = [
            Sample.model_validate(sample)
            for sample in samples
            if sample["_id"] not in validated_sample_ids
        ]
        request.app.state.data_pool.query_strategy.annotations = [
            Annotation.model_validate(annotation)
            for annotation in non_validated_annotations
        ]

    return ids


@router.put(
    "",
    operation_id="update_samples",
    responses={
        200: {"description": "Samples have been updated successfully."},
        404: {"description": "Project or Sample(s) not found with that ID."},
    },
)
async def update_samples(
    request: Request,
    sample_batch: list[SampleUpdateBatchItem],
    project_id: str = Path(
        description="The project ID to associate these samples with."
    ),
):
    """
    Update a list of samples (provided with their IDs) for this project.
    ---------------------------------------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Batch-update properties validation status for multiple samples in a project.

    Use When:
        - You need to mark sample(s) as validated after human annotation

    Do Not Use When:
        - You are creating new samples - use add_samples instead

    Returns:
        None (no response body on success)

    Example User Requests:
        - "Mark these samples as validated"
    """
    db_client = request.app.state.db_client
    await utils.get_project(db_client, project_id)

    for sample_batch_item in sample_batch:
        await utils.update_sample(
            db_client=db_client,
            sample_id=sample_batch_item.id,
            updates=sample_batch_item.updates,
        )


@router.post(
    "/next",
    operation_id="get_next_sample",
    response_model=Sample,
    responses={
        200: {
            "description": "The next sample to annotate has been returned, according to the project's query strategy."
        },
        204: {
            "description": "No more samples are available to annotate for this project."
        },
        409: {"description": "Server is not setup to use the selected project."},
    },
)
async def get_next_sample(
    request: Request,
    project_id: str = Path(description="The project to return the next sample from."),
    visited_sample_ids: list[str] = Body(
        ..., description="The IDs of the samples already seen in this session."
    ),
    sort_by: str = Query(
        "shot_id",
        description="Field to sort responses by, by default '_id' (equivalent to timestamp)",
    ),
    sort_direction: Literal["ascending", "descending"] = Query(
        "ascending",
        description="Direction to sort responses, by default 'descending'",
    ),
) -> Sample:
    """
    Get the next sample to annotate for this project, according to query strategy.
    ------------------------------------------------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Get the next unannotated sample for a project based on its query strategy (random, sequential, etc.), with optional sorting.
        Note that a list of previously visited sample IDs should be stored and provided here to prevent duplicates.

    Use When:
        - You are building an annotation workflow and need the next sample to annotate
        - You want the system to pick samples according to the project's query strategy
        - You are iterating through all samples in a project for annotation

    Do Not Use When:
        - You need all samples at once - use get_samples instead
        - You already know the sample you want - use get_sample instead
        - You need actual diagnostic data from the sample - use get_sample_data instead

    Returns:
        A Sample object, or 204 if no samples remain

    Example User Requests:
        - "What's the next sample I need to annotate?"
        - "Give me the next unvalidated sample in this project"
    """
    # Return the next sample for human validation for this project
    # Should use the query strategy, which access the database to determine the next sample to annotate
    # This should then be passed in to the /data endpoint to get required data for visualisation
    # And the /annotation endpoint to get initial prediction (if available)
    db_client = request.app.state.db_client
    project = await utils.get_project(db_client, project_id)
    samples = await utils.get_samples(
        db_client,
        project_id,
        sort_by=sort_by,
        sort_direction=sort_direction,
    )
    annotations = await utils.get_annotations(db_client, project_id)
    query_strategy = QUERY_STRATEGIES[project.query_strategy](samples, annotations)

    try:
        sample = query_strategy.get_next_sample(visited_sample_ids)
    except RuntimeError as e:
        raise HTTPException(status_code=204, detail="No next sample available!") from e

    return sample


@router.get("/summary", operation_id="get_sample_summary")
async def get_sample_summary(
    request: Request,
    project_id: str = Path(
        description="The ID of the project to get a summary of samples from."
    ),
) -> SampleSummary:
    """
    Get a summary of samples for this project.

    This includes total number of samples, min and max shot IDs, and sample data type.

    MCP Documentation
    -----------------
    Purpose:
        Get aggregate statistics about samples in a project without returning the full sample list.

    Use When:
        - You need a quick count of samples in a project
        - You want to check the shot ID range
        - You need sample data type information for a project overview

    Do Not Use When:
        - You need individual sample details - use get_samples or get_sample instead
        - You need sample diagnostic data - use get_sample_data instead

    Returns:
        A SampleSummary object with total_samples, min_shot_id, max_shot_id, data_type

    Example User Requests:
        - "How many samples does this project have?"
        - "Show me the shot ID range for this project"
    """
    db_client = request.app.state.db_client
    summary = await utils.get_sample_summary(db_client, project_id)
    return summary


@router.get(
    "/{sample_id}",
    operation_id="get_sample",
    response_model=Sample,
    responses={
        200: {"description": "Samples has been returned successfully."},
        404: {"description": "Either Project or Sample not found with specified ID."},
    },
)
async def get_sample(
    request: Request,
    project_id: str = Path(
        description="The ID of the project to retrieve a sample from."
    ),
    sample_id: str = Path(description="The ID of the sample to retrieve."),
) -> Sample:
    """
    Get the specified sample from this project.
    --------------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Retrieve the metadata for a single sample by its ID within a project.

    Use When:
        - You need details about a specific sample (shot_id, data reference, validation status)
        - You have a sample _id and want to verify it exists
        - You need sample metadata before fetching its data content

    Do Not Use When:
        - You need the actual signal/data values - use get_sample_data instead
        - You need all samples - use get_samples instead

    Returns:
        A Sample object

    Example User Requests:
        - "Show me the details for sample 6a8f2340b6b4f8d585fd1a6d"
        - "What is the shot ID for this sample?"
    """
    db_client = request.app.state.db_client
    # Check project exists
    project = await utils.get_project(db_client, project_id)
    # Get specified sample
    sample = await utils.get_sample(db_client, project.id, sample_id)
    return sample


@router.delete(
    "/{sample_id}",
    operation_id="remove_sample",
    responses={
        200: {"description": "Samples has been returned successfully."},
        404: {
            "description": "Sample not found with specified ID within selected project."
        },
    },
)
async def remove_sample(
    request: Request,
    project_id: str = Path(
        description="The ID of the project to delete a sample from."
    ),
    sample_id: str = Path(description="The ID of the sample to delete."),
):
    """
    Get the specified sample from this project.
    --------------------------------------------

    MCP Documentation
    -----------------
    This endpoint is not exposed to the MCP server.
    """
    # Remove samples from the project
    # Dont envisage this actually deleting the data stored about these samples
    # But do we need a separate method for that?
    db_client = request.app.state.db_client
    # Check project exists
    await utils.get_project(db_client, project_id=project_id)

    # Delete sample
    await utils.delete_samples(db_client, project_id=project_id, sample_id=sample_id)

    # Delete any annotations associated with this sample
    await utils.delete_annotations(
        db_client, project_id=project_id, sample_id=sample_id
    )


@router.delete("", operation_id="remove_all_samples")
async def remove_all_samples(
    request: Request,
    project_id: str = Path(
        description="The ID of the project to delete all samples from."
    ),
):
    """
    Remove all samples from this project.
    --------------------------------------------

    MCP Documentation
    -----------------
    This endpoint is not exposed to the MCP server.
    """
    db_client = request.app.state.db_client
    # Check project exists
    await utils.get_project(db_client, project_id=project_id)

    # Delete all samples for this project
    await utils.delete_samples(db_client, project_id=project_id)

    # Delete any annotations associated with these samples
    await utils.delete_annotations(db_client, project_id=project_id)
