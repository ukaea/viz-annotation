from toktagger.api.schemas.projects import Project, ProjectIn
from typing import Literal
from fastapi import APIRouter, Request, HTTPException, Query, Path
from toktagger.api.crud import utils
from toktagger.api.core.data_loaders import LoaderRegistry
from toktagger.api.crud.db import MongoDBClient

router = APIRouter(prefix="/projects", tags=["Projects"])


@router.get(
    "",
    responses={
        200: {"description": "Returns a list of available Projects."},
    },
    operation_id="get_projects",
    tags=["MCP"],
)
async def get_projects(
    request: Request,
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
        description="Index of the first project you want returned when sorted by above parameter",
    ),
    count: int | None = Query(
        None,
        description="Number of projects you want returned, leave blank to return all entries",
    ),
    name: str | None = Query(
        None, description="Name of a project to search for, by default None"
    ),
) -> list[Project]:
    """
    Get a list of all available projects.
    -------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Retrieve all projects in the system, with optional sorting, pagination, and filtering by name

    Use When:
        - You need to discover what projects exist and their configurations
        - You need a project _id to use with other endpoints
        - You want to list projects for auditing or summary purposes

    Do Not Use When:
        - You only need a single project - use get_project instead
        - You need project samples - use get_samples instead

    Returns:
        A list of Project objects, each containing: name, task, query_strategy, data_loader, time_min, time_max, shot_labels, time_region_labels, time_point_labels, bounding_box_labels, polygon_labels, video_bounding_box_labels, model_types, _id, timestamp

    Example User Requests:
        - "What projects are available?"
        - "Show me all projects named Disruption"
    """
    projects = await utils.get_projects(
        db_client=request.app.state.db_client,
        name=name,
        sort_by=sort_by,
        sort_direction=sort_direction,
        start=start,
        count=count,
    )

    return projects


@router.post(
    "",
    operation_id="create_project",
    tags=["MCP"],
    responses={
        200: {
            "description": "Project has been created successfully, returning the Project's ID."
        },
    },
)
async def create_project(request: Request, project: ProjectIn):
    """
    Create a new project.
    ---------------------

    MCP Documentation
    -----------------
    Purpose:
        Create a new project with specified task, data loader, query strategy, label sets, and optional model types.
        Users should be prompted for the required parameters in the ProjectIn schema.

    Use When:
        - You are setting up a new annotation workflow for a dataset
        - You are preparing to add samples and annotations

    Do Not Use When:
        - The project already exists and you wish to update it - use update_project instead

    Returns:
        A dict with _id containing the new project's unique identifier

    Example User Requests:
        - "Create a new time-series annotation project"
        - "Set up a video project with UFO bounding box labels"
    """
    # Create instance of this project class, instantiating all required classes for that task, and return its ID
    # In the future, should be able to specify eg dataloader, data type, query strategy etc
    if project.data_loader not in LoaderRegistry.names():
        raise HTTPException(422, detail="Invalid data loader specified.")

    print(project)

    _id = await request.app.state.db_client.insert(collection="projects", model=project)
    return {"_id": _id}


@router.get(
    "/{project_id}",
    operation_id="get_project",
    tags=["MCP"],
    responses={
        200: {"description": "Project has been retrieved successfully."},
        404: {"description": "Project not found with that ID."},
    },
)
async def get_project(
    request: Request,
    project_id: str = Path(description="The ID of the project to return"),
) -> Project:
    """
    Get a single project using its ID.
    -----------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Retrieve the full details of a single project by its ID.

    Use When:
        - You need the complete configuration of a specific project
        - You have a project _id and need its task, labels, and settings
        - You are verifying project configuration before annotating or training

    Do Not Use When:
        - You want to search/list multiple projects - use get_projects instead
        - You need to get a project by name - use get_projects instead
        - You need sample data - use the samples or data endpoints instead

    Returns:
        A Project object.

    Example User Requests:
        - "Show me the configuration for project 6a8f2340b6b4f8d585fd1a67"
        - "What data loader does this project use?"
    """
    # Return information about a specific project
    # Have put project_id as a string for now, but might want to use ShortUUID?
    db_client = request.app.state.db_client
    project = await utils.get_project(db_client, project_id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found with that ID.")

    return project


@router.put(
    "/{project_id}",
    operation_id="update_project",
    tags=["MCP"],
    responses={
        200: {
            "description": "Project has been successfully set as the active project."
        },
        404: {"description": "Project not found with that ID."},
    },
)
async def update_project(
    request: Request,
    project: Project,
    project_id: str = Path(description="The ID of the project to activate"),
):
    """
    Update a project's information.
    -----------------------------

    MCP Documentation
    -----------------
    Purpose:
        Modify an existing project's configuration, including task, labels, time windows, model types, and other settings.

    Use When:
        - You need to change a project's annotation labels after creation
        - You want to add or remove model types from a project
        - You are updating project metadata (time windows, query strategy, etc.)

    Do Not Use When:
        - You are creating a new project - use create_project instead

    Returns:
        None (no response body on success)

    Example User Requests:
        - "Update the label set for this project"
        - "Change the query strategy for this project to sequential"
    """
    db_client: MongoDBClient = request.app.state.db_client
    await utils.update_project(db_client, project_id, project)


@router.delete(
    "/{project_id}",
    operation_id="delete_project",
    responses={
        200: {"description": "Project has been successfully deleted."},
        404: {"description": "Project not found with that ID."},
    },
)
async def delete_project(
    request: Request,
    project_id: str = Path(description="The ID of the project to delete"),
):
    """
    Permanently delete a project.
    -----------------------------

    MCP Documentation
    -----------------
    This endpoint is not exposed to the MCP server.
    """
    db_client = request.app.state.db_client
    # Delete this specific project
    await utils.delete_projects(db_client=db_client, project_id=project_id)


@router.delete(
    "",
    operation_id="delete_all_projects",
    responses={
        200: {"description": "Projects have been successfully deleted."},
    },
)
async def delete_all_projects(
    request: Request,
):
    """
    Remove all projects.
    --------------------

    MCP Documentation
    -----------------
    This endpoint is not exposed to the MCP server.
    """
    db_client = request.app.state.db_client
    # Check project exists
    await utils.delete_projects(db_client=db_client)
