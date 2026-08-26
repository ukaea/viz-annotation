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
    """Update a project's information.
    -----------------------------
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
    """
    db_client = request.app.state.db_client
    # Check project exists
    await utils.delete_projects(db_client=db_client)
