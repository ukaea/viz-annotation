from typing import Literal
from fastapi import APIRouter, Depends, Request, HTTPException, Query, Path
from toktagger.api.auth.dependencies import (
    get_current_user,
    require_global_admin,
    require_project_viewer,
    require_project_admin_role,
)
from toktagger.api.crud import utils
from toktagger.api.core.data_loaders import LoaderRegistry
from toktagger.api.crud.db import MongoDBClient
from toktagger.api.schemas import convert_to_objectid
from toktagger.api.schemas.projects import Project, ProjectIn
from toktagger.api.schemas.users import UserOut

router = APIRouter(prefix="/projects", tags=["Projects"])


@router.get(
    "", responses={200: {"description": "Returns a list of available Projects."}}
)
async def get_projects(
    request: Request,
    sort_by: str = Query("_id"),
    sort_direction: Literal["ascending", "descending"] = Query("descending"),
    start: int = Query(0),
    count: int | None = Query(None),
    name: str | None = Query(None),
    current_user: UserOut = Depends(get_current_user),
) -> list[Project]:
    """Get a list of projects visible to the current user."""
    return await utils.get_user_projects(
        db_client=request.app.state.db_client,
        user_id=current_user.id,
        global_role=current_user.global_role,
        name=name,
        sort_by=sort_by,
        sort_direction=sort_direction,
        start=start,
        count=count,
    )


@router.post("", responses={200: {"description": "Project created successfully."}})
async def create_project(
    request: Request,
    project: ProjectIn,
    current_user: UserOut = Depends(get_current_user),
):
    """Create a new project and auto-add the creator as project admin."""
    if project.data_loader not in LoaderRegistry.names():
        raise HTTPException(422, detail="Invalid data loader specified.")

    db_client: MongoDBClient = request.app.state.db_client
    project_id = await db_client.insert(collection="projects", model=project)

    # Auto-add creator as project admin
    await utils.add_project_member(db_client, project_id, current_user.id, role="admin")

    return {"_id": project_id}


@router.get(
    "/{project_id}",
    responses={
        200: {"description": "Project retrieved successfully."},
        404: {"description": "Project not found with that ID."},
    },
)
async def get_project(
    request: Request,
    project_id: str = Path(description="The ID of the project to return"),
    current_user: UserOut = Depends(require_project_viewer),
) -> Project:
    """Get a single project using its ID."""
    return await utils.get_project(request.app.state.db_client, project_id)


@router.put(
    "/{project_id}",
    responses={
        200: {"description": "Project updated successfully."},
        404: {"description": "Project not found with that ID."},
    },
)
async def update_project(
    request: Request,
    project: Project,
    project_id: str = Path(description="The ID of the project to update"),
    current_user: UserOut = Depends(require_project_admin_role),
):
    """Update a project's information."""
    await utils.update_project(request.app.state.db_client, project_id, project)


@router.delete(
    "/{project_id}",
    responses={
        200: {"description": "Project deleted successfully."},
        404: {"description": "Project not found with that ID."},
    },
)
async def delete_project(
    request: Request,
    project_id: str = Path(description="The ID of the project to delete"),
    current_user: UserOut = Depends(require_project_admin_role),
):
    """Permanently delete a project."""
    db_client = request.app.state.db_client
    await utils.delete_projects(db_client=db_client, project_id=project_id)
    project_oid = convert_to_objectid(project_id, "projects")
    await db_client.delete_filtered_documents(
        "project_members", {"project_id": project_oid}
    )


@router.delete(
    "",
    responses={
        200: {"description": "Projects have been successfully deleted."},
    },
)
async def delete_all_projects(
    request: Request,
    _: UserOut = Depends(require_global_admin),
):
    """Remove all projects."""
    await utils.delete_projects(db_client=request.app.state.db_client)
