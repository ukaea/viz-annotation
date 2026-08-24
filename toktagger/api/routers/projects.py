from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request

from toktagger.api.auth.dependencies import (
    get_current_user,
    require_global_admin,
    require_project_annotator,
    require_project_viewer,
    require_project_admin_role,
)
from toktagger.api.core.data_loaders import LoaderRegistry
from toktagger.api.crud import utils
from toktagger.api.crud.db import MongoDBClient
from toktagger.api.schemas.projects import (
    Project,
    ProjectIn,
    ProjectMemberOut,
    ProjectMemberCreate,
    ProjectMemberUpdate,
)
from toktagger.api.schemas.users import UserOut

router = APIRouter(
    prefix="/projects",
    tags=["Projects"],
    dependencies=[Depends(get_current_user)],
)


@router.get(
    "", responses={200: {"description": "Returns a list of available Projects."}}
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
    current_user: UserOut = Depends(require_project_annotator),
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
    current_user: UserOut = Depends(require_project_annotator),
):
    """Permanently delete a project."""
    db_client = request.app.state.db_client
    await utils.delete_projects(db_client=db_client, project_id=project_id)


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


@router.get(
    "/{project_id}/members",
    response_model=list[ProjectMemberOut],
)
async def list_project_members(
    request: Request,
    project_id: str = Path(...),
    current_user: UserOut = Depends(get_current_user),
):
    db_client = request.app.state.db_client
    if current_user.global_role != "admin":
        membership = await utils.get_project_membership(
            db_client, project_id, current_user.id
        )
        if not membership:
            raise HTTPException(status_code=403, detail="Not a member of this project")
    return await utils.get_project_members(db_client, project_id)


@router.post("/{project_id}/members", response_model=dict)
async def add_project_member(
    request: Request,
    body: ProjectMemberCreate,
    project_id: str = Path(...),
    current_user: UserOut = Depends(require_project_admin_role),
):
    db_client = request.app.state.db_client
    user = await utils.get_user_by_username(db_client, body.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    member_id = await utils.add_project_member(
        db_client, project_id, user.id, body.role
    )
    return {"_id": member_id}


@router.put("/{project_id}/members/{user_id}")
async def update_project_member(
    request: Request,
    body: ProjectMemberUpdate,
    project_id: str = Path(...),
    user_id: str = Path(...),
    current_user: UserOut = Depends(get_current_user),
):
    db_client = request.app.state.db_client

    # Project admins can change any member. A non-admin may edit only their own
    # membership, and then only its preferences -- `role` always needs project admin.
    # Without the `body.role` clause a viewer could PUT their own membership with
    # {"role": "admin"} and self-promote (mirrors the field-level guard in update_user).
    is_self = current_user.id == user_id
    if (not is_self or body.role is not None) and current_user.global_role != "admin":
        membership = await utils.get_project_membership(
            db_client, project_id, current_user.id
        )
        if not membership or membership.role != "admin":
            raise HTTPException(status_code=403, detail="Project admin access required")

    await utils.update_project_member(db_client, project_id, user_id, body)


@router.delete("/{project_id}/members/{user_id}")
async def remove_project_member(
    request: Request,
    project_id: str = Path(...),
    user_id: str = Path(...),
    current_user: UserOut = Depends(require_project_admin_role),
):
    await utils.remove_project_member(request.app.state.db_client, project_id, user_id)
