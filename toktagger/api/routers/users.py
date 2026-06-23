from fastapi import APIRouter, Depends, HTTPException, Path, Request

from toktagger.api.auth.core import hash_password
from toktagger.api.auth.dependencies import (
    get_current_user,
    require_global_admin,
    require_project_admin_role,
)
from toktagger.api.crud import utils
from toktagger.api.schemas.users import (
    ProjectMemberCreate,
    ProjectMemberOut,
    ProjectMemberUpdate,
    UserCreate,
    UserIn,
    UserOut,
    UserUpdate,
)

router = APIRouter(tags=["Users"])


# ---------------------------------------------------------------------------
# Global user management
# ---------------------------------------------------------------------------


@router.get("/users", response_model=list[UserOut])
async def list_users(
    request: Request,
    _: UserOut = Depends(require_global_admin),
):
    return await utils.get_all_users(request.app.state.db_client)


@router.post("/users", response_model=dict)
async def create_user(
    request: Request,
    body: UserCreate,
    _: UserOut = Depends(require_global_admin),
):
    # Reserved prefixes protect the internal worker namespace and model annotation namespace.
    # NOTE: model predictions currently don't set created_by="model::<name>" — this is a known
    # gap that should be addressed in the models sender.
    if body.username.startswith("model::") or body.username.startswith("__"):
        raise HTTPException(status_code=422, detail="Username uses a reserved prefix")
    user = UserIn(
        username=body.username,
        hashed_password=hash_password(body.password),
        email=body.email,
        global_role=body.global_role,
    )
    user_id = await utils.create_user(request.app.state.db_client, user)
    return {"_id": user_id}


@router.get("/users/{user_id}", response_model=UserOut)
async def get_user(
    request: Request,
    user_id: str = Path(...),
    current_user: UserOut = Depends(get_current_user),
):
    if current_user.global_role != "admin" and current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    user = await utils.get_user_by_id(request.app.state.db_client, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.put("/users/{user_id}")
async def update_user(
    request: Request,
    body: UserUpdate,
    user_id: str = Path(...),
    current_user: UserOut = Depends(get_current_user),
):
    if current_user.global_role != "admin" and current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    db_client = request.app.state.db_client

    # Prevent demoting or deactivating the last active admin
    if body.global_role == "user" or body.is_active is False:
        all_users = await utils.get_all_users(db_client)
        remaining_admins = [
            u
            for u in all_users
            if u.global_role == "admin" and u.is_active and u.id != user_id
        ]
        if not remaining_admins:
            raise HTTPException(
                status_code=422,
                detail="Cannot demote or deactivate the last active admin account",
            )

    await utils.update_user(db_client, user_id, body)


@router.delete("/users/{user_id}")
async def delete_user(
    request: Request,
    user_id: str = Path(...),
    _: UserOut = Depends(require_global_admin),
):
    await utils.delete_user(request.app.state.db_client, user_id)


# ---------------------------------------------------------------------------
# Project membership management
# ---------------------------------------------------------------------------


@router.get(
    "/projects/{project_id}/members",
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


@router.post("/projects/{project_id}/members", response_model=dict)
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


@router.put("/projects/{project_id}/members/{user_id}")
async def update_project_member(
    request: Request,
    body: ProjectMemberUpdate,
    project_id: str = Path(...),
    user_id: str = Path(...),
    current_user: UserOut = Depends(get_current_user),
):
    db_client = request.app.state.db_client

    # Project admins can change any member; non-admins may only update their own preferences
    if current_user.id != user_id and current_user.global_role != "admin":
        membership = await utils.get_project_membership(
            db_client, project_id, current_user.id
        )
        if not membership or membership.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Project admin access required")

    await utils.update_project_member(db_client, project_id, user_id, body)


@router.delete("/projects/{project_id}/members/{user_id}")
async def remove_project_member(
    request: Request,
    project_id: str = Path(...),
    user_id: str = Path(...),
    current_user: UserOut = Depends(require_project_admin_role),
):
    await utils.remove_project_member(request.app.state.db_client, project_id, user_id)
