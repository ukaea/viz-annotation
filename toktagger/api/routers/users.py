from fastapi import APIRouter, Depends, HTTPException, Path, Request

from toktagger.api.auth.dependencies import (
    get_current_user,
    require_global_admin,
    require_project_admin_role,
)
from toktagger.api.auth.core import hash_password
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
# Global user management (admin only)
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
    if body.username.startswith("model::") or body.username.startswith("__"):
        raise HTTPException(
            status_code=422, detail="Username uses a reserved prefix"
        )
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
    doc = await utils.get_user_by_id(request.app.state.db_client, user_id)
    if not doc:
        raise HTTPException(status_code=404, detail="User not found")
    return UserOut.model_validate(doc)


@router.put("/users/{user_id}")
async def update_user(
    request: Request,
    body: UserUpdate,
    user_id: str = Path(...),
    current_user: UserOut = Depends(get_current_user),
):
    if current_user.global_role != "admin" and current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    updates = body.model_dump(exclude_none=True)
    if "password" in updates:
        updates["hashed_password"] = hash_password(updates.pop("password"))

    await utils.update_user(request.app.state.db_client, user_id, updates)


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
    # Any project member or global admin can list members
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
    current_user: UserOut = Depends(get_current_user),
):
    db_client = request.app.state.db_client
    await require_project_admin_role(project_id, request, current_user)

    user_doc = await utils.get_user_by_username(db_client, body.username)
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")

    member_id = await utils.add_project_member(
        db_client, project_id, str(user_doc["_id"]), body.role
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

    # Project admin or the user themselves (for show_others_annotations only)
    if current_user.id != user_id and current_user.global_role != "admin":
        membership = await utils.get_project_membership(
            db_client, project_id, current_user.id
        )
        if not membership or membership.get("role") != "admin":
            # Non-admins may only update their own show_others_annotations
            raise HTTPException(status_code=403, detail="Project admin access required")

    updates = body.model_dump(exclude_none=True)
    await utils.update_project_member(db_client, project_id, user_id, updates)


@router.delete("/projects/{project_id}/members/{user_id}")
async def remove_project_member(
    request: Request,
    project_id: str = Path(...),
    user_id: str = Path(...),
    current_user: UserOut = Depends(get_current_user),
):
    db_client = request.app.state.db_client
    await require_project_admin_role(project_id, request, current_user)
    await utils.remove_project_member(db_client, project_id, user_id)
