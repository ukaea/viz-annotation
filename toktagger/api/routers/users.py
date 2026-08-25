from fastapi import APIRouter, Depends, HTTPException, Path, Request

from toktagger.api.auth.core import hash_password
from toktagger.api.auth.dependencies import (
    get_current_user,
    require_global_admin,
    require_project_admin_role,
)
from toktagger.api.crud import utils
from toktagger.api.crud.db import MongoDBClient
from toktagger.api.schemas.users import (
    ProjectMember,
    ProjectMemberCreate,
    ProjectMemberOut,
    ProjectMemberUpdate,
    UserCreate,
    UserIn,
    UserOut,
    UserUpdate,
)

router = APIRouter(tags=["Users"], dependencies=[Depends(get_current_user)])


# ---------------------------------------------------------------------------
# Global user management
# ---------------------------------------------------------------------------


@router.get("/users", response_model=list[UserOut])
async def list_users(
    request: Request,
    _: UserOut = Depends(require_global_admin),
) -> list[UserOut]:
    return await utils.get_all_users(request.app.state.db_client)


@router.post("/users", response_model=dict)
async def create_user(
    request: Request,
    body: UserCreate,
    _: UserOut = Depends(require_global_admin),
) -> dict[str, str]:
    # Reserved prefixes protect the internal worker namespace and the synthetic
    # created_by values stamped on ML-model predictions (worker.py) and built-in
    # annotator suggestions (core/annotators.py), so a real user can't collide with them.
    if (
        body.username.startswith("model::")
        or body.username.startswith("annotators::")
        or body.username.startswith("__")
    ):
        raise HTTPException(status_code=422, detail="Username uses a reserved prefix")
    # The creating admin knows the password they just typed in, so the new owner
    # always has to replace it on first login.
    user = UserIn(
        username=body.username,
        hashed_password=hash_password(body.password),
        global_role=body.global_role,
        must_change_password=True,
    )
    user_id = await utils.create_user(request.app.state.db_client, user)
    return {"_id": user_id}


@router.get("/users/me/memberships", response_model=list[ProjectMember])
async def list_my_memberships(
    request: Request,
    current_user: UserOut = Depends(get_current_user),
) -> list[ProjectMember]:
    """Every project membership held by the caller.

    Self-scoped, so it needs no role check. The projects list uses it to gate each
    row without issuing one `/projects/{id}/members` request per row. A global admin
    is unrestricted by membership and gets an empty list.
    """
    return await utils.get_user_memberships(
        request.app.state.db_client, current_user.id
    )


@router.get("/users/{user_id}", response_model=UserOut)
async def get_user(
    request: Request,
    user_id: str = Path(...),
    current_user: UserOut = Depends(get_current_user),
) -> UserOut:
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
) -> None:
    if current_user.global_role != "admin" and current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Self-edit is allowed for password (profile page), but a non-admin
    # must not be able to change global_role or is_active — even on their own
    # account. Without this, self != other-user check above lets any user
    # PUT their own record with global_role="admin" and self-promote.
    if current_user.global_role != "admin" and (
        body.global_role is not None or body.is_active is not None
    ):
        raise HTTPException(
            status_code=403,
            detail="Only an admin can change global_role or is_active",
        )

    db_client: MongoDBClient = request.app.state.db_client

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
) -> None:
    db_client: MongoDBClient = request.app.state.db_client

    # Prevent deleting the last active admin (mirrors the demote/deactivate guard
    # in update_user — otherwise the account list becomes unmanageable).
    all_users = await utils.get_all_users(db_client)
    target = next((u for u in all_users if u.id == user_id), None)
    if target and target.global_role == "admin" and target.is_active:
        remaining_admins = [
            u
            for u in all_users
            if u.global_role == "admin" and u.is_active and u.id != user_id
        ]
        if not remaining_admins:
            raise HTTPException(
                status_code=422,
                detail="Cannot delete the last active admin account",
            )

    await utils.delete_user(db_client, user_id)


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
) -> list[ProjectMemberOut]:
    db_client: MongoDBClient = request.app.state.db_client
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
) -> dict[str, str]:
    db_client: MongoDBClient = request.app.state.db_client
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
) -> None:
    db_client: MongoDBClient = request.app.state.db_client

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


@router.delete("/projects/{project_id}/members/{user_id}")
async def remove_project_member(
    request: Request,
    project_id: str = Path(...),
    user_id: str = Path(...),
    current_user: UserOut = Depends(require_project_admin_role),
) -> None:
    await utils.remove_project_member(request.app.state.db_client, project_id, user_id)
