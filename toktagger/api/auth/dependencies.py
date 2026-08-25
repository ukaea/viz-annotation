from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer

from toktagger.api.auth.core import decode_token, get_internal_token
from toktagger.api.crud import utils
from toktagger.api.schemas.projects import ProjectMember
from toktagger.api.schemas.users import UserOut

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)

_INTERNAL_USER = UserOut(
    id="000000000000000000000001",
    username="__internal__",
    global_role="admin",
    is_active=True,
)


async def get_current_user(
    request: Request,
    token: str | None = Depends(oauth2_scheme),
) -> UserOut:
    if token is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Internal server-to-server token used by Ray-worker callbacks (sender.py).
    if token == get_internal_token():
        return _INTERNAL_USER

    try:
        payload = decode_token(token)
        username = payload.get("sub")
        if not username or not isinstance(username, str):
            raise ValueError("Token is missing a subject claim")
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    db_client = request.app.state.db_client
    user = await utils.get_user_by_username(db_client, username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is inactive")
    return user


async def require_global_admin(
    current_user: UserOut = Depends(get_current_user),
) -> UserOut:
    if current_user.global_role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


async def get_project_membership(
    project_id: str,
    request: Request,
    current_user: UserOut = Depends(get_current_user),
) -> ProjectMember | None:
    """Return the membership record, or None for global admins (unrestricted)."""
    if current_user.global_role == "admin":
        return None

    db_client = request.app.state.db_client
    membership = await utils.get_project_membership(
        db_client, project_id, current_user.id
    )
    if not membership:
        raise HTTPException(
            status_code=403, detail="You are not a member of this project"
        )
    return membership


async def require_project_viewer(
    membership: ProjectMember | None = Depends(get_project_membership),
    current_user: UserOut = Depends(get_current_user),
) -> UserOut:
    """Any project member (viewer, annotator, admin) may access read-only resources."""
    return current_user


async def require_project_annotator(
    membership: ProjectMember | None = Depends(get_project_membership),
    current_user: UserOut = Depends(get_current_user),
) -> UserOut:
    if current_user.global_role == "admin":
        return current_user
    # Only roles explicitly allowed to write pass. `membership is None` reaches here
    # only if get_project_membership ever stops raising for non-members, so fail
    # closed on it rather than relying on that behaviour.
    if membership is None or membership.role not in ("admin", "annotator"):
        raise HTTPException(
            status_code=403, detail="Viewers cannot create or modify annotations"
        )
    return current_user


async def require_project_admin_role(
    membership: ProjectMember | None = Depends(get_project_membership),
    current_user: UserOut = Depends(get_current_user),
) -> UserOut:
    if current_user.global_role == "admin":
        return current_user

    if membership is None or membership.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Project admin access required",
        )
    return current_user
