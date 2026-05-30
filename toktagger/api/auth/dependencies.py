from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer

from toktagger.api.auth.core import decode_token
from toktagger.api.schemas.users import UserOut

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


async def get_current_user(
    request: Request,
    token: str | None = Depends(oauth2_scheme),
) -> UserOut:
    # Passthrough mode: no users in DB yet (legacy / first-install)
    if not getattr(request.app.state, "auth_required", True):
        return UserOut.model_validate(
            {
                "_id": "000000000000000000000000",
                "username": "admin",
                "email": "",
                "global_role": "admin",
                "is_active": True,
                "timestamp": "2000-01-01T00:00:00",
            }
        )

    if token is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = decode_token(token)
        username: str = payload.get("sub")
        if not username:
            raise ValueError("missing sub")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    db_client = request.app.state.db_client
    docs = await db_client.get_filtered_documents(
        "users", filters={"username": username}
    )
    if not docs:
        raise HTTPException(status_code=401, detail="User not found")

    user_doc = docs[0]
    if not user_doc.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account is inactive")

    return UserOut.model_validate(user_doc)


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
):
    """Return the membership record, or None for global admins (unrestricted)."""
    if current_user.global_role == "admin":
        return None

    db_client = request.app.state.db_client
    from toktagger.api.schemas import convert_to_objectid
    from bson.errors import InvalidId

    try:
        project_oid = convert_to_objectid(project_id, "projects")
        user_oid = convert_to_objectid(current_user.id, "users")
    except (HTTPException, InvalidId):
        raise HTTPException(status_code=404, detail="Project not found")

    docs = await db_client.get_filtered_documents(
        "project_members",
        filters={"project_id": project_oid, "user_id": user_oid},
    )
    if not docs:
        raise HTTPException(
            status_code=403, detail="You are not a member of this project"
        )
    return docs[0]


async def require_project_annotator(
    membership=Depends(get_project_membership),
    current_user: UserOut = Depends(get_current_user),
):
    if current_user.global_role == "admin":
        return current_user
    if membership and membership.get("role") == "viewer":
        raise HTTPException(
            status_code=403, detail="Viewers cannot create or modify annotations"
        )
    return current_user


async def require_project_admin_role(
    project_id: str,
    request: Request,
    current_user: UserOut = Depends(get_current_user),
):
    if current_user.global_role == "admin":
        return current_user

    db_client = request.app.state.db_client
    from toktagger.api.schemas import convert_to_objectid

    project_oid = convert_to_objectid(project_id, "projects")
    user_oid = convert_to_objectid(current_user.id, "users")

    docs = await db_client.get_filtered_documents(
        "project_members",
        filters={"project_id": project_oid, "user_id": user_oid, "role": "admin"},
    )
    if not docs:
        raise HTTPException(
            status_code=403,
            detail="Project admin access required",
        )
    return current_user
