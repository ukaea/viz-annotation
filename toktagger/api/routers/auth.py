from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm

from toktagger.api.auth.core import create_access_token, verify_password
from toktagger.api.auth.dependencies import get_current_user
from toktagger.api.schemas.users import TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/token", response_model=TokenResponse)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    db_client = request.app.state.db_client
    docs = await db_client.get_filtered_documents(
        "users", filters={"username": form_data.username}
    )
    if not docs:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    user_doc = docs[0]
    if not verify_password(form_data.password, user_doc.get("hashed_password", "")):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if not user_doc.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account is inactive")

    token = create_access_token({"sub": user_doc["username"]})
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserOut)
async def get_me(current_user: UserOut = Depends(get_current_user)):
    return current_user
