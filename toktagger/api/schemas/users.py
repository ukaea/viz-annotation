from typing import Literal, Optional
from pydantic import BaseModel, Field
from toktagger.api.schemas import ConfiguredModel


class UserBase(ConfiguredModel):
    """Shared fields for user models."""

    email: str = ""
    global_role: Literal["admin", "user"] = "user"
    is_active: bool = True


class UserIn(UserBase):
    username: str
    hashed_password: str


class UserOut(UserBase):
    id: str = Field(..., alias="_id")
    username: str


class UserCreate(BaseModel):
    username: str
    password: str
    email: str = ""
    global_role: Literal["admin", "user"] = "user"


class UserUpdate(BaseModel):
    email: Optional[str] = None
    global_role: Optional[Literal["admin", "user"]] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None


class ProjectMember(ConfiguredModel):
    project_id: str
    user_id: str
    role: Literal["admin", "annotator", "viewer"] = "annotator"
    show_others_annotations: bool = True


class ProjectMemberOut(ConfiguredModel):
    id: str = Field(..., alias="_id")
    project_id: str
    user_id: str
    username: str
    role: Literal["admin", "annotator", "viewer"]
    show_others_annotations: bool


class ProjectMemberCreate(BaseModel):
    username: str
    role: Literal["admin", "annotator", "viewer"] = "annotator"


class ProjectMemberUpdate(BaseModel):
    role: Optional[Literal["admin", "annotator", "viewer"]] = None
    show_others_annotations: Optional[bool] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
