from fastapi import APIRouter, Depends, Request
from toktagger.api.auth.dependencies import get_current_user
from toktagger.api.crud import utils

router = APIRouter(
    prefix="/paths", tags=["Paths"], dependencies=[Depends(get_current_user)]
)


@router.get("/files", response_model=list[str])
async def get_files(request: Request, dir_path: str, file_type: str) -> list[str]:
    file_names = await utils.get_files(dir_path, file_type)
    return file_names


@router.get("/directories", response_model=list[str])
async def get_directories(request: Request, dir_path: str, file_type: str) -> list[str]:
    dir_names = await utils.get_directories(dir_path)
    filtered_dirs = await utils.filter_directories_by_file_type(dir_names, file_type)
    return filtered_dirs
