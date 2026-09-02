from fastapi import APIRouter, Request
from toktagger.api.crud import utils

router = APIRouter(prefix="/paths", tags=["Paths"])


@router.get("/files", operation_id="get_files", response_model=list[str])
async def get_files(request: Request, dir_path: str, file_type: str) -> list[str]:
    """
    Get a list of file names within the specified directory.
    ---------------------------------------------------------

    MCP Documentation
    -----------------
    This endpoint is not exposed to the MCP server.
    """
    file_names = await utils.get_files(dir_path, file_type)
    return file_names


@router.get("/directories", operation_id="get_directories", response_model=list[str])
async def get_directories(request: Request, dir_path: str, file_type: str) -> list[str]:
    """
    Get a list of directories within the specified path, filtered by file type.
    ---------------------------------------------------------------------------

    MCP Documentation
    -----------------
    This endpoint is not exposed to the MCP server.
    """
    dir_names = await utils.get_directories(dir_path)
    filtered_dirs = await utils.filter_directories_by_file_type(dir_names, file_type)
    return filtered_dirs
