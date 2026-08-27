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
    Purpose:
        List files in a project's data directory, filtered by type (e.g. "sample", "annotation").

    Use When:
        - You need to discover what data files exist in a project's storage
        - You want to browse the file system of a specific data loader
        - You need to verify sample or annotation files are present

    Do Not Use When:
        - You need the actual file contents — use toktagger_get_sample_data for signal/sample data
        - You need project metadata — use toktagger_read_get_projects instead

    Returns:
        A list of file name strings found in the directory

    Example User Requests:
        - "What sample files are in this project directory?"
        - "Show me the annotation files for project 6a8f2340b6b4f8d585fd1a67"
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
    Purpose:
        List subdirectories in a project's data directory, filtered by type.

    Use When:
        - You need to navigate the directory structure of a project's data storage
        - You want to discover available data subsets or partitions
        - You need to browse directory hierarchy before fetching files

    Do Not Use When:
        - You need actual data content — use toktagger_read_get_files or toktagger_get_sample_data instead
        - You need project-level metadata — use toktagger_read_get_projects instead

    Returns:
        A list of directory name strings matching the specified file type

    Example User Requests:
        - "What subdirectories exist in this data path?"
        - "Show me the directories for sample data"
    """
    dir_names = await utils.get_directories(dir_path)
    filtered_dirs = await utils.filter_directories_by_file_type(dir_names, file_type)
    return filtered_dirs
