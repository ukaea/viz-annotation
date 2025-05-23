from fastapi import APIRouter

from services.api.schemas.projects import Project

router = APIRouter(prefix="/projects")


@router.get("")
async def get_projects() -> list[Project]:
    # Return a list of all projects and info about them
    pass


@router.post("")
async def create_project(project: Project):
    # Create instance of this project class, instantiating all required classes for that task, and return its ID
    # In the future, should be able to specify eg dataloader, data type, query strategy etc
    pass


@router.get("/{project_id}")
async def get_project(project_id: str) -> Project:
    # Return information about a specific project
    # Have put project_id as a string for now, but might want to use ShortUUID?
    pass


@router.put("/{project_id}")
async def update_project(project_id: str, project: Project):
    # Update a project with new information
    # Eg, change the name, change the machine its targeting, etc
    # Are there any things we dont want the user to be able to change once a project is created?
    pass


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    # Delete this specific project
    pass
