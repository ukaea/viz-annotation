from fastapi import APIRouter

from services.api.schemas.projects import Project

router = APIRouter(prefix="/projects")

@router.get("")
async def get_projects():
    # Return a list of all projects and info about them
    pass
    
@router.post("")
async def create_project(project: Project):
    # Create instance of this project class, instantiating all required classes for that task, and return its ID
    # In the future, should be able to specify eg dataloader, data type, query strategy etc
    pass

    