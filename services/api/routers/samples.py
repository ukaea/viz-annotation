from fastapi import APIRouter, Request
from typing import Tuple
from services.api.schemas.samples import Sample

router = APIRouter(prefix="/projects/{project_id}/samples")

@router.get("")
async def get_samples(project_id: str):
    # Return a list of all samples for this project and info about them
    pass

@router.put("")
async def add_samples(project_id: str, samples: Tuple[int, int]):
    # Add samples from the range specified to the project
    # I'm assuming these will be shot/pulse numbers, hence int, but could be unique ID strings instead
    # Depends if for us a 'sample' will always be a shot/pulse, or if it could be a subset eg a single frame of video
    # Do we also want to allow a single value, or list of specific value?
    pass

@router.delete("")
async def remove_samples(project_id: str, samples: Tuple[int, int]):
    # Remove samples from the range specified from the project
    # Dont envisage this actually deleting the data stored about these samples
    # But do we need a separate method for that?
    pass