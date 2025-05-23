from fastapi import APIRouter, Request
from services.api.schemas.samples import Sample

router = APIRouter(prefix="/projects/{project_id}/samples/{sample_id}/data")


@router.get("")
async def get_data(project_id: str, sample_id: int, params: Sample = None):
    # Get data, eg time trace, about the given sample required for the given project
    # Not sure whether we gain anything by having params as a BaseModel
    # But params should contain different things depending on project (ie task) being labelled
    pass


@router.put("")
async def add_data(project_id: str, sample_id: int, request: Request):
    # Add some data for this sample for a given project
    # Eg, could upload a CSV of time trace data for a certain pulse via the web UI
    # Have set the request as just a Request body, because I dont (yet) know what format that needs to be
    pass


@router.delete("")
async def delete_data(project_id: str, sample_id: int, params: Sample = None):
    # Delete data for this sample from this project
    # Not sure if we really need this, but might be nice in case you have a sample which is junk in your dataset
    # Ie the images are all black because the camera failed, etc
    # What if the same data is in use by multiple projects?
    pass
