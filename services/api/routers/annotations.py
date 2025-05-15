from fastapi import APIRouter, Request
from services.api.schemas.samples import Sample
from services.api.schemas.annotators import Annotator
from services.api.schemas.filters import AnnotationFilters

router = APIRouter(prefix="/projects/{project_id}")

@router.get("/annotations")
async def get_all_annotations(project_id: str, filters: AnnotationFilters = None):
    # Return all annotations available for this project across all samples, subject to filters
    # Can filter by eg sample numbers, or whether to only return validated/non-validated annotations
    pass

@router.delete("/annotations")
async def delete_all_annotations(project_id: str, filters: AnnotationFilters = None):
    # Delete annotations available for this project across all samples, subject to filters
    # Can filter by eg sample numbers, or whether to only return validated/non-validated annotations
    pass

@router.get("/annotations/next")
async def get_next_annotation(project_id: str):
    # Return the next annotation for human validation for this project
    # Should use the query strategy, which access the database to determine the next sample to annotate
    # Returns data about the annotation, and also the sample_id and any additional info required (eg camera, frame number)
    # This should then be passed in to the /data endpoint to get required data for visualisation
    pass

@router.get("/samples/{sample_id}/annotations")
async def get_annotations(project_id: str, sample_id: int, params: Sample = None):
    # Return annotations available for this project and sample, if any
    # Can filter by params, eg specific camera or frame being returned (or return all annotations for this sample at once and store client side?)
    # Should return whether these are validated as a boolean
    pass

@router.put("/samples/{sample_id}/annotations")
async def add_annotations(project_id: str, sample_id: int, request: Request):
    # Add human annotations to this project and sample
    # Again dont know what form this data will take so have set to a Request for now
    # This data could be for one or more events per task, ie multiple ELMs or UFOs per pulse
    # This should be added into the database, with validated=True
    pass

@router.delete("/samples/{sample_id}/annotations")
async def remove_annotations(project_id: str, sample_id: int):
    # Remove annotations for this project and sample
    # Probably dont need to be able to specify params here, don't envisage how/why the UI would allow you to remove specific annotations
    pass

@router.get("/annotator")
async def get_annotators(project_id: str):
    # Return a list of all annotators available for this project
    pass

@router.get("/samples/{sample_id}/annotator/{annotator_id}")
async def create_annotations(project_id: str, sample_id: str, annotator_id: str, annotator_params: Annotator, sample_params: Sample = None):
    # Use the specified annotator to label this sample for this project
    # Would use the datapool to load and process the data
    # The pass it through the selected annotator within the Project to make predictions
    # Return these predictions to the user, *without* adding to the database
    # Can be passed a set of annotator params and sample params?
    pass