from fastapi import APIRouter, Request, HTTPException
from toktagger.api.schemas.projects import Project, Task
from toktagger.api.schemas.samples import Sample
from toktagger.api.schemas.data import DataParamTypes
from toktagger.api.schemas.annotators import (
    AnnotatorParamTypes,
    AnnotatorTypes,
)
from toktagger.api.crud.utils import get_project, get_sample
from toktagger.api.core.annotators import ANNOTATORS, ANNOTATORS_PER_TASK
from toktagger.api.core.data_loaders import LoaderRegistry

router = APIRouter(
    prefix="/projects/{project_id}",
    tags=["Annotators"],
)


@router.get("/annotator")
async def get_annotators(request: Request, project_id: str):
    # Dunno if this is of any use
    pass


@router.post("/samples/{sample_id}/annotator/{annotator_type}")
async def create_annotations(
    request: Request,
    project_id: str,
    sample_id: str,
    annotator_type: AnnotatorTypes,
    annotator_params: AnnotatorParamTypes,
    data_params: DataParamTypes,
):
    # Use the specified annotator to label this sample for this project
    # Would use the datapool to load and process the data
    # The pass it through the selected annotator within the Project to make predictions
    # Return these predictions to the user, *without* adding to the database
    # Can be passed a set of annotator params and sample params?
    db_client = request.app.state.db_client
    project: Project = await get_project(db_client, project_id)
    annotator_cls = ANNOTATORS[annotator_type]

    if not annotator_cls:
        raise HTTPException(status_code=404, detail="Specified annotator not found.")
    if annotator_type not in ANNOTATORS_PER_TASK[Task(project.task)]:
        raise HTTPException(
            status_code=409,
            detail=f"The selected annotator cannot be used for {project.task} labelling projects.",
        )

    sample: Sample = await get_sample(db_client, project_id, sample_id)

    data_loader = LoaderRegistry.get(project.data_loader)()
    data_item = data_loader.get_sample(
        sample,
        params=data_params,
        time_min=project.time_min,
        time_max=project.time_max,
        min_time_step=project.min_time_step,
    )

    annotator = annotator_cls(annotator_params)
    annotations = annotator.predict(data_item)

    return annotations
