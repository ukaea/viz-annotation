from typing import Optional
from toktagger.api.core.views import DATA_VIEWS
from toktagger.api.core.data_loaders import LoaderRegistry
from toktagger.api.crud import utils
from toktagger.api.schemas.data import DataResponseType, DataParams, DataParamTypes
from toktagger.api.schemas.views import ViewParams, ViewParamTypes

from fastapi import APIRouter, HTTPException, Request

from toktagger.api.core.data_loaders import DataLoaderError

router = APIRouter(
    prefix="/projects/{project_id}/samples/{sample_id}/data", tags=["Data", "MCP"]
)


@router.post("", operation_id="get_sample_data", response_model=DataResponseType)
async def get_data(
    request: Request,
    project_id: str,
    sample_id: str,
    params: Optional[DataParamTypes] = DataParams(),
    view: Optional[ViewParamTypes] = ViewParams(),
) -> DataResponseType:
    """
    Get data, e.g. time trace, about the given sample required for the given project.
    ----------------------------------------------------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Retrieve the actual diagnostic data for a sample, with optional view transformation (e.g. profile-2d heatmaps).
        Data formats supported are time-series, 2D profiles, and images (encoded in base64 strings by default).
        Note that the data response may be large, so check that it hasn't been trancated before answering the user.

    Use When:
        - You need the raw signal values (e.g. plasma current, density) for a sample to display or analyse
        - You want to visualize time traces, profiles, or other data views
        - You are asked what kind of data a specific sample / project contains

    Do Not Use When:
        - You only need sample metadata (shot_id, validation status) — use toktagger_get_sample instead
        - You need to review existing annotations — use toktagger_get_sample_annotations instead
        - You want to create annotations from built-in annotators, use toktagger_create_automated_sample_annotations instead
        - You want to create predictions from an ML model - use toktagger_create_model_predictions or toktagger_create_sample_model_predictions instead

    Returns:
        Sample data, either single/multivariate time-series, 2D profile(s), or an image encoded in a base64 string.

    Example User Requests:
        - "What does the data in this project/sample look like?"
        - "Get the plasma current time trace for this shot"
        - "Show me the profile-2d heatmap for signal ip for this sample"
        - "Show me frame 100 of the video from this pulse"
        - "Analyse the plasma current trace and identify areas which could represent disruptions"
    """
    db_client = request.app.state.db_client

    project = await utils.get_project(db_client, project_id)
    sample = await utils.get_sample(db_client, project_id, sample_id)

    data_loader = LoaderRegistry.get(project.data_loader)()
    try:
        data = data_loader.get_sample(
            sample,
            params=params,
            time_min=project.time_min,
            time_max=project.time_max,
            min_time_step=project.min_time_step,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except DataLoaderError as e:
        raise HTTPException(404, str(e)) from e

    try:
        data_view = DATA_VIEWS[view.name](view)
        data = data_view(data)
    except Exception as e:
        raise HTTPException(400, str(e)) from e

    return data
