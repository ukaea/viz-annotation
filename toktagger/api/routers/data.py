from typing import Optional
from toktagger.api.core.views import DATA_VIEWS
from toktagger.api.core.data_loaders import LoaderRegistry
from toktagger.api.crud import utils
from toktagger.api.schemas.data import DataResponseType, DataParams, DataParamTypes
from toktagger.api.schemas.views import ViewParams, ViewParamTypes

from fastapi import APIRouter, HTTPException, Request

from toktagger.api.core.data_loaders import DataLoaderError

router = APIRouter(
    prefix="/projects/{project_id}/samples/{sample_id}/data", tags=["Data"]
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
        Retrieve the actual sensor/time-series data for a sample, with optional view transformation (e.g. profile-2d heatmaps).

    Use When:
        - You need the raw signal values (e.g. plasma current, density) for a sample to display or analyze
        - You want to visualize time traces, profiles, or other data views
        - You are building an annotation UI and need to load sample data
        - You need data for model inference or preprocessing

    Do Not Use When:
        - You only need sample metadata (shot_id, validation status) — use toktagger_read_get_sample instead
        - You need annotations — use the annotation endpoints instead
        - You are listing projects — use toktagger_read_get_projects instead

    Returns:
        A DataResponseType object with signal data: { "values": { <signal_name>: { "time": [...], "values": [...] }, ... } } plus optional transformed views

    Example User Requests:
        - "Get the plasma current time trace for this shot"
        - "Show me the profile-2d heatmap for signal ip"
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
