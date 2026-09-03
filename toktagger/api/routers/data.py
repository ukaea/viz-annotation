from typing import Optional
from toktagger.api.core.views import DATA_VIEWS
from toktagger.api.core.data_loaders import LoaderRegistry
from toktagger.api.crud import utils
from toktagger.api.schemas.data import (
    DataResponseType,
    DataParams,
    DataParamTypes,
    ImageData,
    MultiVariateTimeSeriesData,
    MultiProfile2DData,
    SampleSummaryTypes,
    SummaryAxes,
    SummaryValues,
    Summary2DValues,
    SignalSummary,
    Signal2DSummary,
    ImageSampleSummary,
    TimeSeriesSampleSummary,
    Profile2DSampleSummary,
)
from toktagger.api.schemas.views import ViewParams, ViewParamTypes

from fastapi import APIRouter, HTTPException, Request
from toktagger.api.crud.db import MongoDBClient
from toktagger.api.core.data_loaders import DataLoaderError
from PIL import Image
import numpy
import base64
import io

router = APIRouter(
    prefix="/projects/{project_id}/samples/{sample_id}/data", tags=["Data"]
)


async def _get_data(
    db_client: MongoDBClient,
    project_id: str,
    sample_id: str,
    params: Optional[DataParamTypes] = DataParams(),
    view: Optional[ViewParamTypes] = ViewParams(),
) -> DataResponseType:
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
    This endpoint is not exposed to the MCP server, as it can add potentially very large
    datasets directly into the agent's context window - use summary endpoints instead.
    """
    db_client = request.app.state.db_client

    data = _get_data(db_client, project_id, sample_id, params, view)
    return data


@router.post(
    "",
    operation_id="get_sample_data_summary",
    tags=["MCP"],
    response_model=DataResponseType,
)
async def get_sample_data_summary(
    request: Request,
    project_id: str,
    sample_id: str,
    params: Optional[DataParamTypes] = DataParams(),
    view: Optional[ViewParamTypes] = ViewParams(),
) -> SampleSummaryTypes:
    """
    Get a summary of the data returned for a specific sample.
    ---------------------------------------------------------

    MCP Documentation
    -----------------
    Purpose:
        Retrieve a summary of diagnostic data for a sample, with optional view transformation (e.g. profile-2d heatmaps).
        Data formats supported are time-series, 2D profiles, and images.
        Note that this only returns a summary, not the full dataset. Use the Python client to retrieve full data for analysis if requested.

    Use When:
        - You are asked what kind of data a specific sample / project contains
        - You are asked which signals are present in a sample's data
        - You are asked for max/min/number of points in the dataset for a signal

    Do Not Use When:
        - You only need sample metadata (shot_id, validation status) — use toktagger_get_sample instead
        - You need the raw signal values (e.g. plasma current, density) for a sample to analyse - use the Python client instead
        - You want to analyse data / create annotations from built-in annotators, use toktagger_create_automated_sample_annotations instead
        - You want to analyse data / create predictions from an ML model - use toktagger_create_model_predictions or toktagger_create_sample_model_predictions instead

    Returns:
        Summaries of sample data, containing information such as signal names, max/min/mean values, point counts, etc

    Example User Requests:
        - "What does the data in this project/sample look like?"
        - "How many signals are present in the data for this project/sample?"
        - "What is the maximum value in the plasma current in this sample?"
        - "What time span does data in this sample cover?"
        - "What is the height and width of image frames in this sample?"
        - "Are image frames in this project in colour?"
    """
    db_client = request.app.state.db_client

    data = _get_data(db_client, project_id, sample_id, params, view)

    # Compute summaries
    if isinstance(data, ImageData):
        # Convert back to image array
        # TODO may need to change this when return_raw is available
        image_bytes = base64.b64decode(data.values)
        im = Image.open(io.BytesIO(image_bytes))
        arr = numpy.array(im)

        return ImageSampleSummary(
            type="video",
            description="One frame from a camera diagnostic video inside a Tokamak",
            num_signals=1,
            frame_number=data.frame,
            shape=arr.shape,
            height=arr.shape[0],
            width=arr.shape[1],
            colour_mode=im.mode,
            max=arr.max(),
            min=arr.min(),
            mean=arr.mean(),
        )

    if isinstance(data, MultiVariateTimeSeriesData):
        signals = {}
        for signal, time_series in data.values.items():
            signals[signal] = SignalSummary(
                time=SummaryAxes(
                    count=len(time_series.time),
                    max=numpy.max(time_series.time),
                    min=numpy.min(time_series.time),
                ),
                values=SummaryValues(
                    count=len(time_series.values),
                    max=numpy.max(time_series.values),
                    min=numpy.min(time_series.values),
                    mean=numpy.mean(time_series.values),
                ),
            )
        return TimeSeriesSampleSummary(
            type="time-series",
            description="Time series signals from one or more diagnostics inside a Tokamak.",
            num_signals=len(data.values),
            signals=signals,
        )

    if isinstance(data, MultiProfile2DData):
        signals = {}
        for signal, profile_2d in data.values.items():
            arr = numpy.array(profile_2d.values)
            signals[signal] = Signal2DSummary(
                time=SummaryAxes(
                    count=len(profile_2d.time),
                    max=numpy.max(profile_2d.time),
                    min=numpy.min(profile_2d.time),
                ),
                dim_1=SummaryAxes(
                    count=len(profile_2d.dim_1),
                    max=numpy.max(profile_2d.dim_1),
                    min=numpy.min(profile_2d.dim_1),
                ),
                values=Summary2DValues(
                    shape=arr.shape,
                    count=arr.flatten().shape(),
                    max=numpy.max(profile_2d.values),
                    min=numpy.min(profile_2d.values),
                    mean=numpy.mean(profile_2d.values),
                ),
            )
        return Profile2DSampleSummary(
            type="profile-2d",
            description="2D profile signals from one or more diagnostics inside a Tokamak (eg, spectrometers). Contains measurements of points along an axis (dim_1) at each time point.",
            num_signals=len(data.values),
            signals=signals,
        )
