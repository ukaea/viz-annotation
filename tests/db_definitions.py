import importlib
import pathlib

from toktagger.api.auth.core import hash_password
from toktagger.api.schemas.annotations import (
    TimePointBatch,
    TimeRegionBatch,
)
from toktagger.api.schemas.annotators import AnnotatorTypes
from toktagger.api.schemas.projects import ProjectIn, QueryStrategyType, Task
from toktagger.api.schemas.samples import (
    SampleIn,
    ShotData,
    TimeSeriesFileData,
)
from toktagger.api.schemas.projects import ProjectMember
from toktagger.api.schemas.users import UserIn

if importlib.util.find_spec("ray") is not None:
    from tests.models_definitions import (
        MODEL_1 as MODEL_1,
    )
    from tests.models_definitions import (
        MODEL_2 as MODEL_2,
    )
    from tests.models_definitions import (
        MODEL_3 as MODEL_3,
    )
    from tests.models_definitions import (
        MODEL_4 as MODEL_4,
    )
    from tests.models_definitions import (
        MockDisruptionCNN as MockDisruptionCNN,
    )
    from tests.models_definitions import (
        MockParamsTimeSeriesCNN as MockParamsTimeSeriesCNN,
    )
    from tests.models_definitions import (
        MockTimeSeriesCNN as MockTimeSeriesCNN,
    )


# Define some common things to add to db
PROJECT_1 = ProjectIn(
    name="test_project_0",
    task=Task.PROFILE_2D,
    query_strategy=QueryStrategyType.SEQUENTIAL,
    data_loader="uda",
)
PROJECT_2 = ProjectIn(
    name="test_project_1",
    task=Task.TIME_SERIES,
    query_strategy=QueryStrategyType.SEQUENTIAL,
    data_loader="tabular",
)
PROJECT_3 = ProjectIn(
    name="project_2",
    task=Task.VIDEO,
    query_strategy=QueryStrategyType.UNCERTAINTY,
    data_loader="image",
)


SAMPLE_1 = SampleIn(
    shot_id=1,
    data=ShotData(protocol="uda", signal_names=["Ip"]),
    annotations=None,
)
SAMPLE_2 = SampleIn(
    shot_id=2,
    data=ShotData(protocol="sal", signal_names=["Ip"]),
    annotations=None,
)
SAMPLE_3 = SampleIn(
    shot_id=3,
    data=TimeSeriesFileData(
        file_name="test.csv",
        type="csv",
        protocol="s3",
        signal_names=["Ip"],
    ),
    annotations=None,
)
SAMPLE_4 = SampleIn(
    shot_id=4,
    data=TimeSeriesFileData(
        file_name=str(
            pathlib.Path(__file__).parent.joinpath("10000.parquet").absolute()
        ),
        type="parquet",
        protocol="file",
        signal_names=["Ip"],
    ),
    annotations=None,
)

ANNOTATION_1 = TimeRegionBatch(
    shot_id=1,
    time_min=0.2,
    time_max=0.4,
    label="annotation",
    validated=True,
    created_by=AnnotatorTypes.MANUAL_ANNOTATION,
)
ANNOTATION_2 = TimeRegionBatch(
    shot_id=1,
    time_min=0.1,
    time_max=0.2,
    label="ramp_up",
    validated=True,
    created_by=AnnotatorTypes.MANUAL_ANNOTATION,
)
ANNOTATION_3 = TimePointBatch(
    shot_id=1,
    time=0.1,
    label="disruption",
    validated=False,
    uncertainty=0.6,
    created_by=AnnotatorTypes.PEAK_DETECTION,
)
ANNOTATION_4 = TimePointBatch(
    shot_id=2,
    time=0.3,
    label="disruption",
    validated=False,
    uncertainty=0.4,
    created_by=AnnotatorTypes.PEAK_DETECTION,
)
ANNOTATION_5 = TimePointBatch(
    shot_id=4,
    time=0.4,
    label="disruption",
    validated=False,
    uncertainty=0.8,
    created_by=AnnotatorTypes.PEAK_DETECTION,
)


USER_ADMIN = UserIn(
    username="admin",
    hashed_password=hash_password("admin_pass"),
    global_role="admin",
    is_active=True,
)
USER_ALICE = UserIn(
    username="alice",
    hashed_password=hash_password("alice_pass"),
    global_role="user",
    is_active=True,
)
USER_BOB = UserIn(
    username="bob",
    hashed_password=hash_password("bob_pass"),
    global_role="user",
    is_active=True,
)

PROJECT_ADMIN = ProjectMember(role="admin")
