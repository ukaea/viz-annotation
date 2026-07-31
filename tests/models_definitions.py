from toktagger.api.models.base import Model, ModelRegistry
from toktagger.api.schemas.annotations import (
    TimePoint,
    TimeRegion,
)
from toktagger.api.schemas.models import ModelIn

import pathlib
import typing
import random
import pydantic


# Create a mock model for use in our model definitions
@ModelRegistry.register("mock_disruption_cnn", ["time-series"])
class MockDisruptionCNN(Model):
    def define_model(self):
        return "Test Model"

    def train(self, samples, annotations, *args, **kwargs):
        self.log_progress(
            training_status="started",
            progress=50,
            score=20,
        )
        return 60

    def predict(self, samples, *args, **kwargs):
        return [
            [
                TimePoint(
                    validated=False,
                    uncertainty=random.random(),
                    label=self.id,
                    time=random.randint(80, 100),
                    created_by=self.type,
                )
            ]
            for i in range(len(samples))
        ]

    def save(self, file_stem: str):
        pathlib.Path(file_stem).with_suffix(".model").write_text(self.model)

    def load(self, file_path):
        self.model = pathlib.Path(file_path).read_text()


class TimeSeriesCNN(Model):
    def define_model(self):
        return "Test Model"

    def train(self, samples, annotations, params=None):
        self.log_progress(
            training_status="started",
            progress=50,
            score=20,
        )
        return 60

    def predict(self, samples, params=None, data_params=None):
        anns = []
        for i in range(len(samples)):
            ramp_up_start = random.randint(0, 20)
            ramp_up_end = ramp_up_start + random.randint(10, 30)
            flat_top_end = 60

            anns.append(
                [
                    TimeRegion(
                        validated=False,
                        uncertainty=random.random(),
                        label="Ramp Up",
                        time_min=ramp_up_start,
                        time_max=ramp_up_end,
                        created_by=self.type,
                    ),
                    TimeRegion(
                        validated=False,
                        uncertainty=random.random(),
                        label="Flat Top",
                        time_min=ramp_up_end,
                        time_max=flat_top_end,
                        created_by=self.type,
                    ),
                    TimePoint(
                        validated=False,
                        uncertainty=random.random(),
                        label="Disruption",
                        time=flat_top_end + 1,
                        created_by=self.type,
                    ),
                ]
            )
        return anns

    def save(self, file_stem: str):
        pathlib.Path(file_stem).with_suffix(".model").write_text(self.model)

    def load(self, file_path):
        self.model = pathlib.Path(file_path).read_text()


@ModelRegistry.register("mock_timeseries_cnn", ["time-series"])
class MockTimeSeriesCNN(TimeSeriesCNN):
    pass


class TimeSeriesCNNParams(pydantic.BaseModel):
    final_score: int = pydantic.Field(ge=50, lt=100)
    test_string: str
    test_bool: bool = True
    test_selection: typing.Literal["selection_1", "selection_2"]


@ModelRegistry.register(
    "mock_params_timeseries_cnn",
    ["time-series"],
    TimeSeriesCNNParams,
    TimeSeriesCNNParams,
)
class MockParamsTimeSeriesCNN(TimeSeriesCNN):
    def train(self, samples, annotations, params: TimeSeriesCNNParams):
        self.log_progress(
            training_status="started",
            progress=50,
            score=20,
        )
        return params.final_score

    def predict(self, samples, params: TimeSeriesCNNParams, data_params: None):
        anns = []
        for i in range(len(samples)):
            ramp_up_start = random.randint(0, 20)
            ramp_up_end = ramp_up_start + random.randint(10, 30)
            flat_top_end = params.final_score

            anns.append(
                [
                    TimeRegion(
                        validated=False,
                        uncertainty=random.random(),
                        label="Ramp Up",
                        time_min=ramp_up_start,
                        time_max=ramp_up_end,
                        created_by=self.type,
                    ),
                    TimeRegion(
                        validated=False,
                        uncertainty=random.random(),
                        label="Flat Top",
                        time_min=ramp_up_end,
                        time_max=flat_top_end,
                        created_by=self.type,
                    ),
                    TimePoint(
                        validated=False,
                        uncertainty=random.random(),
                        label="Disruption",
                        time=flat_top_end + 1,
                        created_by=self.type,
                    ),
                ]
            )
        return anns


MODEL_1 = ModelIn(
    type="mock_disruption_cnn",
    version=1,
    training_status="completed",
    progress=100,
    score=80,
)

MODEL_2 = ModelIn(
    type="mock_disruption_cnn",
    version=2,
    training_status="completed",
    progress=100,
    score=90,
)
MODEL_3 = ModelIn(
    type="disruption_cnn",
    version=3,
    training_status="started",
    progress=50,
    score=60,
    task_id="abc123",
)
MODEL_4 = ModelIn(
    type="mock_params_timeseries_cnn",
    version=1,
    training_status="completed",
    progress=100,
    score=80,
)
