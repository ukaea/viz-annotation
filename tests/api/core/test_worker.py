import pytest

pytest.importorskip("ray")

from unittest.mock import patch
from unittest.mock import Mock
from toktagger.api.schemas.models import Model, GitlabLoadParams
from toktagger.api.schemas.projects import Project
from tests.models_definitions import TimeSeriesCNN
import tempfile
import uuid
import pathlib
import shutil
from mlflow.entities.model_registry.model_version import ModelVersion

from toktagger.api.core.worker import (
    load_model_local,
    load_model_gitlab,
    check_safetensor,
)

PROJECT = Project(
    name="test_project_1",
    task="time-series",
    query_strategy="sequential",
    data_loader="tabular",
    id="test_project_id",
)


class MockTimeSeriesCNN(TimeSeriesCNN):
    def __init__(
        self,
        model_id: str,
        project: Project,
    ) -> None:
        self.id = model_id
        self.project = project
        self.model = self.define_model()
        self._trained = False


UPDATES = []


class MockActor:
    def __init__(self, model):
        self._model = model

        self.wrapped_load = Mock()
        self.wrapped_load.remote = self._model.wrapped_load

        self.wrapped_save = Mock()
        self.wrapped_save.remote = self._model.wrapped_save


def mock_get_actor(project, model, use_gpu):
    timeseries_model = MockTimeSeriesCNN(str(model.id), PROJECT)
    return MockActor(timeseries_model)


def mock_send_model_updates(project_id, model_id, updates):
    UPDATES.append({"project_id": project_id, "model_id": model_id, "updates": updates})


def mock_mlflow_get_model_version(self, name: str, version: str):
    # Real function converts semantic version to gitlab registry ID
    # We will just return the major version as an int

    return ModelVersion(name=name, version=int(version[1]), creation_timestamp=123)


def mock_requests_get(url, *args, **kwargs):
    response = Mock()
    response.raise_for_status.return_value = None
    response.iter_content.return_value = [str(url).encode()]

    context_manager = Mock()
    context_manager.__enter__ = Mock(return_value=response)
    context_manager.__exit__ = Mock(return_value=None)

    return context_manager


@pytest.fixture
def temp_models_cache(monkeypatch):
    with tempfile.TemporaryDirectory() as tempd:
        monkeypatch.setenv("MODEL_STORAGE", tempd)
        yield pathlib.Path(tempd)
    return


@pytest.fixture
def setup_model():
    model = Model(
        type="mock_timeseries_cnn",
        version=1,
        training_status="queued",
        progress=-0,
        score=0,
        task_id=None,
        project_id="test_project_id",
        id=str(uuid.uuid4()),
    )
    UPDATES.clear()
    return model


@pytest.mark.models_enabled
@pytest.mark.parametrize("safetensors_only", ("True", "False"))
@pytest.mark.parametrize("filename", ("model.pt", "model.safetensors"))
@patch("toktagger.api.core.worker.send_model_updates", mock_send_model_updates)
def test_check_safetensors(
    monkeypatch, safetensors_only, filename, temp_models_cache, setup_model
):
    monkeypatch.setenv("MODELS_SAFETENSORS_ONLY", safetensors_only)
    src_path = pathlib.Path(__file__).parents[2].joinpath(filename)
    # Name them both identically to check its not just checking the suffix...
    dst_path = pathlib.Path(temp_models_cache).joinpath("model.safetensors")
    shutil.copy(src_path, dst_path)

    unsafe = check_safetensor(
        dst_path,
        "test_project_id",
        setup_model.id,
    )

    if safetensors_only == "True" and filename == "model.pt":
        assert unsafe
        assert unsafe.get("message") == "retrieved file is not a SafeTensor!"
        assert UPDATES[-1]["model_id"] == setup_model.id
        assert UPDATES[-1]["updates"].training_status == "failed"
    else:
        assert not unsafe
        assert not UPDATES


@patch("toktagger.api.core.worker.get_actor", mock_get_actor)
@patch("ray.get", lambda val: val)
@patch("toktagger.api.core.worker.send_model_updates", mock_send_model_updates)
@pytest.mark.models_enabled
def test_local_load(temp_models_cache, setup_model):
    # Create tempfile
    with tempfile.NamedTemporaryFile(suffix=".model", mode="w") as tempf:
        tempf.write("Model Weights")
        tempf.flush()

        result = load_model_local(setup_model, PROJECT, pathlib.Path(tempf.name))

        assert result["project_id"] == "test_project_id"
        assert result["model_id"] == setup_model.id
        assert result["message"] is None

        # Check model updated to completed, with 100% completion
        assert UPDATES[-1]["model_id"] == setup_model.id
        assert UPDATES[-1]["updates"].training_status == "completed"
        assert UPDATES[-1]["updates"].progress == 100

        # Check model has been saved after completion
        model_path = temp_models_cache.joinpath(f"{setup_model.id}.model")
        assert model_path.exists()

        # Open the file, check contents are there
        assert model_path.read_text() == "Model Weights"

        # Check original file untouched
        assert pathlib.Path(tempf.name).exists()


@patch("toktagger.api.core.worker.get_actor", mock_get_actor)
@patch("ray.get", lambda val: val)
@patch("toktagger.api.core.worker.send_model_updates", mock_send_model_updates)
@pytest.mark.models_enabled
def test_local_load_missing_file(temp_models_cache, setup_model):
    # Try loading file which doesn't exist
    result = load_model_local(
        setup_model,
        PROJECT,
        pathlib.Path(__file__).parent.joinpath(f"{uuid.uuid4()}.model"),
    )

    assert result["project_id"] == "test_project_id"
    assert result["model_id"] == setup_model.id
    assert "Worker node cannot find weights file at location" in result["message"]

    # Check model updated to failed
    assert UPDATES[-1]["model_id"] == setup_model.id
    assert UPDATES[-1]["updates"].training_status == "failed"

    # Check no model has been saved
    model_path = temp_models_cache.joinpath(f"{setup_model.id}.model")
    assert not model_path.exists()


@patch("toktagger.api.core.worker.get_actor", mock_get_actor)
@patch("ray.get", lambda val: val)
@patch("toktagger.api.core.worker.send_model_updates", mock_send_model_updates)
@pytest.mark.models_enabled
def test_local_load_invalid_file(temp_models_cache, setup_model):
    # Create tempfile
    with tempfile.NamedTemporaryFile(suffix=".model") as tempf:
        # Write invalid bytes which will fail to be read by the model
        tempf.write(b"\xff")
        tempf.flush()

        result = load_model_local(setup_model, PROJECT, pathlib.Path(tempf.name))

        assert result["project_id"] == "test_project_id"
        assert result["model_id"] == setup_model.id
        assert "'utf-8' codec can't decode byte" in result["message"]

        # Check model updated to failed
        assert UPDATES[-1]["model_id"] == setup_model.id
        assert UPDATES[-1]["updates"].training_status == "failed"

        # Check model has not been saved
        model_path = temp_models_cache.joinpath(f"{setup_model.id}.model")
        assert not model_path.exists()

        # Check original file untouched
        assert pathlib.Path(tempf.name).exists()


@patch("toktagger.api.core.worker.get_actor", mock_get_actor)
@patch("ray.get", lambda val: val)
@patch("toktagger.api.core.worker.send_model_updates", mock_send_model_updates)
@patch(
    "toktagger.api.core.worker.MlflowClient.get_model_version",
    mock_mlflow_get_model_version,
)
@patch("toktagger.api.core.worker.requests.get", mock_requests_get)
@pytest.mark.models_enabled
def test_gitlab_load_version(temp_models_cache, setup_model, monkeypatch):
    monkeypatch.setenv("MODELS_GITLAB_URL", "http://test_gitlab_url")
    monkeypatch.setenv("MODELS_GITLAB_TOKEN", "abc123")

    params = GitlabLoadParams(
        weights_path="test_load.model",
        model_name="disruption_cnn",
        model_version="v5.0.0",
        gitlab_project_id=123,
    )
    result = load_model_gitlab(setup_model, PROJECT, params)

    assert result["project_id"] == "test_project_id"
    assert result["model_id"] == setup_model.id
    assert result["message"] is None

    # Check model updated to completed, with 100% completion
    assert UPDATES[-1]["model_id"] == setup_model.id
    assert UPDATES[-1]["updates"].training_status == "completed"
    assert UPDATES[-1]["updates"].progress == 100

    # Check model has been saved after completion
    model_path = temp_models_cache.joinpath(f"{setup_model.id}.model")
    assert model_path.exists()

    # Open the file, check contents are the correct URL
    # {url}/api/v4/projects/{params.gitlab_project_id}/packages/ml_models/{mlflow_model.version}/files/{params.weights_path}
    url = model_path.read_text()
    components = url.split("/")

    assert components[2] == "test_gitlab_url"  # From env var
    assert components[6] == str(params.gitlab_project_id)
    assert components[9] == params.model_version[1]  # Major version
    assert components[11] == params.weights_path
