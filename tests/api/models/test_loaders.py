import pytest

pytest.importorskip("ray")

from unittest.mock import patch
from unittest.mock import Mock
from toktagger.api.schemas.models import (
    Model,
    LocalLoadParams,
    GitlabLoadParams,
    HuggingfaceLoadParams,
)
from toktagger.api.schemas.projects import Project
from tests.models_definitions import TimeSeriesCNN
import tempfile
import uuid
import pathlib
import shutil
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.exceptions import MlflowException
from huggingface_hub.errors import RepositoryNotFoundError
from toktagger.api.models.loaders import (
    ModelLoader,
    LocalLoader,
    GitlabLoader,
    HuggingfaceLoader,
)
from requests import Response

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


def mock_send_model_updates(project_id, model_id, updates):
    UPDATES.append({"project_id": project_id, "model_id": model_id, "updates": updates})


def mock_mlflow_get_model_version(self, name: str, version: str):
    # Real function converts semantic version to gitlab registry ID
    # We will just return the major version as an int
    v = int(version[1])

    # Say if major version higher than 5, return exception as that version doesn't exist
    if v > 5:
        raise MlflowException("Model version not found")

    return ModelVersion(name=name, version=v, creation_timestamp=123)


def mock_mlflow_search_version(self, name: str):
    # Take model name from filter_string
    if name == "invalid":
        return []

    return [
        ModelVersion(name=name, version=2, creation_timestamp=123),
        ModelVersion(name=name, version=3, creation_timestamp=123),
        ModelVersion(name=name, version=1, creation_timestamp=123),
    ]


def mock_requests_get(url, *args, **kwargs):
    response = Mock()
    response.raise_for_status.return_value = None
    response.iter_content.return_value = [str(url).encode()]

    context_manager = Mock()
    context_manager.__enter__ = Mock(return_value=response)
    context_manager.__exit__ = Mock(return_value=None)

    return context_manager


def mock_requests_get_invalid_file(url, *args, **kwargs):
    response = Mock()
    response.raise_for_status.return_value = None
    response.iter_content.return_value = [b"\xff"]

    context_manager = Mock()
    context_manager.__enter__ = Mock(return_value=response)
    context_manager.__exit__ = Mock(return_value=None)

    return context_manager


def mock_hf_hub_download(repo_id, filename, revision, local_dir, *args, **kwargs):
    out_file = pathlib.Path(local_dir).joinpath("downloaded.model")

    if repo_id.split("/")[-1] == "invalid":
        out_file.write_bytes(b"\xff")
    elif repo_id.split("/")[-1] == "missing":
        response = Response()
        response.status_code = 404
        raise RepositoryNotFoundError("Repo not found!", response=response)
    else:
        out_file.write_text(f"{repo_id},{filename},{revision}")

    return str(out_file)


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
        status="queued",
        progress=-0,
        score=0,
        task_id=None,
        project_id="test_project_id",
        id=str(uuid.uuid4()),
    )
    UPDATES.clear()
    timeseries_model = MockTimeSeriesCNN(str(model.id), PROJECT)
    return model, MockActor(timeseries_model)


@pytest.mark.models_enabled
@pytest.mark.parametrize("safetensors_only", ("True", "False"))
@pytest.mark.parametrize("filename", ("model.pt", "model.safetensors"))
@patch("toktagger.api.models.loaders.send_model_updates", mock_send_model_updates)
def test_check_safetensors(
    monkeypatch, safetensors_only, filename, temp_models_cache, setup_model
):
    model, model_actor = setup_model

    monkeypatch.setenv("MODELS_SAFETENSORS_ONLY", safetensors_only)
    src_path = pathlib.Path(__file__).parents[2].joinpath(filename)
    # Name them both identically to check its not just checking the suffix...
    dst_path = pathlib.Path(temp_models_cache).joinpath("model.safetensors")
    shutil.copy(src_path, dst_path)

    loader = ModelLoader(
        project=PROJECT,
        model=model,
        model_actor=model_actor,
        params=LocalLoadParams(weights_path=str(dst_path)),
    )

    unsafe = loader._check_safetensor(
        dst_path,
    )

    if safetensors_only == "True" and filename == "model.pt":
        assert unsafe
        assert unsafe.get("message") == "retrieved file is not a SafeTensor!"
        assert UPDATES[-1]["model_id"] == model.id
        assert UPDATES[-1]["updates"].status == "failed"
    else:
        assert not unsafe


@patch("ray.get", lambda val: val)
@patch("toktagger.api.models.loaders.send_model_updates", mock_send_model_updates)
@pytest.mark.models_enabled
def test_local_load(temp_models_cache, setup_model):
    model, model_actor = setup_model

    # Create tempfile
    with tempfile.NamedTemporaryFile(suffix=".model", mode="w") as tempf:
        tempf.write("Model Weights")
        tempf.flush()

        loader = LocalLoader(
            project=PROJECT,
            model=model,
            model_actor=model_actor,
            params=LocalLoadParams(weights_path=tempf.name),
        )
        result = loader.load()

        assert result["project_id"] == "test_project_id"
        assert result["model_id"] == model.id
        assert result["message"] is None

        # Check model updated to completed, with 100% completion
        assert UPDATES[-1]["model_id"] == model.id
        assert UPDATES[-1]["updates"].status == "completed"
        assert UPDATES[-1]["updates"].progress == 100

        # Check model has been saved after completion
        model_path = temp_models_cache.joinpath(model.id, "weights.model")
        assert model_path.exists()

        # Open the file, check contents are there
        assert model_path.read_text() == "Model Weights"

        # Check original file untouched
        assert pathlib.Path(tempf.name).exists()


@patch("ray.get", lambda val: val)
@patch("toktagger.api.models.loaders.send_model_updates", mock_send_model_updates)
@pytest.mark.models_enabled
def test_local_load_missing_file(temp_models_cache, setup_model):
    model, model_actor = setup_model

    # Try loading file which doesn't exist
    loader = LocalLoader(
        project=PROJECT,
        model=model,
        model_actor=model_actor,
        params=LocalLoadParams(weights_path=f"{uuid.uuid4()}.model"),
    )
    result = loader.load()

    assert result["project_id"] == "test_project_id"
    assert result["model_id"] == model.id
    assert "Worker node cannot find file at" in result["message"]

    # Check model updated to failed
    assert UPDATES[-1]["model_id"] == model.id
    assert UPDATES[-1]["updates"].status == "failed"

    # Check no model has been saved
    model_path = temp_models_cache.joinpath(model.id, "weights.model")
    assert not model_path.exists()


@patch("ray.get", lambda val: val)
@patch("toktagger.api.models.loaders.send_model_updates", mock_send_model_updates)
@pytest.mark.models_enabled
def test_local_load_invalid_file(temp_models_cache, setup_model):
    model, model_actor = setup_model

    # Create tempfile
    with tempfile.NamedTemporaryFile(suffix=".model") as tempf:
        # Write invalid bytes which will fail to be read by the model
        tempf.write(b"\xff")
        tempf.flush()

        loader = LocalLoader(
            project=PROJECT,
            model=model,
            model_actor=model_actor,
            params=LocalLoadParams(weights_path=tempf.name),
        )
        result = loader.load()

        assert result["project_id"] == "test_project_id"
        assert result["model_id"] == model.id
        assert "'utf-8' codec can't decode byte" in result["message"]

        # Check model updated to failed
        assert UPDATES[-1]["model_id"] == model.id
        assert UPDATES[-1]["updates"].status == "failed"

        # Check model has not been saved
        model_path = temp_models_cache.joinpath(model.id, "weights.model")
        assert not model_path.exists()

        # Check original file untouched
        assert pathlib.Path(tempf.name).exists()


@patch("ray.get", lambda val: val)
@patch("toktagger.api.models.loaders.send_model_updates", mock_send_model_updates)
@patch(
    "toktagger.api.models.loaders.MlflowClient.get_model_version",
    mock_mlflow_get_model_version,
)
@patch("toktagger.api.models.loaders.requests.get", mock_requests_get)
@pytest.mark.models_enabled
def test_gitlab_load_version(temp_models_cache, setup_model, monkeypatch):
    model, model_actor = setup_model

    monkeypatch.setenv("MODELS_GITLAB_URL", "http://test_gitlab_url")
    monkeypatch.setenv("MODELS_GITLAB_TOKEN", "abc123")

    params = GitlabLoadParams(
        weights_path="test_load.model",
        model_name="disruption_cnn",
        model_version="v5.0.0",
        gitlab_project_id=123,
    )
    loader = GitlabLoader(
        project=PROJECT,
        model=model,
        model_actor=model_actor,
        params=params,
    )
    result = loader.load()

    assert result["project_id"] == "test_project_id"
    assert result["model_id"] == model.id
    assert result["message"] is None

    # Check model updated to completed, with 100% completion
    assert UPDATES[-1]["model_id"] == model.id
    assert UPDATES[-1]["updates"].status == "completed"
    assert UPDATES[-1]["updates"].progress == 100

    # Check model has been saved after completion
    model_path = temp_models_cache.joinpath(model.id, "weights.model")
    assert model_path.exists()

    # Open the file, check contents are the correct URL
    # {url}/api/v4/projects/{params.gitlab_project_id}/packages/ml_models/{mlflow_model.version}/files/{params.weights_path}
    url = model_path.read_text()
    components = url.split("/")

    assert components[2] == "test_gitlab_url"  # From env var
    assert components[6] == str(params.gitlab_project_id)
    assert components[9] == params.model_version[1]  # Major version
    assert components[11] == params.weights_path

    # Check temp downloaded file deleted
    assert not temp_models_cache.joinpath(
        pathlib.Path(params.weights_path).name
    ).exists()


@patch("ray.get", lambda val: val)
@patch("toktagger.api.models.loaders.send_model_updates", mock_send_model_updates)
@patch(
    "toktagger.api.models.loaders.MlflowClient.get_latest_versions",
    mock_mlflow_search_version,
)
@patch("toktagger.api.models.loaders.requests.get", mock_requests_get)
@pytest.mark.models_enabled
def test_gitlab_load_no_version(temp_models_cache, setup_model, monkeypatch):
    model, model_actor = setup_model

    monkeypatch.setenv("MODELS_GITLAB_URL", "http://test_gitlab_url")
    monkeypatch.setenv("MODELS_GITLAB_TOKEN", "abc123")

    params = GitlabLoadParams(
        weights_path="test_load.model",
        model_name="disruption_cnn",
        gitlab_project_id=123,
    )
    loader = GitlabLoader(
        project=PROJECT,
        model=model,
        model_actor=model_actor,
        params=params,
    )
    result = loader.load()

    assert result["project_id"] == "test_project_id"
    assert result["model_id"] == model.id
    assert result["message"] is None

    # Check model updated to completed, with 100% completion
    assert UPDATES[-1]["model_id"] == model.id
    assert UPDATES[-1]["updates"].status == "completed"
    assert UPDATES[-1]["updates"].progress == 100

    # Check model has been saved after completion
    model_path = temp_models_cache.joinpath(model.id, "weights.model")
    assert model_path.exists()

    # Open the file, check contents are the correct URL
    # {url}/api/v4/projects/{params.gitlab_project_id}/packages/ml_models/{mlflow_model.version}/files/{params.weights_path}
    url = model_path.read_text()
    components = url.split("/")

    assert components[2] == "test_gitlab_url"  # From env var
    assert components[6] == str(params.gitlab_project_id)
    assert components[9] == str(3)  # Highest version
    assert components[11] == params.weights_path

    # Check temp downloaded file deleted
    assert not temp_models_cache.joinpath(
        pathlib.Path(params.weights_path).name
    ).exists()


@patch("ray.get", lambda val: val)
@patch("toktagger.api.models.loaders.send_model_updates", mock_send_model_updates)
@patch(
    "toktagger.api.models.loaders.MlflowClient.get_model_version",
    mock_mlflow_get_model_version,
)
@patch("toktagger.api.models.loaders.requests.get", mock_requests_get)
@pytest.mark.models_enabled
def test_gitlab_load_invalid_version(temp_models_cache, setup_model, monkeypatch):
    model, model_actor = setup_model

    monkeypatch.setenv("MODELS_GITLAB_URL", "http://test_gitlab_url")
    monkeypatch.setenv("MODELS_GITLAB_TOKEN", "abc123")

    params = GitlabLoadParams(
        weights_path="test_load.model",
        model_name="disruption_cnn",
        model_version="v8.0.0",  # Higher than 5, so mock will throw 'not found' exception
        gitlab_project_id=123,
    )
    loader = GitlabLoader(
        project=PROJECT,
        model=model,
        model_actor=model_actor,
        params=params,
    )
    result = loader.load()

    assert result["project_id"] == "test_project_id"
    assert result["model_id"] == model.id
    assert (
        result["message"] == "requested version of selected model could not be found!"
    )

    # Check model updated to failed, with 100% completion
    assert UPDATES[-1]["model_id"] == model.id
    assert UPDATES[-1]["updates"].status == "failed"

    # Check model has not been saved
    model_path = temp_models_cache.joinpath(model.id, "weights.model")
    assert not model_path.exists()


@patch("ray.get", lambda val: val)
@patch("toktagger.api.models.loaders.send_model_updates", mock_send_model_updates)
@patch(
    "toktagger.api.models.loaders.MlflowClient.get_latest_versions",
    mock_mlflow_search_version,
)
@patch("toktagger.api.models.loaders.requests.get", mock_requests_get)
@pytest.mark.models_enabled
def test_gitlab_load_no_versions_found(temp_models_cache, setup_model, monkeypatch):
    model, model_actor = setup_model

    monkeypatch.setenv("MODELS_GITLAB_URL", "http://test_gitlab_url")
    monkeypatch.setenv("MODELS_GITLAB_TOKEN", "abc123")

    params = GitlabLoadParams(
        weights_path="test_load.model",
        model_name="invalid",  # Mock will return empty list when searched for
        gitlab_project_id=123,
    )
    loader = GitlabLoader(
        project=PROJECT,
        model=model,
        model_actor=model_actor,
        params=params,
    )
    result = loader.load()
    assert result["project_id"] == "test_project_id"
    assert result["model_id"] == model.id
    assert (
        result["message"] == "requested version of selected model could not be found!"
    )

    # Check model updated to failed, with 100% completion
    assert UPDATES[-1]["model_id"] == model.id
    assert UPDATES[-1]["updates"].status == "failed"

    # Check model has not been saved
    model_path = temp_models_cache.joinpath(model.id, "weights.model")
    assert not model_path.exists()


@patch("ray.get", lambda val: val)
@patch("toktagger.api.models.loaders.send_model_updates", mock_send_model_updates)
@patch(
    "toktagger.api.models.loaders.MlflowClient.get_model_version",
    mock_mlflow_get_model_version,
)
@patch("toktagger.api.models.loaders.requests.get", mock_requests_get_invalid_file)
@pytest.mark.models_enabled
def test_gitlab_load_invalid_file(temp_models_cache, setup_model, monkeypatch):
    model, model_actor = setup_model

    monkeypatch.setenv("MODELS_GITLAB_URL", "http://test_gitlab_url")
    monkeypatch.setenv("MODELS_GITLAB_TOKEN", "abc123")

    params = GitlabLoadParams(
        weights_path="test_load.model",
        model_name="disruption_cnn",
        model_version="v5.0.0",
        gitlab_project_id=123,
    )
    loader = GitlabLoader(
        project=PROJECT,
        model=model,
        model_actor=model_actor,
        params=params,
    )
    result = loader.load()

    assert result["project_id"] == "test_project_id"
    assert result["model_id"] == model.id
    assert "'utf-8' codec can't decode byte" in result["message"]

    # Check model updated to failed,
    assert UPDATES[-1]["model_id"] == model.id
    assert UPDATES[-1]["updates"].status == "failed"

    # Check model has not been saved after completion
    model_path = temp_models_cache.joinpath(model.id, "weights.model")
    assert not model_path.exists()

    # Check temp downloaded file deleted
    assert not temp_models_cache.joinpath(
        pathlib.Path(params.weights_path).name
    ).exists()


@patch("ray.get", lambda val: val)
@patch("toktagger.api.models.loaders.send_model_updates", mock_send_model_updates)
@patch(
    "toktagger.api.models.loaders.MlflowClient.get_model_version",
    mock_mlflow_get_model_version,
)
@patch("toktagger.api.models.loaders.requests.get", mock_requests_get_invalid_file)
@pytest.mark.models_enabled
def test_gitlab_load_missing_env_vars(temp_models_cache, setup_model):
    model, model_actor = setup_model

    params = GitlabLoadParams(
        weights_path="test_load.model",
        model_name="disruption_cnn",
        model_version="v5.0.0",
        gitlab_project_id=123,
    )
    loader = GitlabLoader(
        project=PROJECT,
        model=model,
        model_actor=model_actor,
        params=params,
    )
    result = loader.load()

    assert result["project_id"] == "test_project_id"
    assert result["model_id"] == model.id
    assert (
        "Gitlab URL, Token or Project ID not specified when trying to load ML model!"
        in result["message"]
    )


@patch("ray.get", lambda val: val)
@patch("toktagger.api.models.loaders.send_model_updates", mock_send_model_updates)
@patch(
    "toktagger.api.models.loaders.hf_hub_download",
    mock_hf_hub_download,
)
@pytest.mark.models_enabled
def test_huggingface_load(temp_models_cache, setup_model):
    model, model_actor = setup_model

    params = HuggingfaceLoadParams(
        weights_path="weights.model",
        model_name="test_model",
        model_version="v1.0.0",
        huggingface_userspace="my_user",
    )
    loader = HuggingfaceLoader(
        project=PROJECT,
        model=model,
        model_actor=model_actor,
        params=params,
    )
    result = loader.load()

    assert result["project_id"] == "test_project_id"
    assert result["model_id"] == model.id
    assert result["message"] is None

    # Check model updated to completed, with 100% completion
    assert UPDATES[-1]["model_id"] == model.id
    assert UPDATES[-1]["updates"].status == "completed"
    assert UPDATES[-1]["updates"].progress == 100

    # Check model has been saved after completion
    model_path = temp_models_cache.joinpath(model.id, "weights.model")
    assert model_path.exists()

    # Open the file, check contents are the correct params
    url = model_path.read_text()
    components = url.split(",")

    assert components[0] == f"{params.huggingface_userspace}/{params.model_name}"
    assert components[1] == params.weights_path
    assert components[2] == params.model_version  # Major version

    # Check temp downloaded file deleted
    assert not temp_models_cache.joinpath(model.id, "downloaded.model").exists()


@patch("ray.get", lambda val: val)
@patch("toktagger.api.models.loaders.send_model_updates", mock_send_model_updates)
@patch(
    "toktagger.api.models.loaders.hf_hub_download",
    mock_hf_hub_download,
)
@pytest.mark.models_enabled
def test_huggingface_load_missing(temp_models_cache, setup_model):
    model, model_actor = setup_model

    params = HuggingfaceLoadParams(
        weights_path="weights.model",
        model_name="missing",  # Mock func raises exception
        model_version="v1.0.0",
        huggingface_userspace="my_user",
    )
    loader = HuggingfaceLoader(
        project=PROJECT,
        model=model,
        model_actor=model_actor,
        params=params,
    )
    result = loader.load()

    assert result["project_id"] == "test_project_id"
    assert result["model_id"] == model.id
    assert result["message"] == "repository not found!"

    # Check model updated to completed, with 100% completion
    assert UPDATES[-1]["model_id"] == model.id
    assert UPDATES[-1]["updates"].status == "failed"

    # Check model has not been saved
    model_path = temp_models_cache.joinpath(model.id, "weights.model")
    assert not model_path.exists()


@patch("ray.get", lambda val: val)
@patch("toktagger.api.models.loaders.send_model_updates", mock_send_model_updates)
@patch(
    "toktagger.api.models.loaders.hf_hub_download",
    mock_hf_hub_download,
)
@pytest.mark.models_enabled
def test_huggingface_load_invalid(temp_models_cache, setup_model):
    model, model_actor = setup_model

    params = HuggingfaceLoadParams(
        weights_path="weights.model",
        model_name="invalid",  # Mock func writes unreadable bytes
        model_version="v1.0.0",
        huggingface_userspace="my_user",
    )
    loader = HuggingfaceLoader(
        project=PROJECT,
        model=model,
        model_actor=model_actor,
        params=params,
    )
    result = loader.load()

    assert result["project_id"] == "test_project_id"
    assert result["model_id"] == model.id
    assert "'utf-8' codec can't decode byte" in result["message"]

    # Check model updated to completed, with 100% completion
    assert UPDATES[-1]["model_id"] == model.id
    assert UPDATES[-1]["updates"].status == "failed"

    # Check model has not been saved
    model_path = temp_models_cache.joinpath(model.id, "weights.model")
    assert not model_path.exists()

    # Check temp downloaded file deleted
    assert not temp_models_cache.joinpath("downloaded.model").exists()
