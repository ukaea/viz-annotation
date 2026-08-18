# tests/test_settings.py

import pathlib
import tempfile

import pydantic
import pytest
import tomllib
from pydantic_settings import SettingsConfigDict

from scripts.generate_example_config import create_default_toml_file
from toktagger.api.config import Settings

ENV_VARS = [
    "SERVER_HOST",
    "SERVER_PORT",
    "SERVER_RELOAD",
    "SERVER_WORKERS",
    "SERVER_CACHE_DIR",
    "DATABASE_MONGO_URL",
    "AUTH_SECRET_KEY",
    "UDA_HOST",
    "UDA_META_PLUGINNAME",
    "UDA_METANEW_PLUGINNAME",
    "SAL_HOST",
    "MODELS_CACHE_DIR",
    "MODELS_MAX_ACTORS",
    "MODELS_MAX_GPU_ACTORS",
    "MODELS_FORCE_NUM_GPUS",
    "MODELS_LOAD_SAFETENSORS_ONLY",
    "MODELS_LOCAL_LOAD_ENABLED",
    "MODELS_GITLAB_LOAD_ENABLED",
    "MODELS_GITLAB_URL",
    "MODELS_GITLAB_TOKEN",
    "MODELS_GITLAB_PROJECT_ID",
    "MODELS_HUGGINGFACE_LOAD_ENABLED",
    "MODELS_HUGGINGFACE_USERSPACE",
]


@pytest.fixture
def setup_test_settings(monkeypatch):
    """
    A Settings subclass that reads TOML from a temp file instead of the real
    project working directory.
    """
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml") as tempf:

        class TestSettings(Settings):
            model_config = SettingsConfigDict(
                toml_file=tempf.name,
                env_nested_delimiter="_",
                env_nested_max_split=1,
            )

        yield TestSettings, tempf


def test_default_settings(setup_test_settings):
    TestSettings, _ = setup_test_settings

    settings = TestSettings()

    assert settings.server.host == "localhost"
    assert settings.server.port == 8002
    assert settings.server.reload is False
    assert settings.server.workers == 1
    assert isinstance(settings.server.cache_dir, pathlib.Path)

    assert settings.database.mongo_url == "./toktagger_db"

    assert settings.auth.secret_key is None

    assert settings.uda.host == "uda2.mast.l"
    assert settings.uda.meta_pluginname == "MASTU_DB"
    assert settings.uda.metanew_pluginname == "MAST_DB"

    assert settings.sal.host == "https://sal.jetdata.eu"

    assert isinstance(settings.models.cache_dir, pathlib.Path)
    assert settings.models.max_actors == 5
    assert settings.models.max_gpu_actors is None
    assert settings.models.force_num_gpus is False
    assert settings.models.load_safetensors_only is False

    assert settings.models.local_load_enabled is True

    assert settings.models.gitlab_load_enabled is True
    assert settings.models.gitlab_url is None
    assert settings.models.gitlab_token is None
    assert settings.models.gitlab_project_id is None

    assert settings.models.huggingface_load_enabled is True
    assert settings.models.huggingface_userspace is None


def test_env_overrides_simple_nested_fields(
    monkeypatch,
    setup_test_settings,
):
    TestSettings, _ = setup_test_settings

    monkeypatch.setenv("SERVER_HOST", "0.0.0.0")
    monkeypatch.setenv("SERVER_PORT", "9000")
    monkeypatch.setenv("SERVER_RELOAD", "true")
    monkeypatch.setenv("SERVER_WORKERS", "4")
    monkeypatch.setenv("AUTH_SECRET_KEY", "test-secret")
    monkeypatch.setenv("UDA_HOST", "uda-test-host")
    monkeypatch.setenv("SAL_HOST", "https://sal.example.com")

    settings = TestSettings()

    assert settings.server.host == "0.0.0.0"
    assert settings.server.port == 9000
    assert settings.server.reload is True
    assert settings.server.workers == 4
    assert settings.auth.secret_key == "test-secret"
    assert settings.uda.host == "uda-test-host"
    assert settings.sal.host == "https://sal.example.com"


def test_env_overrides_fields_with_underscores(
    monkeypatch,
    setup_test_settings,
):
    TestSettings, _ = setup_test_settings

    monkeypatch.setenv(
        "DATABASE_MONGO_URL",
        "mongodb://user:pass@mongo:27017",
    )
    monkeypatch.setenv("UDA_META_PLUGINNAME", "TEST_META")
    monkeypatch.setenv("UDA_METANEW_PLUGINNAME", "TEST_METANEW")
    monkeypatch.setenv("MODELS_MAX_ACTORS", "10")
    monkeypatch.setenv("MODELS_MAX_GPU_ACTORS", "2")
    monkeypatch.setenv("MODELS_FORCE_NUM_GPUS", "true")

    settings = TestSettings()

    assert settings.database.mongo_url == "mongodb://user:pass@mongo:27017"
    assert settings.uda.meta_pluginname == "TEST_META"
    assert settings.uda.metanew_pluginname == "TEST_METANEW"
    assert settings.models.max_actors == 10
    assert settings.models.max_gpu_actors == 2
    assert settings.models.force_num_gpus is True


def test_toml_loading(setup_test_settings):
    TestSettings, toml_file = setup_test_settings

    toml_file.write(
        """
        [server]
        host = "127.0.0.1"
        port = 9999
        reload = true
        cache_dir = "/tmp/toktagger-cache"

        [database]
        mongo_url = "mongodb://mongo:27017"

        [uda]
        host = "uda.example.com"
        meta_pluginname = "CUSTOM_META"
        metanew_pluginname = "CUSTOM_METANEW"

        [sal]
        host = "https://sal.example.com"

        [models]
        cache_dir = "/tmp/toktagger-models"
        max_actors = 3
        max_gpu_actors = 2
        force_num_gpus = true
        load_safetensors_only = true
        local_load_enabled = false

        gitlab_load_enabled = false
        gitlab_url = "https://gitlab.example.com"
        gitlab_token = "toml-token"
        gitlab_project_id = 456

        huggingface_load_enabled = false
        huggingface_userspace = "toml-userspace"
        """
    )
    toml_file.flush()

    settings = TestSettings()

    assert settings.server.host == "127.0.0.1"
    assert settings.server.port == 9999
    assert settings.server.reload is True
    assert settings.server.cache_dir == pathlib.Path("/tmp/toktagger-cache")

    assert settings.database.mongo_url == "mongodb://mongo:27017"

    assert settings.uda.host == "uda.example.com"
    assert settings.uda.meta_pluginname == "CUSTOM_META"
    assert settings.uda.metanew_pluginname == "CUSTOM_METANEW"

    assert settings.sal.host == "https://sal.example.com"

    assert settings.models.cache_dir == pathlib.Path("/tmp/toktagger-models")
    assert settings.models.max_actors == 3
    assert settings.models.max_gpu_actors == 2
    assert settings.models.force_num_gpus is True
    assert settings.models.load_safetensors_only is True
    assert settings.models.local_load_enabled is False

    assert settings.models.gitlab_load_enabled is False
    assert settings.models.gitlab_url == "https://gitlab.example.com"
    assert settings.models.gitlab_token == "toml-token"
    assert settings.models.gitlab_project_id == 456

    assert settings.models.huggingface_load_enabled is False
    assert settings.models.huggingface_userspace == "toml-userspace"


def test_env_takes_precedence_over_toml(
    monkeypatch,
    setup_test_settings,
):
    TestSettings, toml_file = setup_test_settings

    toml_file.write(
        """
        [server]
        host = "toml-host"
        port = 1111
        """
    )
    toml_file.flush()

    monkeypatch.setenv("SERVER_HOST", "env-host")
    monkeypatch.setenv("SERVER_PORT", "2222")

    settings = TestSettings()

    assert settings.server.host == "env-host"
    assert settings.server.port == 2222


def test_env_and_toml_applied(monkeypatch, setup_test_settings):
    TestSettings, toml_file = setup_test_settings

    toml_file.write(
        """
        [server]
        host = "toml-host"
        port = 1111
        """
    )
    toml_file.flush()

    monkeypatch.setenv("SERVER_PORT", "2222")

    settings = TestSettings()

    assert settings.server.host == "toml-host"
    assert settings.server.port == 2222


def test_init_kwargs_take_precedence_over_env_and_toml(
    monkeypatch,
    setup_test_settings,
):
    TestSettings, toml_file = setup_test_settings

    toml_file.write(
        """
        [server]
        host = "toml-host"
        port = 1111
        """
    )
    toml_file.flush()

    monkeypatch.setenv("SERVER_HOST", "env-host")
    monkeypatch.setenv("SERVER_PORT", "2222")

    settings = TestSettings(
        server={
            "host": "init-host",
            "port": 3333,
        }
    )

    assert settings.server.host == "init-host"
    assert settings.server.port == 3333


def test_invalid_models_max_actors_rejected(setup_test_settings):
    TestSettings, _ = setup_test_settings

    with pytest.raises(pydantic.ValidationError):
        TestSettings(models={"max_actors": 0})


def test_invalid_models_max_gpu_actors_rejected(setup_test_settings):
    TestSettings, _ = setup_test_settings

    with pytest.raises(pydantic.ValidationError):
        TestSettings(models={"max_gpu_actors": 0})


def test_invalid_gitlab_project_id_rejected(setup_test_settings):
    TestSettings, _ = setup_test_settings

    with pytest.raises(pydantic.ValidationError):
        TestSettings(
            models={
                "gitlab_project_id": "not-an-integer",
            }
        )


def test_invalid_server_port_rejected(setup_test_settings):
    TestSettings, _ = setup_test_settings

    with pytest.raises(pydantic.ValidationError):
        TestSettings(server={"port": "not-a-port"})


def test_path_env_vars_are_converted_to_paths(
    monkeypatch,
    setup_test_settings,
):
    TestSettings, _ = setup_test_settings

    monkeypatch.setenv("SERVER_CACHE_DIR", "/tmp/server-cache")
    monkeypatch.setenv("MODELS_CACHE_DIR", "/tmp/models-cache")

    settings = TestSettings()

    assert settings.server.cache_dir == pathlib.Path("/tmp/server-cache")
    assert settings.models.cache_dir == pathlib.Path("/tmp/models-cache")


def test_create_toml():
    with tempfile.TemporaryDirectory() as directory:
        example_path = pathlib.Path(directory).joinpath("example.toml")

        create_default_toml_file(example_path)

        with example_path.open("rb") as toml_file:
            example_toml = tomllib.load(toml_file)

        assert all(
            key in example_toml
            for key in ("database", "models", "sal", "server", "uda")
        )
