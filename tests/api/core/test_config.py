# tests/test_settings.py

import pathlib

import pydantic
import pytest
from pydantic_settings import SettingsConfigDict
import tempfile
from toktagger.api.config import Settings

from scripts.generate_example_config import create_default_toml_file
import tomllib

ENV_VARS = [
    "SERVER_HOST",
    "SERVER_PORT",
    "SERVER_RELOAD",
    "SERVER_CACHE_DIR",
    "DATABASE_MONGO_URL",
    "UDA_HOST",
    "UDA_META_PLUGINNAME",
    "UDA_METANEW_PLUGINNAME",
    "SAL_HOST",
    "MODELS_CACHE_DIR",
    "MODELS_MAX_ACTORS",
]


@pytest.fixture
def setup_test_settings(monkeypatch):
    """
    A Settings subclass that reads TOML from a temp file instead of the real
    project working directory.
    """
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    with tempfile.NamedTemporaryFile(mode="w", prefix=".toml") as tempf:

        class TestSettings(Settings):
            model_config = SettingsConfigDict(
                toml_file=tempf.name,
                env_nested_delimiter="_",
            )

        yield TestSettings, tempf


def test_default_settings(setup_test_settings):
    TestSettings, _ = setup_test_settings

    settings = TestSettings()

    assert settings.server.host == "localhost"
    assert settings.server.port == 8002
    assert settings.server.reload is False
    assert isinstance(settings.server.cache_dir, pathlib.Path)

    assert settings.database.mongo_url == "./toktagger_db"

    assert settings.uda.host == "uda2.mast.l"
    assert settings.uda.meta_pluginname == "MASTU_DB"
    assert settings.uda.metanew_pluginname == "MAST_DB"

    assert settings.sal.host == "https://sal.jetdata.eu"

    assert isinstance(settings.models.cache_dir, pathlib.Path)
    assert settings.models.max_actors == 5


def test_env_overrides_simple_nested_fields(monkeypatch, setup_test_settings):
    TestSettings, _ = setup_test_settings

    monkeypatch.setenv("SERVER_HOST", "0.0.0.0")
    monkeypatch.setenv("SERVER_PORT", "9000")
    monkeypatch.setenv("SERVER_RELOAD", "true")
    monkeypatch.setenv("UDA_HOST", "uda-test-host")
    monkeypatch.setenv("SAL_HOST", "https://sal.example.com")

    settings = TestSettings()

    assert settings.server.host == "0.0.0.0"
    assert settings.server.port == 9000
    assert settings.server.reload is True
    assert settings.uda.host == "uda-test-host"
    assert settings.sal.host == "https://sal.example.com"


def test_env_overrides_fields_with_underscores(monkeypatch, setup_test_settings):
    TestSettings, _ = setup_test_settings

    monkeypatch.setenv("DATABASE_MONGO_URL", "mongodb://user:pass@mongo:27017")
    monkeypatch.setenv("UDA_META_PLUGINNAME", "TEST_META")
    monkeypatch.setenv("UDA_METANEW_PLUGINNAME", "TEST_METANEW")
    monkeypatch.setenv("MODELS_MAX_ACTORS", "10")

    settings = TestSettings()

    assert settings.database.mongo_url == "mongodb://user:pass@mongo:27017"
    assert settings.uda.meta_pluginname == "TEST_META"
    assert settings.uda.metanew_pluginname == "TEST_METANEW"
    assert settings.models.max_actors == 10


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


def test_env_takes_precedence_over_toml(monkeypatch, setup_test_settings):
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
    monkeypatch, setup_test_settings
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


def test_invalid_server_port_rejected(setup_test_settings):
    TestSettings, _ = setup_test_settings

    with pytest.raises(pydantic.ValidationError):
        TestSettings(server={"port": "not-a-port"})


def test_path_env_vars_are_converted_to_paths(monkeypatch, setup_test_settings):
    TestSettings, _ = setup_test_settings

    monkeypatch.setenv("SERVER_CACHE_DIR", "/tmp/server-cache")
    monkeypatch.setenv("MODELS_CACHE_DIR", "/tmp/models-cache")

    settings = TestSettings()

    assert settings.server.cache_dir == pathlib.Path("/tmp/server-cache")
    assert settings.models.cache_dir == pathlib.Path("/tmp/models-cache")


def test_create_toml():
    # Create template with defaults
    with tempfile.TemporaryDirectory() as file:
        create_default_toml_file(pathlib.Path(file).joinpath("example.toml"))

        # Check it can be loaded as toml
        with pathlib.Path(file).joinpath("example.toml").open("rb") as toml_file:
            example_toml = tomllib.load(toml_file)

        assert all(
            (
                key in example_toml.keys()
                for key in ("database", "models", "sal", "server", "uda")
            )
        )
