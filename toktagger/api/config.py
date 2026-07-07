from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)
import pydantic
import typing
import pathlib
from platformdirs import user_cache_dir


class UDA(pydantic.BaseModel):
    host: str = pydantic.Field(
        "uda2.mast.l",
        description="Host name for the UDA server to connect to for MAST data loaders.",
    )
    meta_pluginname: str = pydantic.Field(
        "MASTU_DB",
        description="Database location for MAST-U data",
    )
    metanew_pluginname: str = pydantic.Field(
        "MAST_DB",
        description="Database location for MAST data",
    )


class SAL(pydantic.BaseModel):
    host: str = pydantic.Field(
        "https://sal.jetdata.eu",
        description="URL for the SAL server to connect to for JET data loaders.",
    )


class Database(pydantic.BaseModel):
    mongo_url: str = pydantic.Field(
        "./toktagger_db",
        description="URL of the MongoDB server to connect to as a backend. If not set, uses a local mongita client.",
    )


class Server(pydantic.BaseModel):
    host: str = pydantic.Field(
        "localhost",
        description="Address of the host to launch TokTagger on.",
    )
    port: int = pydantic.Field(
        8002,
        description="The port to use for the TokTagger Rest API.",
    )
    reload: bool = pydantic.Field(
        False,
        description="Whether to hot reload the TokTagger server on changes to files.",
    )
    cache_dir: pathlib.Path = pydantic.Field(
        user_cache_dir("toktagger", "ukaea"),
        description="The directory to use for storing entries in the Mongita database, if used.",
        validate_default=True,
    )


class Models(pydantic.BaseModel):
    cache_dir: pathlib.Path = pydantic.Field(
        pathlib.Path(user_cache_dir("toktagger", "ukaea")).joinpath("models"),
        description="The directory to use for storing ML model weights.",
        validate_default=True,
    )
    max_actors: typing.Annotated[
        int | None,
        pydantic.Field(
            default=5,
            description="The maximum number of ML models which can be loaded concurrently, set to None to detect automatically and use all available cores.",
            gt=0,
        ),
    ]
    max_gpu_actors: typing.Annotated[
        int | None,
        pydantic.Field(
            default=None,
            description="The maximum number of GPUs to use for ML model tasks, leave blank to detect automatically and use all available cores.",
            gt=0,
        ),
    ]
    force_num_gpus: bool = pydantic.Field(
        False,
        description="Force the set number of GPU actors available, even if insufficient available GPU cores detected on hardware.",
    )
    local_load_enabled: bool = pydantic.Field(
        True,
        description="Whether to enable the loading of model weights files from local disk. Should be disabled for production servers.",
    )


class Settings(BaseSettings):
    server: Server = pydantic.Field(default_factory=Server)
    database: Database = pydantic.Field(default_factory=Database)
    uda: UDA = pydantic.Field(default_factory=UDA)
    sal: SAL = pydantic.Field(default_factory=SAL)
    models: Models = pydantic.Field(default_factory=Models)

    model_config = SettingsConfigDict(
        toml_file="toktagger.toml",
        env_nested_delimiter="_",
        env_nested_max_split=1,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
            dotenv_settings,
        )


settings = Settings()
