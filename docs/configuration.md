# Configuration Options
The following options can be configured within TokTagger to improve your experience. They can either be set via a `toktagger.toml` configuration file in your working directory, or via environment variables. Environment variables will take precidence over settings within the TOML file.

!!! tip
    You can find an [example TokTagger TOML file here](https://github.com/ukaea/toktagger/blob/main/toktagger.example.toml), or you can generate one locally by running `python -m scripts.generate_example_config` (requires TokTagger v0.2.1 or later.) 


## Server settings
These settings should be defined under the `[server]` heading in the TOML file:

| Setting         | Environment Variable    | Type         | Default                                 | Description                                                              |
|-----------------|-------------------------|--------------|-----------------------------------------|--------------------------------------------------------------------------|
| host            | SERVER_HOST             | str          | localhost                               | Address of the host to launch TokTagger on.                              |
| port            | SERVER_PORT             | int          | 8002                                    | The port to use for the TokTagger Rest API.                              |
| reload          | SERVER_RELOAD           | bool         | False                                   | Whether to hot reload the TokTagger server on changes to files.          |
| cache_dir       | SERVER_CACHE_DIR        | pathlib.Path | ~/.cache/toktagger                      | The directory to use for storing entries in the Mongita database.        |

## Database Settings
These settings should be defined under the `[database]` heading in the TOML file:

| Setting         | Environment Variable    | Type         | Default                                 | Description                                                                                 |
|-----------------|-------------------------|--------------|-----------------------------------------|---------------------------------------------------------------------------------------------|
| mongo_url       | DATABASE_MONGO_URL      | str          | ./toktagger_db                          | URL of the MongoDB server to connect to as a backend, by default uses local Mongita client. |

## Models Settings
These settings should be defined under the `[models]` heading in the TOML file:

| Setting         | Environment Variable    | Type         | Default                                 | Description                                                              |
|-----------------|-------------------------|--------------|-----------------------------------------|--------------------------------------------------------------------------|
| cache_dir       | MODELS_CACHE_DIR        | pathlib.Path | ~/.cache/toktagger/models               | The directory to use for storing ML model weights.                       |
| max_actors      | MODELS_MAX_ACTORS       | int          | 5                                       | The maximum number of ML models which can be loaded concurrently.        |
| local_load_enabled      | MODELS_LOCAL_LOAD_ENABLED       | bool          | true                                       | Whether to enable the loading of model weights files from local disk. Should be disabled for production servers.        |


## UDA Connection Settings
These settings should be defined under the `[uda]` heading in the TOML file:

| Setting            | Environment Variable    | Type         | Default                                 | Description                                                              |
|--------------------|-------------------------|--------------|-----------------------------------------|--------------------------------------------------------------------------|
| host               | UDA_HOST                | str          | uda2.mast.l                             | Host name for the UDA server to connect to for MAST data loaders.        |
| meta_pluginname    | UDA_META_PLUGINNAME     | str          | MASTU_DB                                | ???                                                                      |
| metanew_pluginname | UDA_METANEW_PLUGINNAME  | str        | MAST_DB                                   | ???                                                                      |

## SAL Connection Settings
These settings should be defined under the `[sal]` heading in the TOML file:

| Setting         | Environment Variable    | Type         | Default                                 | Description                                                              |
|-----------------|-------------------------|--------------|-----------------------------------------|--------------------------------------------------------------------------|
| host            | SAL_HOST                | str          | https://sal.jetdata.eu                  | URL for the SAL server to connect to for JET data loaders.               |