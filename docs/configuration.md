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
| workers         | SERVER_WORKERS          | int          | 1                                        | The number of Gunicorn worker processes to use. If set to 1, runs a single-process uvicorn server instead. |
| cache_dir       | SERVER_CACHE_DIR        | pathlib.Path | ~/.cache/toktagger                      | The directory to use for storing entries in the Mongita database.        |

## Database Settings
These settings should be defined under the `[database]` heading in the TOML file:

| Setting         | Environment Variable    | Type         | Default                                 | Description                                                                                 |
|-----------------|-------------------------|--------------|-----------------------------------------|---------------------------------------------------------------------------------------------|
| mongo_url       | DATABASE_MONGO_URL      | str          | ./toktagger_db                          | URL of the MongoDB server to connect to as a backend, by default uses local Mongita client. |

## Auth Settings
These settings should be defined under the `[auth]` heading in the TOML file:

| Setting         | Environment Variable    | Type         | Default                                 | Description                                                              |
|-----------------|-------------------------|--------------|-----------------------------------------|--------------------------------------------------------------------------|
| secret_key      | AUTH_SECRET_KEY         | str          | None                                    | Secret key used to sign auth tokens. If unset, a key is generated and persisted to `secret.key` under the server `cache_dir` on first run. Set this explicitly for multi-worker/multi-process deployments so all processes share the same signing key. |

## Models Settings

These settings should be defined under the `[models]` heading in the TOML file:

| Setting | Environment Variable | Type | Default | Description |
|---------|----------------------|------|---------|-------------|
| cache_dir | MODELS_CACHE_DIR | pathlib.Path | ~/.cache/toktagger/models | The directory to use for storing ML model weights. |
| max_actors | MODELS_MAX_ACTORS | int \| None | 5 | The maximum number of ML models which can be loaded concurrently, set to None to detect automatically and use all available cores. |
| max_gpu_actors | MODELS_MAX_GPU_ACTORS | int \| None | None | The maximum number of GPUs to use for ML model tasks, leave blank to detect automatically and use all available cores. |
| force_num_gpus | MODELS_FORCE_NUM_GPUS | bool | false | Force the set number of GPU actors available, even if insufficient available GPU cores detected on hardware. |
| load_safetensors_only | MODELS_LOAD_SAFETENSORS_ONLY | bool | false | Whether to only allow loading of SafeTensors files for added security. |
| local_load_enabled | MODELS_LOCAL_LOAD_ENABLED | bool | true | Whether to enable the loading of model weights files from local disk. Should be disabled for production servers. |
| gitlab_load_enabled | MODELS_GITLAB_LOAD_ENABLED | bool | true | Whether to enable the loading of model weights files from Gitlab. |
| gitlab_url | MODELS_GITLAB_URL | str \| None | None | The URL of the Gitlab server to load ML model weights from. |
| gitlab_token | MODELS_GITLAB_TOKEN | str \| None | None | The PAT Token to use when connecting to Gitlab to load model weights. |
| gitlab_project_id | MODELS_GITLAB_PROJECT_ID | int \| None | None | Limit the user to load ML model weights from a specific Gitlab project. Leave blank to allow the user to choose. |
| huggingface_load_enabled | MODELS_HUGGINGFACE_LOAD_ENABLED | bool | true | Whether to enable the loading of model weights files from Hugging Face. |
| huggingface_userspace | MODELS_HUGGINGFACE_USERSPACE | str \| None | None | Limit the user to load ML model weights from a specific Hugging Face userspace / organisation. Leave blank to allow the user to choose. |

!!! note
    TOML does not support `None` or `null` values. To leave an optional setting
    unset, omit it from the TOML file or leave the corresponding line commented
    out.


## UDA Connection Settings
These settings should be defined under the `[uda]` heading in the TOML file:

| Setting            | Environment Variable    | Type         | Default                                 | Description                                                              |
|--------------------|-------------------------|--------------|-----------------------------------------|--------------------------------------------------------------------------|
| host               | UDA_HOST                | str          | uda2.mast.l                             | Host name for the UDA server to connect to for MAST data loaders.        |
| meta_pluginname    | UDA_META_PLUGINNAME     | str          | MASTU_DB                                | Database location for MAST-U data                                                                      |
| metanew_pluginname | UDA_METANEW_PLUGINNAME  | str        | MAST_DB                                   | Database location for MAST data                                                                      |

## SAL Connection Settings
These settings should be defined under the `[sal]` heading in the TOML file:

| Setting         | Environment Variable    | Type         | Default                                 | Description                                                              |
|-----------------|-------------------------|--------------|-----------------------------------------|--------------------------------------------------------------------------|
| host            | SAL_HOST                | str          | https://sal.jetdata.eu                  | URL for the SAL server to connect to for JET data loaders.               |