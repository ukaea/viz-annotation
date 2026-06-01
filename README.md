# TokTagger

![TokTagger Logo](docs/assets/logo_small.png)

An open source, interactive annotation platform for Tokamak diagnostic data.

[![Workflow: CI](https://github.com/ukaea/toktagger/actions/workflows/ci.yml/badge.svg)](https://github.com/ukaea/toktagger/actions/workflows/ci.yml)
[![Workflow: Dependabot](https://img.shields.io/badge/Dependabot-enabled-34d058?logo=github)](https://github.com/ukaea/toktagger/actions/workflows/dependabot/dependabot-updates)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-purple)](https://github.com/astral-sh/ruff)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-red)](https://github.com/pylint-dev/pylint-pytest)


## What It Does

TokTagger is a web-based platform for curating labeled datasets from tokamak diagnostics. It lets users browse shots, inspect signals and images, apply consistent labels, and manage annotations in one place. The Python API and React UI support local or team workflows, making it straightforward to create datasets for downstream analysis and machine-learning models.

It currently supports the following features:

- **Data Browsing**: Explore tokamak shots, signals, and images through an intuitive interface.
- **Annotation Tools**: Apply consistent labels to signals and images using a customizable tagging system.
- **ML Models**: Train and infer from ML models within the UI.
- **Dataset Management**: Organize and manage annotations in a central repository.
- **Multi-User Support**: Role-based access control with per-project membership, suitable for team annotation workflows.
- **Extensible API**: A Python API for integrating with existing workflows and tools.


## Installation

To run the application locally:

### Install via pip
To install the package via `pip` (or similarly via `Poetry` or `uv` package managers):
```sh
python -m venv .venv
source .venv/bin/activate
```
To install the package for labelling only (without ML Model functionality):
```sh
pip install toktagger
```
Or to include the ML models:
```sh
pip install toktagger[models]
```
If you intend to add custom data loaders or models to your TokTagger instance, this is the recommended route.

### Install as a uv tool
Alternatively, it can be installed as a tool using `uv`. To install the package for labelling only (without ML Model functionality):

```sh
uv tool install --python 3.12.6 toktagger
```
Or to include the ML models:
```sh
uv tool install --python 3.12.6 toktagger[models]
```

## Quick Start

To start a local single-user instance:

```sh
toktagger
```

This starts the application at `http://localhost:8002`. On first launch an `admin` account is created automatically and the credentials are printed to the terminal.

### Multi-User / Team Deployment

For concurrent multi-user access, run with multiple Gunicorn workers:

```sh
toktagger --workers 4 --host 0.0.0.0 --port 8002 --no-browser
```

Or directly via Gunicorn:

```sh
gunicorn toktagger.api.asgi:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 4 \
    --bind 0.0.0.0:8002
```

With Docker Compose, the production stack defaults to 4 workers. Override with the `WORKERS` environment variable:

```sh
WORKERS=8 docker compose up
```

See the [User Management](docs/user_management.md) guide for creating accounts, assigning roles, and managing project membership.
