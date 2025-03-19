
## Overview

Below is a high level overview of the project structure:
```
.
├── data                # Sample experimental data
├── active_learning     # Experiments in active learning
├── notebooks           # Notebooks for exploring data
├── services            # Implementations of different apis/services
│   ├── data_api        # Data API: For pulling signals for display
│   ├── event_api       # Event API: For storing event data and tags
│   ├── model_api       # Model API: For running and quering models
│   └── ui              # UI: the front end of the application
├── README.md           # This README doc
└── docker-compose.yml  # Master docker compose for running the application
```


## Installation

1. Install `docker` and `docker compose`: https://docs.docker.com/engine/install/
2. Install and setup `git lfs`: https://git-lfs.com/

## Setup

Build the relevant dataset for the ML model locally
```sh
uv venv --python 3.12.6 
source .venv/bin/activate
uv pip install -r requirements.txt
python -m scripts.build_dataset
```

## Run

Run the application by running the following command:

```sh
docker compose up 
```

This will start the following services:

| Service URL                     | Description                |
|---------------------------------|----------------------------|
| `http://localhost:3000/`        | User Interface             |
| `http://localhost:8081/`        | MongoExpress Admin Panel   |
| `http://localhost:8000/`        | Event Database API         |
| `http://localhost:8001/`        | Model Runner API           |
| `http://localhost:8002/`        | Data API                   |