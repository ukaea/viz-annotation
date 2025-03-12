
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

## Run

Run the application by running the following command:

```sh
docker compose up 
```

The UI will be available at `http://localhost:3000/`