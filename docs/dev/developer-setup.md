# Development Setup

## Installation

1. Install and setup `git lfs`: https://git-lfs.com/
2. Create a new python environment and install the dependancies

```sh
uv venv --python 3.12.6 
source .venv/bin/activate
uv pip install -e .[models]
```
3. Install and setup `Node.js`: https://nodejs.org/en/download
4. Install the UI dependencies

```sh
nvm use v22.19.0
npm --prefix toktagger/ui install
```

5. Run the backend API service in development mode. The backend API will be accessible at `http://localhost:8002`.

```sh
toktagger --workers 1 --reload --no-browser
```

This starts a single Uvicorn worker with auto-reload enabled. The `--no-browser` flag suppresses the automatic browser launch since the frontend dev server (step 6) is used instead.

6. Run the frontend UI service in development mode. The UI will be accessible at `http://localhost:5173`
```sh
npm --prefix toktagger/ui run dev
```

## Development Setup with Docker

Alternatively, you can run the application in development mode using docker:

```sh
docker compose --env-file .env.dev -f docker-compose.dev.yml up --build
```

This will start both the backend API and the frontend UI at the following urls:

| Service URL                     | Description                |
|---------------------------------|----------------------------|
| `http://localhost:5173/`        | User Interface             |
| `http://localhost:8002/`        | Backend API                |
| `http://localhost:27017/`       | MongoDB Database           |
| `http://localhost:8081/`        | MongoExpress Admin Panel   |

The development setup runs both the frontend and backend in development mode, so any changes to the code will automatically be reflected in the running application.

## Setup local test data
Once the application is running, the following setup script can be used to automatically set up a basic model

Configure git LFS and pull the model

```sh
git lfs install
git lfs pull
```

Create some example datasets for testing.
```sh
python -m scripts.setup 
```

## Building the Single Page Application (SPA)

The version of the application which get served to users is built using the following command:

```sh
npm --prefix toktagger/ui run build
```

This will run vite build and create a production-ready version of the application in the `toktagger/api/static` directory. This is then packaged with the application when pip installed.