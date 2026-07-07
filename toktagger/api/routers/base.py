from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from importlib.metadata import version, PackageNotFoundError
from toktagger.api.models import models_dependencies_installed
from toktagger.api.crud import utils

router = APIRouter(
    prefix="",
    tags=["Base"],
)


@router.get("/")
def get_app(request: Request):
    """Endpoint to serve the main SPA."""
    return FileResponse(request.app.state.index_file)


@router.get("/health")
async def health_check(request: Request) -> dict:
    """Check the server is running correctly."""
    # Get version
    try:
        vers = version("toktagger")
    except PackageNotFoundError:
        vers = "unknown"

    # Check db connection
    try:
        await utils.get_projects(
            db_client=request.app.state.db_client,
        )
        db_conn = True
    except Exception:
        db_conn = False

    # If available, check whether GPUs enabled for model tasks
    model_gpu_available = False
    if hasattr(request.app.state, "task_registry"):
        model_gpu_available = request.app.state.task_registry.gpu_enabled

    # Return info
    return {
        "name": "TokTagger",
        "version": vers,
        "db_connected": db_conn,
        "models_enabled": models_dependencies_installed(),
        "gpu_available": model_gpu_available,
        "testing_mode": request.app.state.testing_mode,
    }


@router.get("/{full_path:path}")
def spa_fallback(request: Request, full_path: str):
    """Fallback route to serve the SPA's index.html for any unmatched routes.
    This ensures that refreshing pages on the frontend takes the user to the same place.
    """
    return FileResponse(request.app.state.index_file)
