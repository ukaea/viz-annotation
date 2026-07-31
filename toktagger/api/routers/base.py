from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, RedirectResponse
from importlib.metadata import version, PackageNotFoundError
from toktagger.api.models import models_dependencies_installed
from toktagger.api.crud import utils

if models_dependencies_installed():
    import ray

router = APIRouter(
    prefix="",
    tags=["Base"],
)

_ALL_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]


@router.api_route("/", methods=_ALL_METHODS, include_in_schema=False)
def get_app(request: Request):
    """Serve the main SPA."""
    if request.method in ("GET", "HEAD"):
        return FileResponse(request.app.state.index_file)
    return RedirectResponse(url="/", status_code=303)


@router.get("/health")
async def health_check(request: Request) -> dict:
    """Check the server is running correctly."""
    try:
        vers = version("toktagger")
    except PackageNotFoundError:
        vers = "unknown"

    try:
        await utils.get_projects(
            db_client=request.app.state.db_client,
        )
        db_conn = True
    except Exception:
        db_conn = False

    # If available, check whether GPUs enabled for model tasks
    model_gpu_available = False
    if models_dependencies_installed() and hasattr(
        request.app.state, "task_registry"
    ):
        model_gpu_available = ray.get(
            request.app.state.task_registry.gpu_enabled.remote()
        )

    # Return info
    return {
        "name": "TokTagger",
        "version": vers,
        "db_connected": db_conn,
        "models_enabled": models_dependencies_installed(),
        "gpu_available": model_gpu_available,
        "testing_mode": request.app.state.testing_mode,
    }


@router.api_route("/{full_path:path}", methods=_ALL_METHODS, include_in_schema=False)
def spa_fallback(request: Request, full_path: str):
    """Fallback for SPA routing — serves index.html on GET, redirects other methods.

    Without this, a native form POST to any SPA path (e.g. /ui/login) would return
    405 because Starlette sees the path match but the GET-only method doesn't match.
    """
    if request.method in ("GET", "HEAD"):
        return FileResponse(request.app.state.index_file)
    return RedirectResponse(url=f"/{full_path}", status_code=303)
