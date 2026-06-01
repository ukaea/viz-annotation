import subprocess
import uvicorn
import os

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8002))
    workers = int(os.environ.get("WORKERS", 1))
    reload = os.environ.get("RELOAD", "false").lower() == "true"

    os.environ["API_URL"] = f"http://{host}:{port}"

    if workers > 1:
        subprocess.run(
            [
                "gunicorn",
                "toktagger.api.asgi:app",
                "--worker-class",
                "uvicorn.workers.UvicornWorker",
                "--workers",
                str(workers),
                "--bind",
                f"{host}:{port}",
            ],
            check=True,
        )
    else:
        uvicorn.run(
            "toktagger.api.cli:create_app",
            factory=True,
            host=host,
            port=port,
            reload=reload,
        )
