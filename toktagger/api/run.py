import subprocess
import uvicorn
import os

from toktagger.api.config import settings

if __name__ == "__main__":
    host = settings.server.host
    port = settings.server.port
    workers = settings.server.workers
    reload = settings.server.reload

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
