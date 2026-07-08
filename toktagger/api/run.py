import uvicorn
import os

from toktagger.api.config import settings
from toktagger.api.main import run_with_gunicorn

if __name__ == "__main__":
    host = settings.server.host
    port = settings.server.port
    workers = settings.server.workers
    reload = settings.server.reload

    os.environ["API_URL"] = f"http://{host}:{port}"

    if workers > 1:
        run_with_gunicorn(host, port, workers)
    else:
        uvicorn.run(
            "toktagger.api.cli:create_app",
            factory=True,
            host=host,
            port=port,
            reload=reload,
        )
