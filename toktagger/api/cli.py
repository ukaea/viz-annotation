import webbrowser
import argparse
from toktagger.api.main import Server, run_with_gunicorn
from toktagger.api.models import models_dependencies_installed
from toktagger.api.config import settings
import uvicorn
import time
import threading


# Need to point to app as a module level string if we want reload option
def create_app():
    server = Server()
    server._setup_app()
    # Setup ray if required
    if models_dependencies_installed():
        server._setup_ray()
    return server.app


def do_open_browser(host: str, port: int):
    time.sleep(1)  # allow server to start
    display_host = "localhost" if host == "0.0.0.0" else host
    webbrowser.open(f"http://{display_host}:{port}/ui/projects")


def main():
    print("""

  ▗▄▄▄▖▗▄▖ ▗▖ ▗▖▗▄▄▄▖▗▄▖  ▗▄▄▖ ▗▄▄▖▗▄▄▄▖▗▄▄▖ 
    █ ▐▌ ▐▌▐▌▗▞▘  █ ▐▌ ▐▌▐▌   ▐▌   ▐▌   ▐▌ ▐▌
    █ ▐▌ ▐▌▐▛▚▖   █ ▐▛▀▜▌▐▌▝▜▌▐▌▝▜▌▐▛▀▀▘▐▛▀▚▖
    █ ▝▚▄▞▘▐▌ ▐▌  █ ▐▌ ▐▌▝▚▄▞▘▝▚▄▞▘▐▙▄▄▖▐▌ ▐▌

    """)
    argparser = argparse.ArgumentParser(description="Run the FastAPI application")
    argparser.add_argument("--host", default="localhost", help="Host to run the app on")
    argparser.add_argument(
        "--port", default=8002, type=int, help="Port to run the app on"
    )
    argparser.add_argument(
        "--no-browser", action="store_true", help="Don't open a browser"
    )
    argparser.add_argument(
        "--reload",
        action="store_true",
        help="Reload the API on changes (single-worker uvicorn only)",
    )
    argparser.add_argument(
        "--workers",
        default=1,
        type=int,
        help="Number of Gunicorn worker processes (use 1 for single-worker uvicorn dev mode)",
    )
    args = argparser.parse_args()
    open_browser = not args.no_browser
    if open_browser:
        threading.Thread(target=do_open_browser, args=(args.host, args.port)).start()

    if args.host:
        settings.server.host = args.host
    if args.port:
        settings.server.port = args.port
    if args.reload:
        settings.server.reload = args.reload
    if args.workers:
        settings.server.workers = args.workers

    if settings.server.workers > 1:
        if settings.server.reload:
            print("Warning: --reload is ignored when --workers > 1 (gunicorn mode)")
        run_with_gunicorn(
            settings.server.host,
            settings.server.port,
            settings.server.workers,
        )
    else:
        uvicorn.run(
            "toktagger.api.cli:create_app",
            factory=True,
            host=settings.server.host,
            port=settings.server.port,
            reload=settings.server.reload,
        )


if __name__ == "__main__":
    main()
