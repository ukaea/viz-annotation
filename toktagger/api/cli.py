import webbrowser
import argparse
import subprocess
import sys
from toktagger.api.main import Server
from toktagger.api.models import models_dependencies_installed
import uvicorn
import time
import threading
import os


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
    argparser.add_argument("--host", default="0.0.0.0", help="Host to run the app on")
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
        default=4,
        type=int,
        help="Number of Gunicorn worker processes (use 1 for single-worker uvicorn dev mode)",
    )
    args = argparser.parse_args()
    open_browser = not args.no_browser
    if open_browser:
        threading.Thread(target=do_open_browser, args=(args.host, args.port)).start()

    os.environ["API_URL"] = f"http://{args.host}:{args.port}"

    if args.workers > 1:
        if args.reload:
            print("Warning: --reload is ignored when --workers > 1 (gunicorn mode)")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "gunicorn",
                "toktagger.api.asgi:app",
                "--worker-class",
                "uvicorn.workers.UvicornWorker",
                "--workers",
                str(args.workers),
                "--bind",
                f"{args.host}:{args.port}",
            ],
            check=True,
        )
    else:
        uvicorn.run(
            "toktagger.api.cli:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=args.reload,
        )


if __name__ == "__main__":
    main()
