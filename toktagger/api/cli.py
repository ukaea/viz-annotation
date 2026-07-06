import webbrowser
import argparse
from toktagger.api.main import Server
import toktagger.api.config as config
from toktagger.api.models import models_dependencies_installed
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
    webbrowser.open(f"http://{host}:{port}/ui/projects")


def main():
    print("""

  ▗▄▄▄▖▗▄▖ ▗▖ ▗▖▗▄▄▄▖▗▄▖  ▗▄▄▖ ▗▄▄▖▗▄▄▄▖▗▄▄▖ 
    █ ▐▌ ▐▌▐▌▗▞▘  █ ▐▌ ▐▌▐▌   ▐▌   ▐▌   ▐▌ ▐▌
    █ ▐▌ ▐▌▐▛▚▖   █ ▐▛▀▜▌▐▌▝▜▌▐▌▝▜▌▐▛▀▀▘▐▛▀▚▖
    █ ▝▚▄▞▘▐▌ ▐▌  █ ▐▌ ▐▌▝▚▄▞▘▝▚▄▞▘▐▙▄▄▖▐▌ ▐▌

    """)
    argparser = argparse.ArgumentParser(description="Run the FastAPI application")
    argparser.add_argument(
        "--host", help="Host to run the app on, by default localhost"
    )
    argparser.add_argument(
        "--port", type=int, help="Port to run the app on, by default 8002"
    )
    argparser.add_argument(
        "--no-browser", action="store_true", help="Don't open a browser"
    )
    argparser.add_argument(
        "--reload",
        action="store_true",
        help="Reload the API on changes, by default False",
    )
    args = argparser.parse_args()
    open_browser = not args.no_browser
    if open_browser:
        threading.Thread(target=do_open_browser, args=(args.host, args.port)).start()

    if args.host:
        config.settings.server.host = args.host
    if args.port:
        config.settings.server.port = args.port
    if args.reload:
        config.settings.server.reload = args.reload

    uvicorn.run(
        "toktagger.api.cli:create_app",
        factory=True,
        host=config.settings.server.host,
        port=config.settings.server.port,
        reload=config.settings.server.reload,
    )


if __name__ == "__main__":
    main()
