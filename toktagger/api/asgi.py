"""ASGI entry point for Gunicorn + UvicornWorker.

Usage:
    gunicorn toktagger.api.asgi:app \
        --worker-class uvicorn.workers.UvicornWorker \
        --workers 4 \
        --bind 0.0.0.0:8002
"""

from toktagger.api.cli import create_app

app = create_app()
