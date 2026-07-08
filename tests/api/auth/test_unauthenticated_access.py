"""
Sweeps every registered API route and confirms an unauthenticated caller (no
Authorization header at all) can't get anything out of it.

`get_current_user` is wired in as a router-level dependency on every business
router, so it should reject with 401 before any path/query/body validation or
database work happens for that route — this asserts that holds everywhere,
not just the handful of endpoints covered individually elsewhere.
"""

import re

import pytest
import pytest_asyncio

from toktagger.api.main import Server

# Explicitly public: login, health/docs, and the SPA/static catch-alls served by
# the base router (no application data, and must be reachable pre-login).
_PUBLIC_PATHS = {
    "/",
    "/{full_path:path}",
    "/health",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/openapi.json",
}
_PUBLIC_METHOD_PATHS = {("POST", "/auth/token")}


def _protected_routes() -> list[tuple[str, str]]:
    """Introspect the FastAPI app for every (method, path) expected to require auth."""
    server = Server()
    server._setup_app()

    routes = []
    for route in server.app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if not methods or path is None or path in _PUBLIC_PATHS:
            continue
        for method in sorted(methods):
            if method in ("HEAD", "OPTIONS"):
                continue
            if (method, path) in _PUBLIC_METHOD_PATHS:
                continue
            routes.append((method, path))
    return routes


def _fill_path(path: str) -> str:
    """Replace every {param} with a placeholder — auth is checked before path params are used."""
    return re.sub(r"\{[^}]+\}", "x", path)


def _is_models_route(path: str) -> bool:
    """Model routes are also gated by check_models_enabled, which returns 503 (not
    401) when the optional `models` extra isn't installed."""
    return "models" in path.strip("/").split("/")


@pytest_asyncio.fixture
async def unauthenticated_client(api_client, auth_db_client):
    """api_client wired to an (empty) db with auth_required=True — no users needed."""
    api_client.app.state.db_client = auth_db_client
    api_client.app.state.auth_required = True
    return api_client


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", _protected_routes())
async def test_unauthenticated_request_is_rejected(
    unauthenticated_client, method, path
):
    resp = await unauthenticated_client.request(method, _fill_path(path))

    if _is_models_route(path):
        assert resp.status_code in (401, 503), f"{method} {path} -> {resp.status_code}"
    else:
        assert resp.status_code == 401, f"{method} {path} -> {resp.status_code}"
