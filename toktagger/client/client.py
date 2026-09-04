"""Stateless Python client for pulling TokTagger data for analysis.

See docs/dev/client-library.md for the full API specification.
"""

from __future__ import annotations

from typing import Any

import httpx

from pydantic import PrivateAttr

from toktagger.api.schemas.projects import Project as ProjectSchema
from toktagger.api.schemas.samples import Sample as SampleSchema

from toktagger.client.exceptions import (
    NotFoundError,
    TokTaggerAPIError,
    TokTaggerClientError,
)


class Project(ProjectSchema):
    # Bootstrapped model: server Project schema plus a bound client reference
    _client: TokTaggerClient | None = PrivateAttr(default=None)


class Sample(SampleSchema):
    _client: TokTaggerClient | None = PrivateAttr(default=None)


class TokTaggerClient:
    """Stateless client for the TokTagger API; every method takes explicit IDs."""

    def __init__(
        self,
        base_url: str = "http://localhost:8002",
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        # transport is a test hook, eg httpx.MockTransport
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
            headers=headers or {},
            transport=transport,
        )

    def __enter__(self) -> TokTaggerClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        self._http.close()

    def health(self) -> dict[str, Any]:
        """Check the server is running and the database is connected."""
        return self._request("GET", "/health").json()

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, object] | None = None,
        json: dict | list | None = None,
    ) -> httpx.Response:
        try:
            response = self._http.request(method, path, params=params, json=json)
        except httpx.HTTPError as e:
            raise TokTaggerClientError(
                f"Failed to reach TokTagger server at {self._base_url}: {e}"
            ) from e

        if response.status_code >= 400:
            detail = self._extract_detail(response)
            if response.status_code == 404:
                raise NotFoundError(response.status_code, detail)
            raise TokTaggerAPIError(response.status_code, detail)
        return response

    @staticmethod
    def _extract_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return response.text
        detail = body.get("detail") if isinstance(body, dict) else body
        return detail if isinstance(detail, str) else str(detail)

    def _bootstrap_project(self, data: dict[str, Any]) -> Project:
        # Attach this client so the returned model can be used for chained calls
        project = Project.model_validate(data)
        project._client = self
        return project

    def _bootstrap_sample(self, data: dict[str, Any]) -> Sample:
        sample = Sample.model_validate(data)
        sample._client = self
        return sample
