"""Stateless Python client for pulling TokTagger data for analysis.

See docs/dev/client-library.md for the full API specification.
"""

from __future__ import annotations

from typing import Any, Literal

import httpx

from pydantic import PrivateAttr

from toktagger.api.schemas.projects import Project as ProjectSchema
from toktagger.api.schemas.samples import Sample as SampleSchema, SampleSummary

from toktagger.client.exceptions import (
    MultipleResultsFoundError,
    NotFoundError,
    TokTaggerAPIError,
    TokTaggerClientError,
)


class Project(ProjectSchema):
    # Bootstrapped model: server Project schema plus a bound client reference
    _client: TokTaggerClient | None = PrivateAttr(default=None)

    def _require_client(self) -> TokTaggerClient:
        if self._client is None:
            raise TokTaggerClientError(
                "This Project has no client attached; it was not created by a "
                "TokTaggerClient. Create it via e.g. TokTaggerClient.get_project() "
                "to use its methods."
            )
        return self._client

    def list_samples(
        self,
        shot_id: int | None = None,
        start: int = 0,
        count: int = 10,
        sort_by: str = "_id",
        sort_direction: Literal["ascending", "descending"] = "descending",
    ) -> list[Sample]:
        return self._require_client().list_samples(
            self.id,
            shot_id=shot_id,
            start=start,
            count=count,
            sort_by=sort_by,
            sort_direction=sort_direction,
        )

    def get_sample(self, sample_id: str) -> Sample:
        return self._require_client().get_sample(self.id, sample_id)

    def get_sample_by_shot_id(self, shot_id: int) -> Sample:
        return self._require_client().get_sample_by_shot_id(self.id, shot_id)

    def get_samples_summary(self) -> SampleSummary:
        return self._require_client().get_samples_summary(self.id)


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

    def list_projects(
        self,
        name: str | None = None,
        start: int = 0,
        count: int = 10,
        sort_by: str = "_id",
        sort_direction: Literal["ascending", "descending"] = "descending",
    ) -> list[Project]:
        params = {
            "start": start,
            "count": count,
            "sort_by": sort_by,
            "sort_direction": sort_direction,
        }
        if name is not None:
            params["name"] = name

        data = self._request("GET", "/projects", params=params).json()
        return [self._bootstrap_project(project) for project in data]

    def get_project(self, project_id: str) -> Project:
        data = self._request("GET", f"/projects/{project_id}").json()
        return self._bootstrap_project(data)

    def get_project_by_name(self, name: str) -> Project:
        projects = self.list_projects(name=name)
        if len(projects) == 0:
            raise NotFoundError(None, f"No project found matching name '{name}'.")
        if len(projects) > 1:
            raise MultipleResultsFoundError(
                f"{len(projects)} projects matched name '{name}'; "
                "use get_project() with a specific ID."
            )
        return projects[0]

    def list_samples(
        self,
        project_id: str,
        shot_id: int | None = None,
        start: int = 0,
        count: int = 10,
        sort_by: str = "_id",
        sort_direction: Literal["ascending", "descending"] = "descending",
    ) -> list[Sample]:
        params = {
            "start": start,
            "count": count,
            "sort_by": sort_by,
            "sort_direction": sort_direction,
        }
        if shot_id is not None:
            params["shot_id"] = shot_id
        data = self._request(
            "GET", f"/projects/{project_id}/samples", params=params
        ).json()
        return [self._bootstrap_sample(sample) for sample in data]

    def get_sample(self, project_id: str, sample_id: str) -> Sample:
        data = self._request(
            "GET", f"/projects/{project_id}/samples/{sample_id}"
        ).json()
        return self._bootstrap_sample(data)

    def get_sample_by_shot_id(self, project_id: str, shot_id: int) -> Sample:
        samples = self.list_samples(project_id, shot_id=shot_id)
        if len(samples) == 0:
            raise NotFoundError(
                None,
                f"No sample found with shot ID {shot_id} in project {project_id}.",
            )
        if len(samples) > 1:
            raise MultipleResultsFoundError(
                f"{len(samples)} samples matched shot ID {shot_id} in project "
                f"{project_id}; use get_sample() with a specific ID."
            )
        return samples[0]

    def get_samples_summary(self, project_id: str) -> SampleSummary:
        data = self._request("GET", f"/projects/{project_id}/samples/summary").json()
        return SampleSummary.model_validate(data)

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
