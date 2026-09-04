"""Error hierarchy for the TokTagger client."""


class TokTaggerClientError(Exception):
    """Base error for all TokTagger client errors."""


class TokTaggerAPIError(TokTaggerClientError):
    """Non-2xx response from the TokTagger API, or no match for a client-side lookup."""

    def __init__(self, status_code: int | None, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"TokTagger API error (status {status_code}): {detail}")


class NotFoundError(TokTaggerAPIError):
    """404 response, or a by-name / by-shot-id lookup which matched nothing."""


class MultipleResultsFound(TokTaggerClientError):
    """A by-name / by-shot-id lookup matched more than one record."""
