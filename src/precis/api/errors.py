"""Map the domain error hierarchy to RFC 7807 problem responses.

Because every precis failure descends from ``PrecisError``, the API can render
stable, typed problem details without each route handling errors itself.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from precis.api.schemas import ProblemDetail
from precis.domain.errors import (
    ConfigError,
    LLMError,
    ParseError,
    PipelineError,
    PrecisError,
    PromptNotFoundError,
)
from precis.observability import get_logger

_logger = get_logger(__name__)

# Most-specific first; resolved by isinstance walk.
_STATUS_BY_TYPE: list[tuple[type[PrecisError], int]] = [
    (PromptNotFoundError, 404),
    (ParseError, 422),
    (ConfigError, 500),
    (LLMError, 502),  # includes StructuredOutputError (upstream model failure)
    (PipelineError, 500),
]

PROBLEM_JSON = "application/problem+json"


def _status_for(exc: PrecisError) -> int:
    for error_type, status in _STATUS_BY_TYPE:
        if isinstance(exc, error_type):
            return status
    return 500


def install_exception_handlers(app: FastAPI) -> None:
    """Register the PrecisError -> problem+json handler on the app."""

    @app.exception_handler(PrecisError)
    async def _handle_precis_error(request: Request, exc: PrecisError) -> JSONResponse:
        status = _status_for(exc)
        _logger.warning(
            "request_failed",
            path=request.url.path,
            error=type(exc).__name__,
            status=status,
        )
        problem = ProblemDetail(
            title=type(exc).__name__,
            status=status,
            detail=str(exc),
        )
        return JSONResponse(
            status_code=status,
            content=problem.model_dump(),
            media_type=PROBLEM_JSON,
        )
