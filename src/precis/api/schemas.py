"""API data-transfer objects.

DTOs are deliberately separate from domain models: they form the versioned
public contract of the HTTP API and shield internal models from wire concerns.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from precis.config.settings import DEFAULT_MODEL, DEFAULT_PROVIDER


class SummarizeTextRequest(BaseModel):
    """Request to summarize a block of text."""

    text: str = Field(min_length=1, description="The text to summarize.")
    provider: str = Field(default=DEFAULT_PROVIDER, description="LLM provider id.")
    model: str = Field(default=DEFAULT_MODEL, description="Provider model name.")
    prompt_key: str = Field(
        default="webpage_summary", description="Prompt template key to apply."
    )


class SummaryResponse(BaseModel):
    """A generated summary plus the model that produced it."""

    summary: str
    provider: str
    model: str


class PromptsResponse(BaseModel):
    """Available prompt keys."""

    keys: list[str]


class HealthResponse(BaseModel):
    """Liveness/readiness status."""

    status: str


class ProblemDetail(BaseModel):
    """RFC 7807 problem detail (``application/problem+json``)."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
