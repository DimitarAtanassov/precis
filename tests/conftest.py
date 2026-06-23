"""Shared pytest fixtures and test doubles.

These doubles let the suite exercise the real orchestration code paths
(summarizer pipeline, services) deterministically and without any network or
database access.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from precis.llms.llm_base import BaseLLMService
from precis.models import Paper, Section


class FakeLLMService(BaseLLMService):
    """Deterministic, offline stand-in for a real LLM provider.

    Records every call and returns canned structured outputs keyed by the
    requested schema, so pipeline behavior can be asserted without a network.
    """

    # Canned values returned for each known structured-output schema.
    _STRUCTURED: dict[str, dict[str, Any]] = {
        "SectionSummaryOutput": {
            "summary": "fake section summary",
            "key_points": ["point one", "point two"],
        },
        "ChunkSummaryOutput": {"updated_summary": "fake chunk summary"},
        "ExecutiveSummaryOutput": {"executive_summary": "fake executive summary"},
        "ContributionsOutput": {"contributions": ["contribution a", "contribution b"]},
    }

    def __init__(self, model_name: str = "fake-model-1") -> None:
        super().__init__(model_name)
        self.chat = None  # type: ignore[assignment]  # network is never touched
        self.calls: list[tuple[str, str]] = []

    @property
    def provider_name(self) -> str:
        return "Fake"

    def _init_chat(self) -> Any:  # pragma: no cover - never invoked
        raise NotImplementedError

    async def ask(self, prompt: str) -> str:
        self.calls.append(("ask", prompt))
        return f"FAKE_ANSWER::{prompt[:40]}"

    async def ask_structured(
        self,
        prompt: str,
        output_schema: type[BaseModel],
        system_prompt: str | None = None,
    ) -> Any:
        name = output_schema.__name__
        self.calls.append((name, prompt))
        if name not in self._STRUCTURED:
            raise KeyError(f"No fake response registered for schema {name}")
        return output_schema(**self._STRUCTURED[name])


@pytest.fixture
def fake_llm() -> FakeLLMService:
    """A deterministic offline LLM service."""
    return FakeLLMService()


@pytest.fixture
def sample_paper() -> Paper:
    """A small two-section paper for pipeline tests."""
    return Paper(
        title="A Study of Fakes",
        authors=["Ada Lovelace"],
        abstract="We study deterministic test doubles.",
        source_path="/papers/fakes.pdf",
        total_pages=3,
        sections=[
            Section(
                title="Introduction",
                content="Intro paragraph one.\n\nIntro paragraph two.",
                level=1,
                page_start=1,
                page_end=1,
                subsections=[
                    Section(
                        title="Background",
                        content="Some background content.",
                        level=2,
                        page_start=1,
                        page_end=2,
                    )
                ],
            ),
            Section(
                title="Method",
                content="We describe the method in detail.",
                level=1,
                page_start=2,
                page_end=3,
            ),
        ],
    )
