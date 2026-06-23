"""Port for large language model providers."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMProvider(Protocol):
    """Async contract every LLM adapter must satisfy.

    The core summarization/orchestration code depends on this Protocol, not on
    any specific SDK. Methods are async so a single provider can be driven
    concurrently by the pipeline and the API without blocking the event loop.
    """

    model_name: str

    @property
    def provider_name(self) -> str:
        """Human-readable provider name (e.g. ``"Claude"``)."""
        ...

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set a default system prompt for subsequent calls."""
        ...

    async def ask(self, prompt: str) -> str:
        """Return a free-text completion for ``prompt``."""
        ...

    async def ask_structured(
        self,
        prompt: str,
        output_schema: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Return a validated instance of ``output_schema`` for ``prompt``."""
        ...
