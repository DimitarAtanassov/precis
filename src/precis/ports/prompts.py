"""Port for prompt repositories."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from precis.domain import Prompt


@runtime_checkable
class PromptRepository(Protocol):
    """Contract for any prompt source (YAML file, database, cache, ...)."""

    def get(self, key: str, **kwargs: object) -> Prompt:
        """Resolve a prompt by key, substituting ``kwargs`` into the template."""
        ...

    def exists(self, key: str) -> bool:
        """Return whether a prompt with ``key`` exists."""
        ...

    def list_keys(self) -> list[str]:
        """List all available prompt keys."""
        ...
