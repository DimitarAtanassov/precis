"""Composition root.

The single place where concrete implementations are wired together from
configuration. Inbound adapters (CLI, API) build a ``Container`` once and ask it
for fully-constructed services, instead of newing up dependencies inline and
reaching for global singletons. This keeps wiring explicit, swappable, and
testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from precis.config import Settings, get_settings
from precis.core.output import OutputWriter
from precis.observability import configure_logging
from precis.services.llm_service import LLMService
from precis.services.prompt_service import PromptService


@dataclass(frozen=True)
class Container:
    """Application dependency container built from settings."""

    settings: Settings

    def prompt_service(self) -> PromptService:
        """Prompt source selected by settings (YAML or database)."""
        return PromptService.from_settings(self.settings)

    def llm_service(self) -> LLMService:
        """A configurable LLM orchestration service sharing the container's wiring."""
        return LLMService(prompts=self.prompt_service(), settings=self.settings)

    def output_writer(self) -> OutputWriter:
        """Filesystem output writer rooted at the configured output directory."""
        return OutputWriter(output_dir=self.settings.output_dir)


def build_container(settings: Settings | None = None) -> Container:
    """Build the application container and initialize cross-cutting concerns.

    Configures structured logging from settings so every consumer that builds a
    container gets consistent, level-aware logs.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    return Container(settings=settings)
