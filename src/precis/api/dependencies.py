"""FastAPI dependency providers wiring routes to the composition root.

Routes depend on these small providers (not on concrete adapters), so tests can
override any seam via ``app.dependency_overrides`` — e.g. swapping in a fake LLM
provider with zero network access.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request

from precis.container import Container
from precis.llm_factory import get_llm_service
from precis.ports import LLMProvider, PromptRepository

# Builds a provider for a given (provider_id, model) pair.
ProviderFactory = Callable[[str, str], LLMProvider]


def get_container(request: Request) -> Container:
    """Return the application container stored on app state."""
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def get_prompts(container: ContainerDep) -> PromptRepository:
    """Return the configured prompt repository."""
    return container.prompt_service()


def get_provider_factory(container: ContainerDep) -> ProviderFactory:
    """Return a factory that builds LLM providers from settings-resolved keys."""
    settings = container.settings

    def factory(provider: str, model: str) -> LLMProvider:
        return get_llm_service(provider, model, settings)

    return factory


PromptsDep = Annotated[PromptRepository, Depends(get_prompts)]
ProviderFactoryDep = Annotated[ProviderFactory, Depends(get_provider_factory)]
