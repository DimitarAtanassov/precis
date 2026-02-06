"""
Prompt Service - unified interface for prompt management.

Acts as a facade over different prompt providers (YAML, Database, etc.)
following the Strategy Pattern.

Design Patterns:
    - Facade: Simple interface hiding provider complexity
    - Strategy: Swappable providers at runtime
    - Singleton: Single instance per provider type (optional)
"""

from pathlib import Path

from modelo_kit.models import Prompt
from modelo_kit.services.prompt_provider import (
    DatabasePromptProvider,
    PromptProvider,
    YamlPromptProvider,
)


class PromptService:
    """
    Unified service for prompt management.

    Provides a simple interface to get prompts regardless of the
    underlying storage mechanism (YAML file or database).

    Default: Uses database provider (floating_prompts).

    Usage:
        # Default: Database provider
        service = PromptService()

        # Explicit database provider
        service = PromptService.from_database()

        # YAML fallback (for offline/testing)
        service = PromptService.from_yaml()

        # Get prompts
        prompt = service.get("paper_summary", content="...")
    """

    _instance: "PromptService | None" = None

    def __init__(self, provider: PromptProvider | None = None) -> None:
        """
        Initialize the service with a provider.

        Args:
            provider: The prompt provider to use. Defaults to DatabasePromptProvider.
        """
        self._provider = provider or DatabasePromptProvider()

    def __new__(cls, provider: PromptProvider | None = None) -> "PromptService":
        """
        Singleton pattern - returns existing instance if no provider specified.

        If a provider is explicitly passed, creates a new instance.
        This allows both singleton behavior and custom instances.
        """
        if provider is not None:
            # Custom provider = new instance
            instance = super().__new__(cls)
            return instance

        # No provider = singleton with default database
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def from_database(cls, version: int | None = None) -> "PromptService":
        """
        Create a service using the database provider.

        Args:
            version: Specific prompt version to use. None = latest.

        Returns:
            A new PromptService instance using DatabasePromptProvider.
        """
        return cls(provider=DatabasePromptProvider(version=version))

    @classmethod
    def from_yaml(cls, yaml_path: str | None = None) -> "PromptService":
        """
        Create a service using a YAML file.

        Useful for offline development or testing without database.

        Args:
            yaml_path: Path to the YAML file. None = default prompts.yaml.

        Returns:
            A new PromptService instance using YamlPromptProvider.
        """
        path = Path(yaml_path) if yaml_path else None
        return cls(provider=YamlPromptProvider(yaml_path=path))

    def get(self, key: str, **kwargs: object) -> Prompt:
        """
        Get a prompt by key with variable substitution.

        Args:
            key: The prompt identifier.
            **kwargs: Variables to substitute into the prompt template.

        Returns:
            A Prompt object with system_prompt and user_prompt.

        Raises:
            KeyError: If prompt not found.
            ValueError: If required variables are missing.

        Example:
            prompt = service.get("paper_summary", title="AI Paper", content="...")
            print(prompt.system_prompt)
            print(prompt.user_prompt)
        """
        return self._provider.get(key, **kwargs)

    def exists(self, key: str) -> bool:
        """Check if a prompt exists."""
        return self._provider.exists(key)

    def list_keys(self) -> list[str]:
        """List all available prompt keys."""
        return self._provider.list_keys()

    @property
    def provider(self) -> PromptProvider:
        """Get the underlying provider (for advanced usage)."""
        return self._provider

