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

from precis.config import Settings, get_settings
from precis.models import Prompt
from precis.services.prompt_provider import (
    DatabasePromptProvider,
    PromptProvider,
    YamlPromptProvider,
)


class PromptService:
    """
    Unified service for prompt management.

    Provides a simple interface to get prompts regardless of the
    underlying storage mechanism (YAML file or database).

    Default: Uses the bundled YAML provider (offline-capable). The
    floating_prompts database provider is opt-in via ``from_database()``.

    Usage:
        # Default: YAML provider (bundled prompts.yaml)
        service = PromptService()

        # Opt into the database provider
        service = PromptService.from_database()

        # Explicit YAML file
        service = PromptService.from_yaml()

        # Get prompts
        prompt = service.get("paper_summary", content="...")
    """

    def __init__(self, provider: PromptProvider | None = None) -> None:
        """
        Initialize the service with a provider.

        Args:
            provider: The prompt provider to use. Defaults to YamlPromptProvider,
                the bundled offline-capable source. Use ``from_database()`` or
                ``from_settings()`` to opt into the floating_prompts database
                provider.
        """
        self._provider = provider or YamlPromptProvider()

    @classmethod
    def from_settings(cls, settings: "Settings | None" = None) -> "PromptService":
        """Build a service whose provider is chosen by application settings.

        ``prompt_source = "database"`` selects the floating_prompts provider;
        anything else uses the bundled YAML provider.
        """
        settings = settings or get_settings()
        if settings.prompt_source == "database":
            return cls.from_database(version=settings.prompt_version)
        return cls.from_yaml()

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
