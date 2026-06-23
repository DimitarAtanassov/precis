"""
Prompt Provider Interface and Implementations.

Follows the Strategy Pattern to allow swapping between different
prompt sources (YAML files, databases, etc.) without changing
the consuming code.

Design Patterns:
    - Strategy: Different providers implement the same interface
    - Adapter: DatabasePromptProvider adapts floating_prompts to our interface
    - Dependency Inversion: LLMService depends on abstraction, not concrete classes
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from precis.models import Prompt


class PromptProvider(ABC):
    """
    Abstract base class for prompt providers.

    All prompt sources must implement this interface, allowing the
    PromptService to work with any provider interchangeably.

    Methods:
        get: Retrieve a prompt by key with variable substitution.
        exists: Check if a prompt exists.
        list_keys: List all available prompt keys.
    """

    @abstractmethod
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
        """
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a prompt exists."""
        ...

    @abstractmethod
    def list_keys(self) -> list[str]:
        """List all available prompt keys."""
        ...


class YamlPromptProvider(PromptProvider):
    """
    Loads prompts from a YAML file.

    This is the default provider for local development and
    when a database is not available.

    YAML Structure:
        prompt_name:
            system_prompt:
                prompt: "System instructions..."
            user_prompt:
                prompt: "User template with {variables}..."

        # Or simple format:
        simple_prompt:
            prompt: "Simple template..."
    """

    def __init__(self, yaml_path: Path | None = None) -> None:
        """
        Initialize the YAML provider.

        Args:
            yaml_path: Path to prompts.yaml. Defaults to package prompts.yaml.
        """
        self._yaml_path = yaml_path or self._default_path()
        self._prompts: dict[str, Any] = {}
        self._loaded = False

    def _default_path(self) -> Path:
        """Get the default prompts.yaml path."""
        return Path(__file__).parent.parent / "prompts.yaml"

    def _load(self) -> None:
        """Load prompts from YAML file (lazy loading)."""
        if self._loaded:
            return

        if not self._yaml_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {self._yaml_path}")

        with open(self._yaml_path) as f:
            self._prompts = yaml.safe_load(f) or {}

        self._loaded = True

    def get(self, key: str, **kwargs: object) -> Prompt:
        """Get a prompt from the YAML file."""
        self._load()

        if key not in self._prompts:
            raise KeyError(f"Prompt '{key}' not found in {self._yaml_path}")

        entry = self._prompts[key]
        return self._parse_entry(key, entry, kwargs)

    def _parse_entry(
        self, key: str, entry: dict[str, Any] | str, variables: dict[str, object]
    ) -> Prompt:
        """Parse a YAML entry into a Prompt object."""
        system: str = ""
        user: str = ""

        # Case 1: Full format with system_prompt and user_prompt
        has_both = (
            isinstance(entry, dict)
            and "system_prompt" in entry
            and "user_prompt" in entry
        )
        if has_both and isinstance(entry, dict):
            system_entry = entry["system_prompt"]
            user_entry = entry["user_prompt"]
            if isinstance(system_entry, dict):
                system = system_entry.get("prompt", "")
            if isinstance(user_entry, dict):
                user = user_entry.get("prompt", "")
        # Case 2: Simple format with just "prompt"
        elif isinstance(entry, dict) and "prompt" in entry:
            user = str(entry["prompt"])
        # Case 3: Direct string
        elif isinstance(entry, str):
            user = entry
        else:
            raise ValueError(f"Prompt '{key}' has unrecognized structure")

        # Apply variable substitution
        if variables:
            try:
                user = user.format(**variables)
                if system:
                    system = system.format(**variables)
            except KeyError as e:
                raise ValueError(f"Missing variable for prompt '{key}': {e}") from None

        return Prompt(system_prompt=system, user_prompt=user)

    def exists(self, key: str) -> bool:
        """Check if a prompt exists in the YAML file."""
        self._load()
        return key in self._prompts

    def list_keys(self) -> list[str]:
        """List all prompt keys in the YAML file."""
        self._load()
        return list(self._prompts.keys())


class DatabasePromptProvider(PromptProvider):
    """
    Loads prompts from the floating_prompts PostgreSQL database.

    This provider adapts the floating_prompts library to the
    PromptProvider interface used by precis.

    Requirements:
        - floating_prompts package must be installed
        - Database must be configured via environment variables

    Usage:
        provider = DatabasePromptProvider()
        prompt = provider.get("paper_summary", content="...")
    """

    def __init__(self, version: int | None = None) -> None:
        """
        Initialize the database provider.

        Args:
            version: Specific prompt version to use. None = latest.
        """
        self._version = version

    @staticmethod
    def _load_backend() -> tuple[Any, Any]:
        """Lazily import floating_prompts.

        The database prompt store is an optional, externally-evolving
        dependency. Importing it eagerly couples the whole ``precis`` package
        to its public API, so a drift in that library would break even the
        YAML-only and CLI code paths at import time. Deferring the import here
        keeps ``import precis`` working and surfaces any incompatibility only
        when the database provider is actually used.
        """
        from floating_prompts import (  # noqa: PLC0415
            PromptRepository,
            get_session,
        )

        return PromptRepository, get_session

    def get(self, key: str, **kwargs: object) -> Prompt:
        """
        Get a prompt from the database.

        Args:
            key: The prompt name in the database.
            **kwargs: Variables to substitute into the prompt.

        Returns:
            A Prompt object with rendered templates.

        Raises:
            KeyError: If prompt not found in database.
            ValueError: If required variables are missing.
        """
        prompt_repository, get_session = self._load_backend()
        with get_session() as session:
            repo = prompt_repository(session)
            db_prompt = repo.get_by_name(key, version=self._version)

            if db_prompt is None:
                raise KeyError(f"Prompt '{key}' not found in database")

            # Render with variables
            try:
                str_kwargs = {k: str(v) for k, v in kwargs.items()}
                system, user = db_prompt.render(**str_kwargs)
            except KeyError as e:
                raise ValueError(f"Missing variable for prompt '{key}': {e}") from None

            return Prompt(
                system_prompt=system or "",
                user_prompt=user,
            )

    def exists(self, key: str) -> bool:
        """Check if a prompt exists in the database."""
        prompt_repository, get_session = self._load_backend()
        with get_session() as session:
            repo = prompt_repository(session)
            result: bool = repo.exists(key, version=self._version)
            return result

    def list_keys(self) -> list[str]:
        """List all prompt names in the database."""
        prompt_repository, get_session = self._load_backend()
        with get_session() as session:
            repo = prompt_repository(session)
            result: list[str] = repo.list_names()
            return result


class CachedPromptProvider(PromptProvider):
    """
    Decorator that adds caching to any PromptProvider.

    Useful for reducing database calls in high-throughput scenarios.

    Usage:
        provider = CachedPromptProvider(DatabasePromptProvider())
    """

    def __init__(self, provider: PromptProvider, ttl_seconds: int = 300) -> None:
        """
        Initialize the cached provider.

        Args:
            provider: The underlying provider to cache.
            ttl_seconds: Cache time-to-live in seconds (default: 5 minutes).
        """
        self._provider = provider
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[Prompt, float]] = {}

    def _cache_key(self, key: str, kwargs: dict[str, object]) -> str:
        """Generate a cache key from prompt key and variables."""
        sorted_kwargs = sorted(kwargs.items())
        return f"{key}:{sorted_kwargs}"

    def _is_valid(self, cache_key: str) -> bool:
        """Check if a cache entry is still valid."""
        if cache_key not in self._cache:
            return False
        _, timestamp = self._cache[cache_key]
        return (time.time() - timestamp) < self._ttl

    def get(self, key: str, **kwargs: object) -> Prompt:
        """Get a prompt, using cache if available."""
        cache_key = self._cache_key(key, kwargs)

        if self._is_valid(cache_key):
            return self._cache[cache_key][0]

        prompt = self._provider.get(key, **kwargs)
        self._cache[cache_key] = (prompt, time.time())
        return prompt

    def exists(self, key: str) -> bool:
        """Check if prompt exists (delegated to underlying provider)."""
        return self._provider.exists(key)

    def list_keys(self) -> list[str]:
        """List prompt keys (delegated to underlying provider)."""
        return self._provider.list_keys()

    def clear_cache(self) -> None:
        """Clear all cached prompts."""
        self._cache.clear()
