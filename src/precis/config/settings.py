"""Typed application configuration.

A single ``Settings`` object is the one source of truth for every tunable in
precis. Validated once at startup, it replaces scattered ``os.getenv`` calls and
``load_dotenv`` side effects, so misconfiguration fails fast and loudly instead
of surfacing deep inside a request.

Provider API keys keep their conventional environment names (``ANTHROPIC_API_KEY``
etc.) because operators and the provider SDKs already expect them. Everything
precis-specific is namespaced under ``PRECIS_`` to avoid collisions.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PromptSource = Literal["yaml", "database"]

# Default provider/model used by inbound adapters (CLI/API) when unspecified.
DEFAULT_PROVIDER = "claude"
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

# Maps the public provider identifier to its Settings attribute holding the key.
_PROVIDER_KEY_ATTR: dict[str, str] = {
    "claude": "anthropic_api_key",
    "openai": "openai_api_key",
    "gemini": "google_api_key",
    "deepseek": "deepseek_api_key",
}


class Settings(BaseSettings):
    """Validated, environment-driven application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- Provider credentials (conventional env names) ---
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    google_api_key: SecretStr | None = Field(default=None, alias="GOOGLE_API_KEY")
    deepseek_api_key: SecretStr | None = Field(default=None, alias="DEEPSEEK_API_KEY")

    # --- LLM behavior ---
    llm_request_timeout: int = Field(default=60, alias="PRECIS_LLM_TIMEOUT")
    llm_max_retries: int = Field(default=3, alias="PRECIS_LLM_MAX_RETRIES")

    # --- Prompt source ---
    prompt_source: PromptSource = Field(default="yaml", alias="PRECIS_PROMPT_SOURCE")
    prompt_version: int | None = Field(default=None, alias="PRECIS_PROMPT_VERSION")

    # --- Output ---
    output_dir: Path = Field(default=Path("outputs"), alias="PRECIS_OUTPUT_DIR")

    # --- Observability ---
    service_name: str = Field(default="precis", alias="PRECIS_SERVICE_NAME")
    log_level: str = Field(default="INFO", alias="PRECIS_LOG_LEVEL")
    log_json: bool = Field(default=False, alias="PRECIS_LOG_JSON")
    otel_enabled: bool = Field(default=False, alias="PRECIS_OTEL_ENABLED")
    otel_endpoint: str | None = Field(default=None, alias="PRECIS_OTEL_ENDPOINT")

    def api_key_for(self, provider: str) -> SecretStr | None:
        """Return the configured API key for a provider id, or None if unset."""
        attr = _PROVIDER_KEY_ATTR.get(provider)
        if attr is None:
            return None
        value: SecretStr | None = getattr(self, attr)
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so configuration is parsed once. Tests can reset it via
    ``get_settings.cache_clear()``.
    """
    return Settings()
