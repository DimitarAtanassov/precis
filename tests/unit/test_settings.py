"""Tests for typed application settings."""

import pytest

from precis.config.settings import Settings, get_settings


def _settings(**env: str) -> Settings:
    """Build Settings from explicit field values, ignoring any .env file."""
    return Settings(_env_file=None, **env)  # type: ignore[call-arg]


def test_defaults() -> None:
    s = _settings()
    assert s.prompt_source == "yaml"
    assert s.llm_request_timeout == 60
    assert s.llm_max_retries == 3
    assert s.log_level == "INFO"
    assert s.log_json is False


def test_api_key_for_maps_providers() -> None:
    s = _settings(
        anthropic_api_key="a",
        openai_api_key="o",
        google_api_key="g",
        deepseek_api_key="d",
    )
    assert s.api_key_for("claude").get_secret_value() == "a"
    assert s.api_key_for("openai").get_secret_value() == "o"
    assert s.api_key_for("gemini").get_secret_value() == "g"
    assert s.api_key_for("deepseek").get_secret_value() == "d"
    assert s.api_key_for("unknown") is None


def test_reads_conventional_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    s = _settings()
    assert s.api_key_for("claude").get_secret_value() == "from-env"


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()
