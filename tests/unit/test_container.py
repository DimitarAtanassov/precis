"""Tests for the composition root."""

from pathlib import Path

from precis.config.settings import Settings
from precis.container import build_container
from precis.core.output import OutputWriter
from precis.services.llm_service import LLMService
from precis.services.prompt_service import PromptService


def _settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, output_dir=tmp_path)  # type: ignore[call-arg]


def test_build_container_uses_provided_settings(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    container = build_container(settings)
    assert container.settings is settings


def test_container_builds_services(tmp_path: Path) -> None:
    container = build_container(_settings(tmp_path))

    assert isinstance(container.prompt_service(), PromptService)
    assert isinstance(container.llm_service(), LLMService)

    writer = container.output_writer()
    assert isinstance(writer, OutputWriter)
    assert writer.output_dir == tmp_path


def test_prompt_service_defaults_to_yaml(tmp_path: Path) -> None:
    # With the default (yaml) prompt source, prompts resolve offline.
    container = build_container(_settings(tmp_path))
    prompt = container.prompt_service().get("webpage_summary", content="hello")
    assert prompt.user_prompt
