"""Characterization tests for the YAML prompt provider.

Pins parsing of all three supported YAML entry shapes plus variable
substitution and error behavior before the Phase 2 move to adapters/prompts.
"""

from pathlib import Path

import pytest

from precis.services.prompt_provider import YamlPromptProvider

YAML_CONTENT = """\
full:
  system_prompt:
    prompt: "You are {role}."
  user_prompt:
    prompt: "Summarize {topic}."
simple:
  prompt: "Just {thing}."
direct: "Plain {value} string."
"""


@pytest.fixture
def provider(tmp_path: Path) -> YamlPromptProvider:
    yaml_file = tmp_path / "prompts.yaml"
    yaml_file.write_text(YAML_CONTENT, encoding="utf-8")
    return YamlPromptProvider(yaml_path=yaml_file)


def test_full_entry_with_substitution(provider: YamlPromptProvider) -> None:
    prompt = provider.get("full", role="an expert", topic="fakes")
    assert prompt.system_prompt == "You are an expert."
    assert prompt.user_prompt == "Summarize fakes."


def test_simple_entry(provider: YamlPromptProvider) -> None:
    prompt = provider.get("simple", thing="this")
    assert prompt.system_prompt == ""
    assert prompt.user_prompt == "Just this."


def test_direct_string_entry(provider: YamlPromptProvider) -> None:
    prompt = provider.get("direct", value="text")
    assert prompt.user_prompt == "Plain text string."


def test_exists_and_list_keys(provider: YamlPromptProvider) -> None:
    assert provider.exists("full") is True
    assert provider.exists("missing") is False
    assert sorted(provider.list_keys()) == ["direct", "full", "simple"]


def test_missing_key_raises(provider: YamlPromptProvider) -> None:
    with pytest.raises(KeyError):
        provider.get("nope")


def test_no_variables_skips_substitution(provider: YamlPromptProvider) -> None:
    # Current behavior: with no kwargs, the template is returned verbatim.
    assert provider.get("simple").user_prompt == "Just {thing}."


def test_missing_variable_raises_value_error(provider: YamlPromptProvider) -> None:
    # When some variables are supplied but a referenced one is absent.
    with pytest.raises(ValueError, match="Missing variable"):
        provider.get("full", role="an expert")  # 'topic' not provided


def test_missing_file_raises(tmp_path: Path) -> None:
    provider = YamlPromptProvider(yaml_path=tmp_path / "absent.yaml")
    with pytest.raises(FileNotFoundError):
        provider.get("anything")
