"""Tests for the Typer CLI, with the LLM provider faked (no network)."""

import pytest
from typer.testing import CliRunner

from precis.cli.app import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "summarize" in result.stdout
    assert "serve" in result.stdout


def test_prompts_command_lists_keys() -> None:
    result = runner.invoke(app, ["prompts"])
    assert result.exit_code == 0
    assert "webpage_summary" in result.stdout


def test_summarize_text_command(fake_llm, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("precis.cli.app.get_llm_service", lambda *_a, **_k: fake_llm)
    result = runner.invoke(app, ["summarize", "text", "hello world"])
    assert result.exit_code == 0
    assert "FAKE_ANSWER::" in result.stdout


def test_summarize_text_requires_input() -> None:
    result = runner.invoke(app, ["summarize", "text"])
    assert result.exit_code == 1
