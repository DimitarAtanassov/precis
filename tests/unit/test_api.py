"""Tests for the FastAPI service.

The LLM provider seam is overridden with the offline ``fake_llm`` so the whole
HTTP stack (routing, DTO validation, DI, error mapping, request-id) is exercised
without any network.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from precis.api.dependencies import get_provider_factory
from precis.api.main import create_app
from precis.config.settings import Settings
from precis.container import build_container


@pytest.fixture
def client(fake_llm, tmp_path: Path) -> TestClient:
    settings = Settings(_env_file=None, output_dir=tmp_path)  # type: ignore[call-arg]
    app = create_app(build_container(settings))
    app.dependency_overrides[get_provider_factory] = lambda: (
        lambda _provider, _model: fake_llm
    )
    return TestClient(app)


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_list_prompts(client: TestClient) -> None:
    response = client.get("/v1/prompts")
    assert response.status_code == 200
    assert "webpage_summary" in response.json()["keys"]


def test_summarize_text(client: TestClient) -> None:
    response = client.post("/v1/summarize/text", json={"text": "hello world"})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"].startswith("FAKE_ANSWER::")
    assert body["provider"] == "claude"
    assert "X-Request-ID" in response.headers


def test_validation_error_on_empty_text(client: TestClient) -> None:
    response = client.post("/v1/summarize/text", json={"text": ""})
    assert response.status_code == 422


def test_config_error_renders_problem_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No provider override and no API keys -> ConfigError -> RFC7807 502/500.
    for key in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None, output_dir=tmp_path)  # type: ignore[call-arg]
    app = create_app(build_container(settings))
    client = TestClient(app)

    response = client.post("/v1/summarize/text", json={"text": "hi"})

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["title"] == "ConfigError"
