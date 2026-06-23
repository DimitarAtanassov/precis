"""Tests for the in-house orchestration framework (Pipeline, Step, ToolRegistry)."""

from dataclasses import dataclass, field

import pytest

from precis.domain.errors import ConfigError, PipelineError
from precis.orchestration import Pipeline, Tool, ToolRegistry


@dataclass
class _Ctx:
    trail: list[str] = field(default_factory=list)


class _AppendStep:
    def __init__(self, label: str) -> None:
        self.name = f"append_{label}"
        self._label = label

    async def run(self, ctx: _Ctx) -> _Ctx:
        ctx.trail.append(self._label)
        return ctx


class _BoomStep:
    name = "boom"

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def run(self, ctx: _Ctx) -> _Ctx:
        raise self._exc


async def test_pipeline_runs_steps_in_order() -> None:
    pipeline = Pipeline([_AppendStep("a"), _AppendStep("b"), _AppendStep("c")])
    result = await pipeline.run(_Ctx())
    assert result.trail == ["a", "b", "c"]
    assert pipeline.step_names == ["append_a", "append_b", "append_c"]


async def test_pipeline_wraps_unexpected_errors() -> None:
    pipeline = Pipeline([_BoomStep(RuntimeError("kaboom"))])
    with pytest.raises(PipelineError, match="boom"):
        await pipeline.run(_Ctx())


async def test_pipeline_propagates_domain_errors() -> None:
    # Domain errors are meaningful and must not be wrapped.
    pipeline = Pipeline([_BoomStep(ConfigError("missing key"))])
    with pytest.raises(ConfigError):
        await pipeline.run(_Ctx())


async def test_tool_registry_register_and_call() -> None:
    registry = ToolRegistry()

    async def echo(value: str) -> str:
        return value.upper()

    registry.register(Tool(name="echo", description="upper-cases input", func=echo))

    assert "echo" in registry
    assert registry.names() == ["echo"]
    assert await registry.call("echo", value="hi") == "HI"


def test_tool_registry_rejects_duplicates() -> None:
    registry = ToolRegistry()

    async def noop() -> None:
        return None

    registry.register(Tool(name="t", description="", func=noop))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Tool(name="t", description="", func=noop))


def test_tool_registry_missing_tool_raises() -> None:
    with pytest.raises(KeyError):
        ToolRegistry().get("absent")
