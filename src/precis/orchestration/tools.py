"""ToolRegistry: the seam for tool-calling agents.

A tool is a named, described, async callable. The registry is the explicit
extension point where future LLM-driven agents discover and invoke capabilities
(web fetch, vault search, calculators, ...). Built in-house and deliberately
small: registration, lookup, and invocation — no framework lock-in.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

ToolFn = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class Tool:
    """A named async capability an agent can invoke."""

    name: str
    description: str
    func: ToolFn


class ToolRegistry:
    """Registry of tools available to steps and agents."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool, rejecting duplicate names."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Look up a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        """List registered tool names."""
        return list(self._tools)

    async def call(self, name: str, /, **kwargs: object) -> Any:  # noqa: ANN401
        """Invoke a registered tool by name (return type is tool-defined)."""
        return await self.get(name).func(**kwargs)
