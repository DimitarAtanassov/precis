"""The Step abstraction: one unit of work in a pipeline.

A Step transforms a typed context and returns it (mutated or replaced). Steps
are intentionally tiny and single-purpose so pipelines stay composable and each
step is independently testable. This is the in-house building block the agent
layer is assembled from — no third-party orchestration framework.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

# Invariant context type shared across the steps of a single pipeline.
TContext = TypeVar("TContext")


@runtime_checkable
class Step(Protocol[TContext]):
    """An async unit of work over a pipeline context."""

    name: str

    async def run(self, ctx: TContext) -> TContext:
        """Execute the step and return the (possibly updated) context."""
        ...
