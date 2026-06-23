"""Pipeline: a sequential runner over an ordered list of steps.

The runner owns cross-cutting concerns once, so steps stay pure business logic:
per-step structured logging + timing, and uniform error wrapping into
``PipelineError`` (while letting domain errors propagate untouched). Sequencing
is explicit and deterministic; concurrency, when wanted, lives inside a single
step (e.g. fanning out over sections) rather than across the pipeline.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from precis.domain.errors import PipelineError, PrecisError
from precis.observability import get_logger
from precis.orchestration.step import Step

_logger = get_logger(__name__)


class Pipeline[TContext]:
    """Runs an ordered sequence of steps over a shared context."""

    def __init__(
        self, steps: Sequence[Step[TContext]], *, name: str = "pipeline"
    ) -> None:
        self._steps: list[Step[TContext]] = list(steps)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def step_names(self) -> list[str]:
        return [step.name for step in self._steps]

    async def run(self, ctx: TContext) -> TContext:
        """Execute every step in order, returning the final context."""
        _logger.info("pipeline_start", pipeline=self._name, steps=self.step_names)

        for step in self._steps:
            started = time.perf_counter()
            _logger.debug("step_start", pipeline=self._name, step=step.name)
            try:
                ctx = await step.run(ctx)
            except PrecisError:
                # Domain errors are already meaningful; let them propagate.
                raise
            except Exception as exc:
                raise PipelineError(
                    f"Step '{step.name}' failed in pipeline '{self._name}': {exc}"
                ) from exc

            elapsed = time.perf_counter() - started
            _logger.info(
                "step_done",
                pipeline=self._name,
                step=step.name,
                seconds=round(elapsed, 3),
            )

        _logger.info("pipeline_done", pipeline=self._name)
        return ctx
