"""Per-call LLM usage and cost metrics.

Every model call records input/output tokens and an estimated USD cost as a
structured ``llm_usage`` event. Emitting via structlog keeps it dependency-light
while remaining trivially aggregatable downstream (log pipeline, dashboards).
Prices are coarse per-1M-token estimates; extend ``_PRICES_PER_MTOK`` as needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from precis.observability.logging import get_logger

_logger = get_logger(__name__)

# (input_price, output_price) in USD per 1,000,000 tokens. Approximate.
_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-opus-4-5-20251101": (15.0, 75.0),
    "gpt-4o": (2.5, 10.0),
    "deepseek-chat": (0.27, 1.10),
}

_MILLION = 1_000_000


@dataclass(frozen=True)
class UsageRecord:
    """Token usage and estimated cost for a single LLM call."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of a call from token counts; 0.0 if model unknown."""
    input_price, output_price = _PRICES_PER_MTOK.get(model, (0.0, 0.0))
    cost = (input_tokens * input_price + output_tokens * output_price) / _MILLION
    return round(cost, 6)


def record_llm_usage(
    provider: str, model: str, input_tokens: int, output_tokens: int
) -> UsageRecord:
    """Record and emit usage/cost for one LLM call."""
    record = UsageRecord(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimate_cost(model, input_tokens, output_tokens),
    )
    _logger.info(
        "llm_usage",
        provider=record.provider,
        model=record.model,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        estimated_cost_usd=record.estimated_cost_usd,
    )
    return record
