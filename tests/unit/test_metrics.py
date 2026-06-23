"""Tests for LLM usage/cost metrics."""

from precis.observability.metrics import estimate_cost, record_llm_usage


def test_estimate_cost_known_model() -> None:
    # claude-sonnet: $3 / 1M input, $15 / 1M output.
    cost = estimate_cost("claude-sonnet-4-5-20250929", 1_000_000, 1_000_000)
    assert cost == 18.0


def test_estimate_cost_unknown_model_is_zero() -> None:
    assert estimate_cost("mystery-model", 1000, 1000) == 0.0


def test_record_llm_usage_returns_record() -> None:
    record = record_llm_usage("openai", "gpt-4o", input_tokens=1000, output_tokens=500)
    assert record.provider == "openai"
    assert record.model == "gpt-4o"
    assert record.input_tokens == 1000
    assert record.output_tokens == 500
    # gpt-4o: (1000*2.5 + 500*10)/1e6
    assert record.estimated_cost_usd == round((1000 * 2.5 + 500 * 10.0) / 1_000_000, 6)
