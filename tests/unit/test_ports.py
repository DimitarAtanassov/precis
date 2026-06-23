"""Tests for the ports (Protocol) contract layer."""

from precis.ports import LLMProvider, TokenCounter
from precis.services.summarizer_service import SimpleTokenCounter


def test_fake_llm_satisfies_llm_provider_port(fake_llm) -> None:
    # runtime_checkable Protocol: the test double is structurally a provider.
    assert isinstance(fake_llm, LLMProvider)


def test_token_counter_implementations_satisfy_port() -> None:
    assert isinstance(SimpleTokenCounter(), TokenCounter)


async def test_provider_ask_is_async(fake_llm) -> None:
    result = await fake_llm.ask("hello")
    assert result.startswith("FAKE_ANSWER::")
