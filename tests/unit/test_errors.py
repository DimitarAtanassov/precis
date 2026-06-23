"""Tests pinning the domain error hierarchy contract."""

from precis.domain.errors import (
    ConfigError,
    LLMError,
    ParseError,
    PrecisError,
    PromptNotFoundError,
    StructuredOutputError,
)
from precis.llms import llm_base


def test_all_errors_derive_from_precis_error() -> None:
    for err in (ConfigError, LLMError, ParseError, StructuredOutputError):
        assert issubclass(err, PrecisError)


def test_structured_output_is_an_llm_error() -> None:
    assert issubclass(StructuredOutputError, LLMError)


def test_prompt_not_found_is_keyerror_for_backward_compat() -> None:
    assert issubclass(PromptNotFoundError, KeyError)
    assert issubclass(PromptNotFoundError, PrecisError)


def test_llm_base_reexports_canonical_errors() -> None:
    # Backward-compatible imports must resolve to the domain definitions.
    assert llm_base.LLMError is LLMError
    assert llm_base.StructuredOutputError is StructuredOutputError
