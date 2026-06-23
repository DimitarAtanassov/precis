"""
Base LLM service with common functionality.

All provider-specific services inherit from this base class,
which handles retry logic, structured output, and fallback parsing.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base exception for LLM errors."""

    pass


class StructuredOutputError(LLMError):
    """Raised when structured output parsing fails."""

    pass


class BaseLLMService(ABC):
    """
    Base class for LLM service implementations.

    Provides common functionality for all LLM providers:
    - Message building
    - Retry logic with exponential backoff
    - Structured output with fallback parsing

    Subclasses only need to:
    1. Initialize `self.chat` with the appropriate LangChain chat model
    2. Set `self.model_name`
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds

    def __init__(self, model_name: str) -> None:
        """
        Initialize the LLM service.

        Args:
            model_name: The name of the model to use.
        """
        self.model_name = model_name
        self.system_prompt: str | None = None
        self.chat: BaseChatModel  # Must be set by subclass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'Claude', 'OpenAI')."""
        ...

    @abstractmethod
    def _init_chat(self) -> BaseChatModel:
        """
        Initialize and return the LangChain chat model.

        Subclasses implement this to set up their specific provider.
        """
        ...

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set the system prompt for subsequent calls."""
        self.system_prompt = system_prompt

    def ask(self, prompt: str) -> str:
        """
        Send a prompt and get a string response.

        Args:
            prompt: The user prompt.

        Returns:
            The model's response as a string.
        """
        messages = self._build_messages(prompt)
        response = self.chat.invoke(messages)
        content = response.content
        return str(content) if not isinstance(content, str) else content

    def ask_structured(
        self, prompt: str, output_schema: type[T], system_prompt: str | None = None
    ) -> T:
        """
        Send a prompt and get a structured response.

        Uses retry logic with exponential backoff. Falls back to
        raw text parsing if structured output fails.

        Args:
            prompt: The user prompt.
            output_schema: Pydantic model class for the response.
            system_prompt: Optional system prompt override.

        Returns:
            Instance of the output_schema model.

        Raises:
            StructuredOutputError: If parsing fails after all retries.
        """
        messages = self._build_messages(prompt, system_prompt)
        last_error: Exception | None = None

        for attempt in range(self.MAX_RETRIES):
            try:
                structured_llm = self.chat.with_structured_output(output_schema)
                result = structured_llm.invoke(messages)

                if result is None:
                    schema_name = output_schema.__name__
                    raise StructuredOutputError(
                        f"{self.provider_name} returned None for {schema_name}"
                    )
                return result  # type: ignore[return-value]

            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))

        return self._fallback_parse(messages, output_schema, last_error or Exception())

    def _build_messages(
        self, prompt: str, system_prompt: str | None = None
    ) -> list[BaseMessage]:
        """Build the message list for the chat model."""
        messages: list[BaseMessage] = []
        sys_prompt = system_prompt or self.system_prompt
        if sys_prompt:
            messages.append(SystemMessage(content=sys_prompt))
        messages.append(HumanMessage(content=prompt))
        return messages

    def _fallback_parse(
        self,
        messages: list[Any],
        output_schema: type[T],
        original_error: Exception,
    ) -> T:
        """
        Fallback parser when structured output fails.

        Gets raw text response and constructs a minimal valid schema instance.
        """
        try:
            response = self.chat.invoke(messages)
            content = response.content
            text = str(content) if content else ""

            return self._construct_fallback_response(output_schema, text)

        except Exception as fallback_error:
            raise StructuredOutputError(
                f"Structured output failed for {output_schema.__name__}. "
                f"Original: {original_error}. Fallback: {fallback_error}"
            ) from fallback_error

    def _construct_fallback_response(self, output_schema: type[T], text: str) -> T:
        """Construct a minimal valid response for known schema types."""
        schema_name = output_schema.__name__

        # Map schema names to their fallback constructors
        fallback_map: dict[str, dict[str, Any]] = {
            "SectionSummaryOutput": {
                "summary": text[:500] if text else "Unable to generate summary.",
                "key_points": [],
            },
            "ExecutiveSummaryOutput": {
                "executive_summary": text or "Unable to generate executive summary.",
            },
            "ChunkSummaryOutput": {
                "updated_summary": text or "Unable to generate chunk summary.",
            },
            "ContributionsOutput": {
                "contributions": self._extract_contributions(text),
            },
        }

        if schema_name in fallback_map:
            return output_schema(**fallback_map[schema_name])

        raise StructuredOutputError(f"No fallback handler for {schema_name}")

    def _extract_contributions(self, text: str) -> list[str]:
        """Extract contribution lines from raw text."""
        if not text:
            return ["Unable to extract contributions."]
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        contributions = [line.lstrip("0123456789.-) ") for line in lines[:7]]
        return contributions or ["Unable to extract contributions."]
