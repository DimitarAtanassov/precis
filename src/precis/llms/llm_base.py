"""
Base LLM service with common functionality.

All provider-specific services inherit from this base class,
which handles retry logic, structured output, and fallback parsing.
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, SecretStr
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Re-exported for backward compatibility; the canonical definitions now live in
# precis.domain.errors so every layer shares one error hierarchy.
from precis.domain.errors import ConfigError, LLMError, StructuredOutputError
from precis.observability.metrics import record_llm_usage

T = TypeVar("T", bound=BaseModel)

__all__ = ["BaseLLMService", "LLMError", "StructuredOutputError"]


class BaseLLMService(ABC):
    """
    Base class for LLM service implementations.

    Provides common functionality for all LLM providers:
    - Message building
    - Retry logic with exponential backoff
    - Structured output with fallback parsing

    Subclasses only need to:
    1. Accept injected credentials via ``super().__init__`` and assign
       ``self.chat`` from ``self._init_chat()``.
    2. Implement ``provider_name`` and ``_init_chat``.

    Credentials are injected (never read from the environment here) so the
    composition root owns configuration and providers stay unit-testable.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds

    def __init__(
        self,
        model_name: str,
        *,
        api_key: SecretStr | None = None,
        timeout: int = 60,
    ) -> None:
        """
        Initialize the LLM service.

        Args:
            model_name: The name of the model to use.
            api_key: Provider API key, injected from settings.
            timeout: Per-request timeout in seconds.
        """
        self.model_name = model_name
        self._api_key = api_key
        self._timeout = timeout
        self.system_prompt: str | None = None
        self.chat: BaseChatModel  # Must be set by subclass from _init_chat()

    def _require_api_key(self) -> SecretStr:
        """Return the injected API key or raise a clear configuration error."""
        if self._api_key is None:
            raise ConfigError(
                f"{self.provider_name} API key is not configured. "
                "Set the appropriate *_API_KEY environment variable."
            )
        return self._api_key

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

    async def ask(self, prompt: str) -> str:
        """
        Send a prompt and get a string response.

        Args:
            prompt: The user prompt.

        Returns:
            The model's response as a string.
        """
        messages = self._build_messages(prompt)
        response = await self.chat.ainvoke(messages)
        self._record_usage(response)
        content = response.content
        return content if isinstance(content, str) else str(content)

    def _record_usage(self, response: BaseMessage) -> None:
        """Emit token/cost metrics from a chat response when available."""
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return
        record_llm_usage(
            self.provider_name,
            self.model_name,
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
        )

    async def ask_structured(
        self, prompt: str, output_schema: type[T], system_prompt: str | None = None
    ) -> T:
        """
        Send a prompt and get a structured response.

        Retries with exponential backoff (via tenacity), then falls back to raw
        text parsing if structured output keeps failing.

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
        last_error: Exception = StructuredOutputError(
            f"{self.provider_name} produced no output for {output_schema.__name__}"
        )

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.MAX_RETRIES),
                wait=wait_exponential(multiplier=self.RETRY_DELAY),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            ):
                with attempt:
                    structured_llm = self.chat.with_structured_output(output_schema)
                    result = await structured_llm.ainvoke(messages)
                    if result is None:
                        raise StructuredOutputError(
                            f"{self.provider_name} returned None for "
                            f"{output_schema.__name__}"
                        )
                    return cast(T, result)
        except Exception as exc:
            last_error = exc

        return await self._fallback_parse(messages, output_schema, last_error)

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

    async def _fallback_parse(
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
            response = await self.chat.ainvoke(messages)
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
