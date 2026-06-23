"""
Claude (Anthropic) LLM service.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from precis.llms.llm_base import BaseLLMService


class ClaudeLLMService(BaseLLMService):
    """Anthropic Claude LLM service. API key is injected via settings."""

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-5-20250929",
        *,
        api_key: SecretStr | None = None,
        timeout: int = 60,
    ) -> None:
        super().__init__(model_name, api_key=api_key, timeout=timeout)
        self.chat = self._init_chat()

    @property
    def provider_name(self) -> str:
        return "Claude"

    def _init_chat(self) -> BaseChatModel:
        return ChatAnthropic(
            api_key=self._require_api_key(),
            model_name=self.model_name,
            timeout=self._timeout,
            stop=None,
        )
