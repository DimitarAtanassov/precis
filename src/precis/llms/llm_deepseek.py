"""
DeepSeek LLM service.

Uses OpenAI-compatible API with custom base URL.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from precis.llms.llm_base import BaseLLMService


class DeepSeekLLMService(BaseLLMService):
    """DeepSeek LLM service (OpenAI-compatible). API key is injected via settings."""

    API_BASE = "https://api.deepseek.com/v1"

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        *,
        api_key: SecretStr | None = None,
        timeout: int = 60,
    ) -> None:
        super().__init__(model_name, api_key=api_key, timeout=timeout)
        self.chat = self._init_chat()

    @property
    def provider_name(self) -> str:
        return "DeepSeek"

    def _init_chat(self) -> BaseChatModel:
        return ChatOpenAI(
            api_key=self._require_api_key(),
            model=self.model_name,
            base_url=self.API_BASE,
            timeout=self._timeout,
        )
