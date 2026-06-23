"""
OpenAI LLM service.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from precis.llms.llm_base import BaseLLMService


class OpenAILLMService(BaseLLMService):
    """OpenAI LLM service (GPT-4o, etc.). API key is injected via settings."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        *,
        api_key: SecretStr | None = None,
        timeout: int = 60,
    ) -> None:
        super().__init__(model_name, api_key=api_key, timeout=timeout)
        self.chat = self._init_chat()

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    def _init_chat(self) -> BaseChatModel:
        return ChatOpenAI(
            api_key=self._require_api_key(),
            model=self.model_name,
            timeout=self._timeout,
        )
