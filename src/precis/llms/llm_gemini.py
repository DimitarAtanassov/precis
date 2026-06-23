"""
Google Gemini LLM service.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from precis.llms.llm_base import BaseLLMService


class GeminiLLMService(BaseLLMService):
    """Google Gemini LLM service. API key is injected via settings."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        *,
        api_key: SecretStr | None = None,
        timeout: int = 60,
    ) -> None:
        super().__init__(model_name, api_key=api_key, timeout=timeout)
        self.chat = self._init_chat()

    @property
    def provider_name(self) -> str:
        return "Gemini"

    def _init_chat(self) -> BaseChatModel:
        return ChatGoogleGenerativeAI(
            api_key=self._require_api_key(),
            model=self.model_name,
        )
