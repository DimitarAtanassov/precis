"""
Google Gemini LLM service.
"""

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from modelo_kit.llms.llm_base import BaseLLMService


class GeminiLLMService(BaseLLMService):
    """
    Google Gemini LLM service.

    Requires GOOGLE_API_KEY environment variable.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        super().__init__(model_name)
        self.chat = self._init_chat()

    @property
    def provider_name(self) -> str:
        return "Gemini"

    def _init_chat(self) -> BaseChatModel:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment.")

        return ChatGoogleGenerativeAI(api_key=SecretStr(api_key), model=self.model_name)
