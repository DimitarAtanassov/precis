"""
DeepSeek LLM service.

Uses OpenAI-compatible API with custom base URL.
"""

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from modelo_kit.llms.llm_base import BaseLLMService


class DeepSeekLLMService(BaseLLMService):
    """
    DeepSeek LLM service using OpenAI-compatible API.

    Requires DEEPSEEK_API_KEY environment variable.
    """

    API_BASE = "https://api.deepseek.com/v1"

    def __init__(self, model_name: str = "deepseek-chat") -> None:
        super().__init__(model_name)
        self.chat = self._init_chat()

    @property
    def provider_name(self) -> str:
        return "DeepSeek"

    def _init_chat(self) -> BaseChatModel:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment.")

        return ChatOpenAI(
            api_key=SecretStr(api_key),
            model=self.model_name,
            base_url=self.API_BASE,
        )
