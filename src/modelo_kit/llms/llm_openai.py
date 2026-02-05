"""
OpenAI LLM service.
"""

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from modelo_kit.llms.llm_base import BaseLLMService


class OpenAILLMService(BaseLLMService):
    """
    OpenAI LLM service (GPT-4, etc.).

    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(self, model_name: str = "gpt-4o") -> None:
        super().__init__(model_name)
        self.chat = self._init_chat()

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    def _init_chat(self) -> BaseChatModel:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")

        return ChatOpenAI(api_key=SecretStr(api_key), model=self.model_name)
