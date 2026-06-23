"""
Claude (Anthropic) LLM service.
"""

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from precis.llms.llm_base import BaseLLMService


class ClaudeLLMService(BaseLLMService):
    """
    Anthropic Claude LLM service.

    Requires ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self, model_name: str = "claude-sonnet-4-5-20250929") -> None:
        super().__init__(model_name)
        self.chat = self._init_chat()

    @property
    def provider_name(self) -> str:
        return "Claude"

    def _init_chat(self) -> BaseChatModel:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment.")

        return ChatAnthropic(
            api_key=SecretStr(api_key),
            model_name=self.model_name,
            timeout=60,
            stop=None,
        )
