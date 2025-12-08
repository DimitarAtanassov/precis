import os
from langchain_deepseek import ChatDeepSeek  # Hypothetical import
from modelo_kit.llm_base import BaseLLMService

class DeepSeekLLMService(BaseLLMService):
    def __init__(self, model_name="deepseek-chat"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment.")
        self.chat = ChatDeepSeek(api_key=api_key, model=model_name)

    def ask(self, prompt: str) -> str:
        response = self.chat.invoke(prompt)
        return response.content