import os
from langchain_openai import ChatOpenAI
from modelo_kit.llms.llm_base import BaseLLMService

class OpenAILLMService(BaseLLMService):
    def __init__(self, model_name="gpt-5"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")
        self.chat = ChatOpenAI(api_key=api_key, model=model_name)

    def ask(self, prompt: str) -> str:
        response = self.chat.invoke(prompt)
        return response.content