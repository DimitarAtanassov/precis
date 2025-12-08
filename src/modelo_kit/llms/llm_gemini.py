import os
from langchain_google_genai import ChatGoogleGenerativeAI
from modelo_kit.llm_base import BaseLLMService

class GeminiLLMService(BaseLLMService):
    def __init__(self, model_name="gemini-pro"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment.")
        self.chat = ChatGoogleGenerativeAI(api_key=api_key, model=model_name)

    def ask(self, prompt: str) -> str:
        response = self.chat.invoke(prompt)
        return response.content