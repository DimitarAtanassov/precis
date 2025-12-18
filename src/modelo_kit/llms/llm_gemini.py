import os
from langchain_google_genai import ChatGoogleGenerativeAI
from modelo_kit.llms.llm_base import BaseLLMService

class GeminiLLMService(BaseLLMService):
    def __init__(self, model_name="gemini-pro"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment.")
        self.chat = ChatGoogleGenerativeAI(api_key=api_key, model=model_name)
        self.system_prompt = None

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def ask(self, prompt: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.chat.invoke(messages)
        return response.content