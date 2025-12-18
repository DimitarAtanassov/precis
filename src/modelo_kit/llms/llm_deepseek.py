import os
import httpx
from modelo_kit.llms.llm_base import BaseLLMService

class DeepSeekLLMService(BaseLLMService):
    API_URL = "https://api.deepseek.com/v1/chat/completions"

    def __init__(self, model_name="deepseek-chat"):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment.")
        self.model_name = model_name
        self.system_prompt = None

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def ask(self, prompt: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model_name,
            "messages": messages,
        }
        with httpx.Client(timeout=30) as client:
            response = client.post(self.API_URL, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
        return result["choices"][0]["message"]["content"]