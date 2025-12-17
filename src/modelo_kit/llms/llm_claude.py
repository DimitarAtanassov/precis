import os
from langchain_anthropic import ChatAnthropic

class ClaudeLLMService():
    def __init__(self, model_name="claude-3-opus-20240229"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment.")
        self.chat = ChatAnthropic(api_key=api_key, model=model_name)

    def ask(self, prompt: str) -> str:
        response = self.chat.invoke(prompt)
        return response.content