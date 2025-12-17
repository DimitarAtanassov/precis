import os
import httpx

class DeepSeekLLMService():
    API_URL = "https://api.deepseek.com/v1/chat/completions"  # Update if DeepSeek docs specify otherwise

    def __init__(self, model_name="deepseek-chat"):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment.")
        self.model_name = model_name

    def ask(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        with httpx.Client(timeout=30) as client:
            response = client.post(self.API_URL, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
        # Adjust the following line based on DeepSeek's actual response structure
        return result["choices"][0]["message"]["content"]