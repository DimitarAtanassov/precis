from modelo_kit.llms.llm_openai import OpenAILLMService
from modelo_kit.llms.llm_claude import ClaudeLLMService
from modelo_kit.llms.llm_gemini import GeminiLLMService

def get_llm_service(provider: str, model_name: str):
    providers = {
        "openai": OpenAILLMService,
        "claude": ClaudeLLMService,
        "gemini": GeminiLLMService,
    }
    if provider not in providers:
        raise ValueError(f"Provider '{provider}' not supported.")
    return providers[provider](model_name=model_name)