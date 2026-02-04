from modelo_kit.llms.llm_base import BaseLLMService
from modelo_kit.llms.llm_claude import ClaudeLLMService
from modelo_kit.llms.llm_deepseek import DeepSeekLLMService
from modelo_kit.llms.llm_gemini import GeminiLLMService
from modelo_kit.llms.llm_openai import OpenAILLMService


def get_llm_service(provider: str, model_name: str) -> BaseLLMService:
    providers = {
        "openai": OpenAILLMService,
        "claude": ClaudeLLMService,
        "gemini": GeminiLLMService,
        "deepseek": DeepSeekLLMService,
    }
    if provider not in providers:
        raise ValueError(f"Provider '{provider}' not supported.")
    return providers[provider](model_name=model_name)  # type: ignore[abstract]
