from modelo_kit.llms.llm_base import BaseLLMService
from modelo_kit.llms.llm_claude import ClaudeLLMService
from modelo_kit.llms.llm_deepseek import DeepSeekLLMService
from modelo_kit.llms.llm_gemini import GeminiLLMService
from modelo_kit.llms.llm_openai import OpenAILLMService

# Map of provider names to their concrete service classes
_PROVIDERS: dict[str, type[BaseLLMService]] = {
    "openai": OpenAILLMService,
    "claude": ClaudeLLMService,
    "gemini": GeminiLLMService,
    "deepseek": DeepSeekLLMService,
}


def get_llm_service(provider: str, model_name: str) -> BaseLLMService:
    """
    Factory function to create an LLM service instance.

    Args:
        provider: The LLM provider name (openai, claude, gemini, deepseek).
        model_name: The model name to use.

    Returns:
        An instance of the appropriate LLM service.

    Raises:
        ValueError: If the provider is not supported.
    """
    if provider not in _PROVIDERS:
        supported = ", ".join(_PROVIDERS.keys())
        raise ValueError(f"Provider '{provider}' not supported. Use: {supported}")

    service_class = _PROVIDERS[provider]
    return service_class(model_name=model_name)
