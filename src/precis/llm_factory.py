"""Factory for constructing LLM provider services from configuration."""

from precis.config import Settings, get_settings
from precis.llms.llm_base import BaseLLMService
from precis.llms.llm_claude import ClaudeLLMService
from precis.llms.llm_deepseek import DeepSeekLLMService
from precis.llms.llm_gemini import GeminiLLMService
from precis.llms.llm_openai import OpenAILLMService

# Map of provider names to their concrete service classes
_PROVIDERS: dict[str, type[BaseLLMService]] = {
    "openai": OpenAILLMService,
    "claude": ClaudeLLMService,
    "gemini": GeminiLLMService,
    "deepseek": DeepSeekLLMService,
}


def get_llm_service(
    provider: str,
    model_name: str,
    settings: Settings | None = None,
) -> BaseLLMService:
    """
    Create an LLM service instance with credentials resolved from settings.

    Args:
        provider: The LLM provider name (openai, claude, gemini, deepseek).
        model_name: The model name to use.
        settings: Settings to source the API key/timeout from. Defaults to the
            process-wide settings.

    Returns:
        An instance of the appropriate LLM service.

    Raises:
        ValueError: If the provider is not supported.
    """
    if provider not in _PROVIDERS:
        supported = ", ".join(_PROVIDERS.keys())
        raise ValueError(f"Provider '{provider}' not supported. Use: {supported}")

    settings = settings or get_settings()
    service_class = _PROVIDERS[provider]
    return service_class(
        model_name=model_name,
        api_key=settings.api_key_for(provider),
        timeout=settings.llm_request_timeout,
    )
