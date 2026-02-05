"""
LLM Service for managing LLM interactions.

Centralizes LLM configuration and prompt handling to reduce code duplication.
"""

from dataclasses import dataclass

from modelo_kit.llm_factory import get_llm_service
from modelo_kit.llms.llm_base import BaseLLMService
from modelo_kit.services.prompt_service import PromptService


@dataclass
class LLMRequest:
    """
    Request for LLM generation.

    Encapsulates all information needed to make an LLM call.
    """

    prompt_key: str
    variables: dict[str, str]


class LLMService:
    """
    Service for LLM calls with prompt management.

    Handles the common pattern of:
    1. Getting prompts from YAML
    2. Setting system prompt
    3. Making the LLM call

    Usage:
        llm_service = LLMService()
        llm_service.configure("claude", "claude-sonnet-4-5-20250929")

        response = llm_service.generate(LLMRequest(
            prompt_key="obsidian_note_summary",
            variables={"title": note.title, "content": note.body}
        ))
    """

    def __init__(self, prompts: PromptService | None = None) -> None:
        """
        Initialize the service.

        Args:
            prompts: PromptService instance. Uses singleton if not provided.
        """
        self._prompts = prompts or PromptService()
        self._llm: BaseLLMService | None = None
        self._provider: str = ""
        self._model: str = ""

    def configure(self, provider: str, model: str) -> None:
        """
        Configure the LLM provider.

        Args:
            provider: Provider name (e.g., "claude", "openai").
            model: Model name (e.g., "claude-sonnet-4-5-20250929").
        """
        self._provider = provider
        self._model = model
        self._llm = get_llm_service(provider, model)

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._model

    @property
    def provider(self) -> str:
        """Get the current provider name."""
        return self._provider

    @property
    def is_configured(self) -> bool:
        """Check if the service is configured."""
        return self._llm is not None

    def get_llm(self) -> BaseLLMService:
        """
        Get the underlying LLM instance.

        Returns:
            The configured BaseLLMService instance.

        Raises:
            RuntimeError: If service not configured.
        """
        if not self._llm:
            msg = "LLMService not configured. Call configure() first."
            raise RuntimeError(msg)
        return self._llm

    def generate(self, request: LLMRequest) -> str:
        """
        Generate response using configured LLM.

        Args:
            request: LLMRequest with prompt key and variables.

        Returns:
            Generated response text.

        Raises:
            RuntimeError: If service not configured.
        """
        if not self._llm:
            msg = "LLMService not configured. Call configure() first."
            raise RuntimeError(msg)

        prompt = self._prompts.get(request.prompt_key, **request.variables)

        if prompt.system_prompt:
            self._llm.set_system_prompt(prompt.system_prompt)

        result: str = self._llm.ask(prompt.user_prompt)
        return result

    def ask(self, user_prompt: str, system_prompt: str | None = None) -> str:
        """
        Direct ask without using prompt templates.

        Args:
            user_prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            Generated response text.
        """
        if not self._llm:
            msg = "LLMService not configured. Call configure() first."
            raise RuntimeError(msg)

        if system_prompt:
            self._llm.set_system_prompt(system_prompt)

        result: str = self._llm.ask(user_prompt)
        return result
