"""
LLM Service for managing LLM interactions.

Centralizes LLM configuration and prompt handling to reduce code duplication.
"""

from dataclasses import dataclass

from precis.config import Settings, get_settings
from precis.llm_factory import get_llm_service
from precis.ports import LLMProvider
from precis.services.prompt_service import PromptService


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

    def __init__(
        self,
        prompts: PromptService | None = None,
        settings: Settings | None = None,
    ) -> None:
        """
        Initialize the service.

        Args:
            prompts: PromptService instance. Defaults to a settings-driven one.
            settings: Application settings used to resolve provider credentials.
        """
        self._settings = settings or get_settings()
        self._prompts = prompts or PromptService.from_settings(self._settings)
        self._llm: LLMProvider | None = None
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
        self._llm = get_llm_service(provider, model, self._settings)

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

    def get_llm(self) -> LLMProvider:
        """
        Get the underlying LLM instance.

        Returns:
            The configured LLMProvider instance.

        Raises:
            RuntimeError: If service not configured.
        """
        if not self._llm:
            msg = "LLMService not configured. Call configure() first."
            raise RuntimeError(msg)
        return self._llm

    async def generate(self, request: LLMRequest) -> str:
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

        return await self._llm.ask(prompt.user_prompt)

    async def ask(self, user_prompt: str, system_prompt: str | None = None) -> str:
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

        return await self._llm.ask(user_prompt)
