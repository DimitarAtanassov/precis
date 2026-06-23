"""
Enums for type-safe choices throughout the application.
"""

from enum import Enum


class LLMProvider(Enum):
    """
    Supported LLM providers with their default models.

    Each provider is defined as a tuple of (provider_name, default_model).
    This enum is used for menu selection and configuration.

    Example:
        >>> provider = LLMProvider.CLAUDE
        >>> provider.provider_name
        'claude'
        >>> provider.default_model
        'claude-sonnet-4-5-20250929'
    """

    CLAUDE = ("claude", "claude-sonnet-4-5-20250929")
    OPENAI = ("openai", "gpt-4o")
    GEMINI = ("gemini", "gemini-2.0-flash")
    DEEPSEEK = ("deepseek", "deepseek-chat")

    @property
    def provider_name(self) -> str:
        """Get the provider identifier string."""
        return self.value[0]

    @property
    def default_model(self) -> str:
        """Get the default model for this provider."""
        return self.value[1]

    @classmethod
    def from_choice(cls, choice: str) -> "LLMProvider":
        """
        Get provider from menu choice number.

        Args:
            choice: Menu selection ("1", "2", "3", or "4").

        Returns:
            Corresponding LLMProvider, defaults to CLAUDE if invalid.
        """
        mapping = {
            "1": cls.CLAUDE,
            "2": cls.OPENAI,
            "3": cls.GEMINI,
            "4": cls.DEEPSEEK,
        }
        return mapping.get(choice, cls.CLAUDE)

    @classmethod
    def choices(cls) -> list[tuple[str, str]]:
        """Get list of (number, display_name) for menu display."""
        return [
            ("1", "Claude"),
            ("2", "OpenAI"),
            ("3", "Gemini"),
            ("4", "DeepSeek"),
        ]
