from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base exception for LLM errors."""

    pass


class StructuredOutputError(LLMError):
    """Raised when structured output parsing fails."""

    pass


class BaseLLMService(ABC):
    @abstractmethod
    def ask(self, prompt: str) -> str:
        """Send a prompt and get a string response."""
        pass

    @abstractmethod
    def set_system_prompt(self, system_prompt: str) -> None:
        """Set the system prompt for the LLM."""
        pass

    @abstractmethod
    def ask_structured(
        self, prompt: str, output_schema: type[T], system_prompt: str | None = None
    ) -> T:
        """
        Send a prompt and get a structured response.

        Args:
            prompt: The user prompt
            output_schema: Pydantic model class for the response
            system_prompt: Optional system prompt override

        Returns:
            Instance of the output_schema model

        Raises:
            StructuredOutputError: If parsing fails after retries
        """
        pass
