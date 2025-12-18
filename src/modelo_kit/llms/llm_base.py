from abc import ABC, abstractmethod

class BaseLLMService(ABC):
    @abstractmethod
    def ask(self, prompt: str) -> str:
        pass

    @abstractmethod
    def set_system_prompt(self, system_prompt: str):
        """
        Set the system prompt for the LLM, if supported by the provider.
        """
        pass