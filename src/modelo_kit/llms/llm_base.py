from abc import ABC, abstractmethod

class BaseLLMService(ABC):
    @abstractmethod
    def ask(self, prompt: str) -> str:
        pass