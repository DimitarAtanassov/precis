"""Ports: the abstract contracts at the edges of the hexagon.

Application/services code depends only on these Protocols, never on concrete
adapters (LangChain providers, YAML/DB prompt stores, the filesystem). This is
the dependency-inversion boundary that keeps the core testable and lets adapters
be swapped without touching business logic.
"""

from precis.ports.llm import LLMProvider
from precis.ports.prompts import PromptRepository
from precis.ports.tokens import TokenCounter

__all__ = ["LLMProvider", "PromptRepository", "TokenCounter"]
