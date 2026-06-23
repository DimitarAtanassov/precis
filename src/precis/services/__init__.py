"""
Services module for business logic.

Contains:
- LLMService: Centralized LLM call management
- ObsidianVault: Obsidian note repository
- PromptService: Unified prompt management (YAML or Database)
- PromptProvider: Abstract interface for prompt sources
- PaperSummarizer: Research paper summarization
- WebService: Web content fetching
"""

from precis.services.llm_service import LLMRequest, LLMService
from precis.services.obsidian_vault import ObsidianVault
from precis.services.prompt_provider import (
    CachedPromptProvider,
    DatabasePromptProvider,
    PromptProvider,
    YamlPromptProvider,
)
from precis.services.prompt_service import PromptService
from precis.services.summarizer_service import PaperSummarizer, SummarizerConfig
from precis.services.web_service import WebService

__all__ = [
    # LLM
    "LLMRequest",
    "LLMService",
    # Prompts
    "PromptService",
    "PromptProvider",
    "YamlPromptProvider",
    "DatabasePromptProvider",
    "CachedPromptProvider",
    # Other services
    "ObsidianVault",
    "PaperSummarizer",
    "SummarizerConfig",
    "WebService",
]
