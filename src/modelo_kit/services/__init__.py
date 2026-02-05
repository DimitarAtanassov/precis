"""
Services module for business logic.

Contains:
- LLMService: Centralized LLM call management
- ObsidianVault: Obsidian note repository
- PromptService: YAML prompt template management
- PaperSummarizer: Research paper summarization
- WebService: Web content fetching
"""

from modelo_kit.services.llm_service import LLMRequest, LLMService
from modelo_kit.services.obsidian_vault import ObsidianVault
from modelo_kit.services.prompt_service import PromptService
from modelo_kit.services.summarizer_service import PaperSummarizer, SummarizerConfig
from modelo_kit.services.web_service import WebService

__all__ = [
    "LLMRequest",
    "LLMService",
    "ObsidianVault",
    "PaperSummarizer",
    "PromptService",
    "SummarizerConfig",
    "WebService",
]
