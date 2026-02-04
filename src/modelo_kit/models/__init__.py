"""
Models for modelo_kit.
"""

from modelo_kit.models.content import (
    ChunkSummaryOutput,
    ContributionsOutput,
    ExecutiveSummaryOutput,
    ObsidianLink,
    ObsidianNote,
    Paper,
    PaperSummary,
    ParsedPaper,
    Section,
    SectionSummary,
    SectionSummaryOutput,
    VaultStats,
)
from modelo_kit.models.llm import (
    LLMOutput,
    Prompt,
    Summary,
)

__all__ = [
    # LLM models
    "LLMOutput",
    "Prompt",
    "Summary",
    # Paper models
    "Paper",
    "ParsedPaper",
    "Section",
    "PaperSummary",
    "SectionSummary",
    # Structured output models
    "SectionSummaryOutput",
    "ChunkSummaryOutput",
    "ExecutiveSummaryOutput",
    "ContributionsOutput",
    # Obsidian models
    "ObsidianNote",
    "ObsidianLink",
    "VaultStats",
]
