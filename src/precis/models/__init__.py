"""
Models for precis.
"""

from precis.models.content import (
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
from precis.models.llm import (
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
