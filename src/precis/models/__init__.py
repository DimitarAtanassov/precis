"""Backward-compatible facade over :mod:`precis.domain`.

The canonical home for models is now ``precis.domain``. This module re-exports
them so existing ``from precis.models import ...`` call sites keep working.
Prefer importing from ``precis.domain`` in new code.
"""

from precis.domain import (
    ChunkSummaryOutput,
    ContributionsOutput,
    ExecutiveSummaryOutput,
    LLMOutput,
    ObsidianLink,
    ObsidianNote,
    Paper,
    PaperSummary,
    ParsedPaper,
    Prompt,
    Section,
    SectionSummary,
    SectionSummaryOutput,
    Summary,
    VaultStats,
)

__all__ = [
    "ChunkSummaryOutput",
    "ContributionsOutput",
    "ExecutiveSummaryOutput",
    "LLMOutput",
    "ObsidianLink",
    "ObsidianNote",
    "Paper",
    "PaperSummary",
    "ParsedPaper",
    "Prompt",
    "Section",
    "SectionSummary",
    "SectionSummaryOutput",
    "Summary",
    "VaultStats",
]
