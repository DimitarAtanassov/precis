"""Domain layer: pure models and error types with no I/O dependencies.

This package is the stable core of precis. Everything here is framework- and
transport-agnostic so it can be reused by the CLI, the API, tests, and any
future surface without dragging in adapters.
"""

from precis.domain.content import (
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
from precis.domain.errors import (
    ConfigError,
    LLMError,
    ParseError,
    PipelineError,
    PrecisError,
    PromptNotFoundError,
    StructuredOutputError,
)
from precis.domain.llm import LLMOutput, Prompt, Summary

__all__ = [
    # Content + summary models
    "ChunkSummaryOutput",
    "ContributionsOutput",
    "ExecutiveSummaryOutput",
    "ObsidianLink",
    "ObsidianNote",
    "Paper",
    "PaperSummary",
    "ParsedPaper",
    "Section",
    "SectionSummary",
    "SectionSummaryOutput",
    "VaultStats",
    # LLM-interaction models
    "LLMOutput",
    "Prompt",
    "Summary",
    # Errors
    "ConfigError",
    "LLMError",
    "ParseError",
    "PipelineError",
    "PrecisError",
    "PromptNotFoundError",
    "StructuredOutputError",
]
