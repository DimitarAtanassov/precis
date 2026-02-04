"""
Models for parsed content.

Contains models for papers, Obsidian notes, and other content sources.
"""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

# =============================================================================
# Paper Models
# =============================================================================


class Section(BaseModel):
    """A section within a paper."""

    title: str
    content: str
    level: int = 1  # Heading level (1 = h1, 2 = h2, etc.)
    page_start: int = 0
    page_end: int = 0
    subsections: list["Section"] = Field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.content.split())


class Paper(BaseModel):
    """A parsed research paper."""

    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    sections: list[Section] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    source_path: str = ""
    total_pages: int = 0

    @property
    def full_text(self) -> str:
        """Get the complete text of the paper."""
        parts = [self.title, self.abstract]
        parts.extend(section.content for section in self.sections)
        return "\n\n".join(parts)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


# Alias for backward compatibility
ParsedPaper = Paper


# =============================================================================
# Summary Models
# =============================================================================


class SectionSummary(BaseModel):
    """Summary of a single section."""

    section_title: str
    level: int = 1
    summary: str
    key_points: list[str] = Field(default_factory=list)
    page_range: str = ""


class PaperSummary(BaseModel):
    """Complete summary of a research paper."""

    title: str
    abstract: str = ""
    executive_summary: str = ""
    section_summaries: list[SectionSummary] = Field(default_factory=list)
    key_contributions: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Convert summary to markdown format."""
        parts = [
            f"# {self.title}",
            "",
            "## Executive Summary",
            self.executive_summary,
            "",
        ]

        if self.key_contributions:
            parts.append("## Key Contributions")
            parts.extend(f"- {contrib}" for contrib in self.key_contributions)
            parts.append("")

        if self.section_summaries:
            parts.append("## Section Summaries")
            for section in self.section_summaries:
                indent = "  " * (section.level - 1)
                parts.append(f"{indent}### {section.section_title}")
                parts.append(f"{indent}{section.summary}")
                if section.key_points:
                    parts.extend(f"{indent}- {point}" for point in section.key_points)
                parts.append("")

        return "\n".join(parts)


# =============================================================================
# Structured Output Models (for LLM responses)
# =============================================================================


class SectionSummaryOutput(BaseModel):
    """Structured output for section summarization."""

    summary: str
    key_points: list[str] = Field(default_factory=list)


class ChunkSummaryOutput(BaseModel):
    """Structured output for chunk summarization."""

    updated_summary: str


class ExecutiveSummaryOutput(BaseModel):
    """Structured output for executive summary."""

    executive_summary: str


class ContributionsOutput(BaseModel):
    """Structured output for key contributions extraction."""

    contributions: list[str] = Field(default_factory=list)


# =============================================================================
# Obsidian Models
# =============================================================================


class ObsidianLink(BaseModel):
    """A link to another note in an Obsidian vault."""

    target: str
    alias: str | None = None
    is_embed: bool = False


class ObsidianNote(BaseModel):
    """A parsed Obsidian markdown note."""

    title: str
    path: Path
    content: str
    body: str
    frontmatter: dict[str, str | list[str] | int | float | bool] = Field(
        default_factory=dict
    )
    tags: list[str] = Field(default_factory=list)
    links: list[ObsidianLink] = Field(default_factory=list)
    backlinks: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    modified_at: datetime | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def word_count(self) -> int:
        return len(self.body.split())

    def to_context_string(self, include_metadata: bool = True) -> str:
        parts = [f"# {self.title}", ""]
        if include_metadata and self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
            parts.append("")
        parts.append(self.body)
        return "\n".join(parts)


class VaultStats(BaseModel):
    """Statistics about an Obsidian vault."""

    total_notes: int = 0
    total_words: int = 0
    total_links: int = 0
    unique_tags: list[str] = Field(default_factory=list)
    orphan_notes: list[str] = Field(default_factory=list)
