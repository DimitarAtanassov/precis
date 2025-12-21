"""
Pydantic models for structured LLM outputs.
Used with LangChain's with_structured_output for type-safe responses.
"""
from typing import List
from pydantic import BaseModel, Field


class SectionSummaryOutput(BaseModel):
    """Structured output for section summaries."""
    summary: str = Field(description="A 2-3 sentence summary of the section")
    key_points: List[str] = Field(
        default_factory=list,
        description="2-4 bullet points of key takeaways"
    )


class ExecutiveSummaryOutput(BaseModel):
    """Structured output for executive summaries."""
    executive_summary: str = Field(
        description="A 3-5 paragraph executive summary synthesizing the paper"
    )


class ContributionsOutput(BaseModel):
    """Structured output for key contributions."""
    contributions: List[str] = Field(
        description="3-7 key contributions, each as a single clear sentence"
    )


class ChunkSummaryOutput(BaseModel):
    """Structured output for chunk summaries."""
    updated_summary: str = Field(
        description="Updated running summary incorporating new content, under 500 words"
    )