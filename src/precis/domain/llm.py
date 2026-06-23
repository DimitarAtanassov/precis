"""
Models for LLM interactions.

Contains models for prompts, outputs, and summaries.
"""

from pydantic import BaseModel, Field


class Prompt(BaseModel):
    """A prompt with optional system and user components."""

    system_prompt: str = ""
    user_prompt: str = ""

    def __str__(self) -> str:
        if self.system_prompt:
            return f"[System]: {self.system_prompt}\n[User]: {self.user_prompt}"
        return self.user_prompt


class LLMOutput(BaseModel):
    """Structured output from an LLM call."""

    content: str
    model: str = ""
    tokens_used: int = 0
    finish_reason: str = ""

    @property
    def is_complete(self) -> bool:
        """Check if the response completed normally."""
        return self.finish_reason in ("stop", "end_turn", "")


class Summary(BaseModel):
    """A summary of content with metadata."""

    text: str
    source: str = ""
    word_count: int = 0
    key_points: list[str] = Field(default_factory=list)

    def __str__(self) -> str:
        return self.text
