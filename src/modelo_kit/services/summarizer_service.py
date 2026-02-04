"""
Service for summarizing research papers using LLMs.
Handles large documents by chunking and maintaining context.
"""

from dataclasses import dataclass, field, fields
from typing import Protocol, runtime_checkable

import tiktoken

from modelo_kit.llms.llm_base import BaseLLMService
from modelo_kit.models import (
    ChunkSummaryOutput,
    ContributionsOutput,
    ExecutiveSummaryOutput,
    PaperSummary,
    ParsedPaper,
    Section,
    SectionSummary,
    SectionSummaryOutput,
)
from modelo_kit.services.prompt_service import PromptService

# --- Token Counting Strategy Pattern ---


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for token counting strategies."""

    def count(self, text: str) -> int:
        """Count tokens in text."""
        ...


class SimpleTokenCounter:
    """
    Simple token counter using word-based estimation.
    Approximates ~4 characters per token (industry standard).
    """

    CHARS_PER_TOKEN = 4

    def count(self, text: str) -> int:
        return len(text) // self.CHARS_PER_TOKEN


class TiktokenCounter:
    """
    Accurate token counter using tiktoken (OpenAI's tokenizer).
    Falls back to simple counting if tiktoken unavailable.
    """

    def __init__(self, model: str = "gpt-4") -> None:
        self._encoder = None
        self._fallback = SimpleTokenCounter()

        try:
            # Map common model names to tiktoken encodings
            encoding_map = {
                "gpt-4": "cl100k_base",
                "gpt-4o": "cl100k_base",
                "gpt-5": "cl100k_base",
                "claude": "cl100k_base",
                "gemini": "cl100k_base",
            }
            encoding_name = encoding_map.get(model.split("-")[0], "cl100k_base")
            self._encoder = tiktoken.get_encoding(encoding_name)
        except ImportError:
            pass  # Use fallback

    def count(self, text: str) -> int:
        if self._encoder:
            return len(self._encoder.encode(text))
        return self._fallback.count(text)


# --- Model Defaults ---


@dataclass
class ModelDefaults:
    """Default token limits for common LLM models."""

    DEFAULTS = {
        # Claude models
        "claude-sonnet-4-5-20250929": 200_000,
        "claude-haiku-4-5-20251001": 200_000,
        "claude-opus-4-5-20251101": 200_000,
        # OpenAI GPT-5 models
        "gpt-5-2025-08-07": 400_000,
        "gpt-5-nano-2025-08-07": 400_000,
        "gpt-5-mini-2025-08-07": 400_000,
        "gpt-5.2-pro-2025-12-11": 400_000,
        # Google Gemini 3.x models
        "gemini-3-pro-preview": 1_048_576,
        "gemini-3-flash-preview": 1_048_576,
        # Google Gemini 2.5 models
        "gemini-2.5-pro": 1_048_576,
        "gemini-2.5-flash": 1_048_576,
        "gemini-2.5-flash-lite": 1_048_576,
        # DeepSeek models
        "deepseek-chat": 128_000,
        "deepseek-reasoner": 128_000,
    }

    CHUNK_RATIO = 0.3

    @classmethod
    def get_max_chunk_tokens(cls, model: str) -> int:
        """Get recommended max chunk tokens for a model."""
        context_size = cls.DEFAULTS.get(model, 8_000)
        return int(context_size * cls.CHUNK_RATIO)

    @classmethod
    def get_context_size(cls, model: str) -> int:
        """Get full context window size for a model."""
        return cls.DEFAULTS.get(model, 8_000)


# --- Configuration ---


@dataclass
class SummarizerConfig:
    """Configuration for the paper summarizer."""

    max_chunk_tokens: int | None = None
    model_name: str = "gpt-4o"
    include_section_summaries: bool = True
    verbose: bool = True
    token_counter: TokenCounter = field(default_factory=SimpleTokenCounter)

    def __post_init__(self) -> None:
        if self.max_chunk_tokens is None:
            self.max_chunk_tokens = ModelDefaults.get_max_chunk_tokens(self.model_name)

    @classmethod
    def for_model(cls, model_name: str, **kwargs: object) -> "SummarizerConfig":
        """Factory method to create config optimized for a specific model."""
        valid_fields = {f.name for f in fields(cls)}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
        # Handle token_counter
        token_counter = filtered_kwargs.get("token_counter")
        if token_counter is not None and not isinstance(token_counter, TokenCounter):
            token_counter = SimpleTokenCounter()
        # Handle bools
        include_section_summaries = bool(
            filtered_kwargs.get("include_section_summaries", True)
        )
        verbose = bool(filtered_kwargs.get("verbose", True))
        return cls(
            model_name=model_name,
            max_chunk_tokens=ModelDefaults.get_max_chunk_tokens(model_name),
            include_section_summaries=include_section_summaries,
            verbose=verbose,
            token_counter=(
                token_counter if token_counter is not None else SimpleTokenCounter()
            ),
        )


class PaperSummarizer:
    """
    Summarizes research papers using LLMs with structured outputs.

    Handles large documents by:
    1. Chunking long sections (token-aware)
    2. Maintaining running context
    3. Hierarchical summarization (sections → paper)

    Usage:
        llm = get_llm_service("claude", "claude-sonnet-4-5-20250929")
        config = SummarizerConfig.for_model("claude-sonnet-4-5-20250929")

        summarizer = PaperSummarizer(llm, config)
        summary = summarizer.summarize(parsed_paper)
        print(summary.to_markdown())
    """

    MAX_CONTEXT_TOKENS = 500

    def __init__(
        self,
        llm: BaseLLMService,
        config: SummarizerConfig | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or SummarizerConfig()
        self.prompts = PromptService()
        self._token_counter = self.config.token_counter

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using configured counter."""
        return self._token_counter.count(text)

    def summarize(self, paper: ParsedPaper) -> PaperSummary:
        """Generate a complete summary of a research paper."""
        self._log(f"📝 Summarizing: {paper.title}")
        self._log(f"   Sections: {len(paper.sections)}, Pages: {paper.total_pages}")
        self._log(f"   Max chunk: {self.config.max_chunk_tokens} tokens")

        # Step 1: Summarize each section
        section_summaries = self._summarize_sections(paper)

        # Step 2: Generate executive summary
        executive_summary = self._generate_executive_summary(
            paper.title, section_summaries
        )

        # Step 3: Extract key contributions
        contributions = self._extract_contributions(paper.title, section_summaries)

        return PaperSummary(
            title=paper.title,
            abstract=paper.abstract,
            executive_summary=executive_summary,
            section_summaries=section_summaries,
            key_contributions=contributions,
        )

    def _summarize_sections(self, paper: ParsedPaper) -> list[SectionSummary]:
        """Summarize all sections, maintaining context."""
        summaries = []
        running_context = ""

        all_sections = self._flatten_sections(paper.sections)

        for i, section in enumerate(all_sections):
            self._log(f"   [{i + 1}/{len(all_sections)}] {section.title}")

            summary = self._summarize_section(
                paper_title=paper.title, section=section, context=running_context
            )
            summaries.append(summary)

            running_context = self._update_context(running_context, summary)

        return summaries

    def _summarize_section(
        self, paper_title: str, section: Section, context: str
    ) -> SectionSummary:
        """Summarize a single section, handling long content via chunking."""
        content = section.content
        content_tokens = self._count_tokens(content)

        if (
            self.config.max_chunk_tokens is not None
            and content_tokens <= self.config.max_chunk_tokens
        ):
            return self._summarize_content(paper_title, section, content, context)

        return self._summarize_long_section(paper_title, section, content, context)

    def _summarize_content(
        self, paper_title: str, section: Section, content: str, context: str
    ) -> SectionSummary:
        """Summarize content that fits in one LLM call using structured output."""
        prompt = self.prompts.get(
            "paper_section_concise",
            paper_title=paper_title,
            section_title=section.title,
            content=content,
            context=context or "This is the first section.",
        )

        # Use structured output - no format instructions needed in prompt
        response: SectionSummaryOutput = self.llm.ask_structured(
            prompt=prompt.user_prompt or "",
            output_schema=SectionSummaryOutput,
            system_prompt=prompt.system_prompt or "",
        )

        return SectionSummary(
            section_title=section.title,
            level=section.level,
            summary=response.summary,
            key_points=response.key_points,
            page_range=f"{section.page_start}-{section.page_end}",
        )

    def _summarize_long_section(
        self, paper_title: str, section: Section, content: str, context: str
    ) -> SectionSummary:
        """Handle sections too long for one LLM call."""
        chunks = self._chunk_content(content)
        running_summary = ""

        self._log(f"      (Chunking: {len(chunks)} parts)")

        for chunk in chunks:
            prompt = self.prompts.get(
                "paper_chunk_summary",
                paper_title=paper_title,
                section_title=section.title,
                running_summary=running_summary or "Starting new section.",
                content=chunk,
            )

            # Use structured output for chunk summaries
            response: ChunkSummaryOutput = self.llm.ask_structured(
                prompt=prompt.user_prompt or "",
                output_schema=ChunkSummaryOutput,
                system_prompt=prompt.system_prompt or "",
            )
            running_summary = response.updated_summary

        return SectionSummary(
            section_title=section.title,
            level=section.level,
            summary=running_summary,
            key_points=[],
            page_range=f"{section.page_start}-{section.page_end}",
        )

    def _generate_executive_summary(
        self, title: str, section_summaries: list[SectionSummary]
    ) -> str:
        """Generate executive summary from section summaries."""
        self._log("   Generating executive summary...")

        summaries_text = "\n\n".join(
            [f"### {s.section_title}\n{s.summary}" for s in section_summaries]
        )

        prompt = self.prompts.get(
            "paper_executive_summary",
            paper_title=title,
            section_summaries=summaries_text,
        )

        # Use structured output
        response: ExecutiveSummaryOutput = self.llm.ask_structured(
            prompt=prompt.user_prompt or "",
            output_schema=ExecutiveSummaryOutput,
            system_prompt=prompt.system_prompt or "",
        )

        return response.executive_summary

    def _extract_contributions(
        self, title: str, section_summaries: list[SectionSummary]
    ) -> list[str]:
        """Extract key contributions from summaries."""
        self._log("   Extracting contributions...")

        summaries_text = "\n".join(
            [f"- {s.section_title}: {s.summary}" for s in section_summaries]
        )

        prompt = self.prompts.get(
            "paper_extract_contributions", paper_title=title, summaries=summaries_text
        )

        # Use structured output
        response: ContributionsOutput = self.llm.ask_structured(
            prompt=prompt.user_prompt or "",
            output_schema=ContributionsOutput,
            system_prompt=prompt.system_prompt or "",
        )

        return response.contributions

    def _flatten_sections(self, sections: list[Section]) -> list[Section]:
        """Flatten nested sections into a list."""
        result = []
        for section in sections:
            result.append(section)
            if section.subsections:
                result.extend(self._flatten_sections(section.subsections))
        return result

    def _chunk_content(self, content: str) -> list[str]:
        """Split content into chunks that fit in context window."""
        max_tokens = self.config.max_chunk_tokens
        paragraphs = content.split("\n\n")

        chunks = []
        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            if max_tokens is not None and current_tokens + para_tokens < max_tokens:
                current_chunk += para + "\n\n"
                current_tokens += para_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
                current_tokens = para_tokens

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _update_context(self, current_context: str, new_summary: SectionSummary) -> str:
        """Update running context with new section summary."""
        new_entry = f"{new_summary.section_title}: {new_summary.summary}"
        updated = f"{current_context}\n{new_entry}".strip()

        while self._count_tokens(updated) > self.MAX_CONTEXT_TOKENS:
            first_newline = updated.find("\n")
            if first_newline > 0:
                updated = updated[first_newline + 1 :]
            else:
                break

        return updated

    def _log(self, message: str) -> None:
        """Print log message if verbose."""
        if self.config.verbose:
            print(message)
