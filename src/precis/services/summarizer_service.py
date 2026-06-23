"""Research-paper summarization, expressed as an orchestration pipeline.

The work is decomposed into three composable steps —
``SummarizeSectionsStep`` → ``ExecutiveSummaryStep`` → ``ContributionsStep`` —
run by :class:`precis.orchestration.Pipeline` over a shared
:class:`SummarizationContext`. ``PaperSummarizer`` is the thin facade that wires
the pipeline together. Section summarization is sequential by default (to keep a
running cross-section context); set ``section_concurrency > 1`` to fan out with
a bounded semaphore, trading that running context for latency.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, fields

import tiktoken

from precis.domain import (
    ChunkSummaryOutput,
    ContributionsOutput,
    ExecutiveSummaryOutput,
    PaperSummary,
    ParsedPaper,
    Section,
    SectionSummary,
    SectionSummaryOutput,
)
from precis.observability import get_logger
from precis.orchestration import Pipeline, Step
from precis.ports import LLMProvider, PromptRepository, TokenCounter
from precis.services.prompt_service import PromptService

_logger = get_logger(__name__)

# Token budget for the rolling cross-section context.
MAX_CONTEXT_TOKENS = 500


# --- Token Counting Strategies (implement the ports.TokenCounter contract) ---


class SimpleTokenCounter:
    """Word/char-based token estimate (~4 characters per token)."""

    CHARS_PER_TOKEN = 4

    def count(self, text: str) -> int:
        return len(text) // self.CHARS_PER_TOKEN


class TiktokenCounter:
    """Accurate token counter using tiktoken, falling back to estimation."""

    def __init__(self, model: str = "gpt-4") -> None:
        self._encoder = None
        self._fallback = SimpleTokenCounter()

        try:
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
    section_concurrency: int = 1
    token_counter: TokenCounter = field(default_factory=SimpleTokenCounter)

    def __post_init__(self) -> None:
        if self.max_chunk_tokens is None:
            self.max_chunk_tokens = ModelDefaults.get_max_chunk_tokens(self.model_name)

    @classmethod
    def for_model(cls, model_name: str, **kwargs: object) -> SummarizerConfig:
        """Factory method to create config optimized for a specific model."""
        valid_fields = {f.name for f in fields(cls)}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
        token_counter = filtered_kwargs.get("token_counter")
        if token_counter is not None and not isinstance(token_counter, TokenCounter):
            token_counter = SimpleTokenCounter()
        concurrency = filtered_kwargs.get("section_concurrency", 1)
        return cls(
            model_name=model_name,
            max_chunk_tokens=ModelDefaults.get_max_chunk_tokens(model_name),
            include_section_summaries=bool(
                filtered_kwargs.get("include_section_summaries", True)
            ),
            verbose=bool(filtered_kwargs.get("verbose", True)),
            section_concurrency=concurrency if isinstance(concurrency, int) else 1,
            token_counter=(
                token_counter if token_counter is not None else SimpleTokenCounter()
            ),
        )


# --- Pipeline state ---


@dataclass
class SummarizationContext:
    """Mutable state threaded through the summarization pipeline steps."""

    paper: ParsedPaper
    config: SummarizerConfig
    llm: LLMProvider
    prompts: PromptRepository
    token_counter: TokenCounter
    section_summaries: list[SectionSummary] = field(default_factory=list)
    executive_summary: str = ""
    contributions: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        """Emit a progress log when verbose."""
        if self.config.verbose:
            _logger.info(message)

    def to_summary(self) -> PaperSummary:
        """Assemble the final paper summary from accumulated state."""
        return PaperSummary(
            title=self.paper.title,
            abstract=self.paper.abstract,
            executive_summary=self.executive_summary,
            section_summaries=self.section_summaries,
            key_contributions=self.contributions,
        )


# --- Pure helpers ---


def _flatten_sections(sections: list[Section]) -> list[Section]:
    """Flatten nested sections into a depth-first list."""
    result: list[Section] = []
    for section in sections:
        result.append(section)
        if section.subsections:
            result.extend(_flatten_sections(section.subsections))
    return result


def _chunk_content(
    content: str, max_tokens: int | None, counter: TokenCounter
) -> list[str]:
    """Split content into chunks that fit within ``max_tokens``."""
    paragraphs = content.split("\n\n")
    chunks: list[str] = []
    current_chunk = ""
    current_tokens = 0

    for para in paragraphs:
        para_tokens = counter.count(para)
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


def _update_context(
    current_context: str, new_summary: SectionSummary, counter: TokenCounter
) -> str:
    """Append a section summary to the rolling context, trimming to budget."""
    new_entry = f"{new_summary.section_title}: {new_summary.summary}"
    updated = f"{current_context}\n{new_entry}".strip()

    while counter.count(updated) > MAX_CONTEXT_TOKENS:
        first_newline = updated.find("\n")
        if first_newline > 0:
            updated = updated[first_newline + 1 :]
        else:
            break
    return updated


# --- Steps ---


class SummarizeSectionsStep:
    """Summarize every (flattened) section, optionally with bounded concurrency."""

    name = "summarize_sections"

    async def run(self, ctx: SummarizationContext) -> SummarizationContext:
        sections = _flatten_sections(ctx.paper.sections)
        if ctx.config.section_concurrency > 1:
            ctx.section_summaries = await self._concurrent(ctx, sections)
        else:
            ctx.section_summaries = await self._sequential(ctx, sections)
        return ctx

    async def _sequential(
        self, ctx: SummarizationContext, sections: list[Section]
    ) -> list[SectionSummary]:
        summaries: list[SectionSummary] = []
        running_context = ""
        for i, section in enumerate(sections):
            ctx.log(f"   [{i + 1}/{len(sections)}] {section.title}")
            summary = await self._summarize_section(ctx, section, running_context)
            summaries.append(summary)
            running_context = _update_context(
                running_context, summary, ctx.token_counter
            )
        return summaries

    async def _concurrent(
        self, ctx: SummarizationContext, sections: list[Section]
    ) -> list[SectionSummary]:
        semaphore = asyncio.Semaphore(ctx.config.section_concurrency)

        async def worker(section: Section) -> SectionSummary:
            async with semaphore:
                # No running context when fanning out; each section is independent.
                return await self._summarize_section(ctx, section, "")

        return list(await asyncio.gather(*(worker(s) for s in sections)))

    async def _summarize_section(
        self, ctx: SummarizationContext, section: Section, context: str
    ) -> SectionSummary:
        content_tokens = ctx.token_counter.count(section.content)
        fits = (
            ctx.config.max_chunk_tokens is not None
            and content_tokens <= ctx.config.max_chunk_tokens
        )
        if fits:
            return await self._summarize_content(ctx, section, context)
        return await self._summarize_long(ctx, section, context)

    async def _summarize_content(
        self, ctx: SummarizationContext, section: Section, context: str
    ) -> SectionSummary:
        prompt = ctx.prompts.get(
            "paper_section_concise",
            paper_title=ctx.paper.title,
            section_title=section.title,
            content=section.content,
            context=context or "This is the first section.",
        )
        response: SectionSummaryOutput = await ctx.llm.ask_structured(
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

    async def _summarize_long(
        self,
        ctx: SummarizationContext,
        section: Section,
        context: str,  # noqa: ARG002
    ) -> SectionSummary:
        chunks = _chunk_content(
            section.content, ctx.config.max_chunk_tokens, ctx.token_counter
        )
        ctx.log(f"      (Chunking: {len(chunks)} parts)")
        running_summary = ""
        for chunk in chunks:
            prompt = ctx.prompts.get(
                "paper_chunk_summary",
                paper_title=ctx.paper.title,
                section_title=section.title,
                running_summary=running_summary or "Starting new section.",
                content=chunk,
            )
            response: ChunkSummaryOutput = await ctx.llm.ask_structured(
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


class ExecutiveSummaryStep:
    """Produce an executive summary from the section summaries."""

    name = "executive_summary"

    async def run(self, ctx: SummarizationContext) -> SummarizationContext:
        ctx.log("   Generating executive summary...")
        summaries_text = "\n\n".join(
            f"### {s.section_title}\n{s.summary}" for s in ctx.section_summaries
        )
        prompt = ctx.prompts.get(
            "paper_executive_summary",
            paper_title=ctx.paper.title,
            section_summaries=summaries_text,
        )
        response: ExecutiveSummaryOutput = await ctx.llm.ask_structured(
            prompt=prompt.user_prompt or "",
            output_schema=ExecutiveSummaryOutput,
            system_prompt=prompt.system_prompt or "",
        )
        ctx.executive_summary = response.executive_summary
        return ctx


class ContributionsStep:
    """Extract key contributions from the section summaries."""

    name = "contributions"

    async def run(self, ctx: SummarizationContext) -> SummarizationContext:
        ctx.log("   Extracting contributions...")
        summaries_text = "\n".join(
            f"- {s.section_title}: {s.summary}" for s in ctx.section_summaries
        )
        prompt = ctx.prompts.get(
            "paper_extract_contributions",
            paper_title=ctx.paper.title,
            summaries=summaries_text,
        )
        response: ContributionsOutput = await ctx.llm.ask_structured(
            prompt=prompt.user_prompt or "",
            output_schema=ContributionsOutput,
            system_prompt=prompt.system_prompt or "",
        )
        ctx.contributions = response.contributions
        return ctx


# --- Facade ---


class PaperSummarizer:
    """Summarize a research paper by running the summarization pipeline.

    Usage:
        llm = get_llm_service("claude", "claude-sonnet-4-5-20250929")
        config = SummarizerConfig.for_model("claude-sonnet-4-5-20250929")

        summarizer = PaperSummarizer(llm, config)
        summary = await summarizer.summarize(parsed_paper)
        print(summary.to_markdown())
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: SummarizerConfig | None = None,
        prompts: PromptRepository | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or SummarizerConfig()
        self.prompts = prompts or PromptService()

    def _build_pipeline(self) -> Pipeline[SummarizationContext]:
        steps: list[Step[SummarizationContext]] = [
            SummarizeSectionsStep(),
            ExecutiveSummaryStep(),
            ContributionsStep(),
        ]
        return Pipeline(steps, name="paper_summarization")

    async def summarize(self, paper: ParsedPaper) -> PaperSummary:
        """Generate a complete summary of a research paper."""
        ctx = SummarizationContext(
            paper=paper,
            config=self.config,
            llm=self.llm,
            prompts=self.prompts,
            token_counter=self.config.token_counter,
        )
        ctx.log(f"📝 Summarizing: {paper.title}")
        ctx.log(f"   Sections: {len(paper.sections)}, Pages: {paper.total_pages}")
        ctx.log(f"   Max chunk: {self.config.max_chunk_tokens} tokens")

        result = await self._build_pipeline().run(ctx)
        return result.to_summary()
