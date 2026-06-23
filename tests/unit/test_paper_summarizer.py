"""Characterization test for the end-to-end paper summarization pipeline.

Runs the real PaperSummarizer against a deterministic FakeLLMService and the
bundled prompts.yaml (no database, no network). This is the most important
safety-net test: Phases 2-3 re-express this pipeline and must keep this green.
"""

import pytest

from precis.services import summarizer_service
from precis.services.prompt_service import PromptService
from precis.services.summarizer_service import PaperSummarizer, SummarizerConfig


@pytest.fixture(autouse=True)
def use_yaml_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the summarizer to the bundled YAML prompts instead of the DB."""
    monkeypatch.setattr(
        summarizer_service,
        "PromptService",
        lambda: PromptService.from_yaml(),
    )


async def test_summarize_produces_full_summary(fake_llm, sample_paper) -> None:
    config = SummarizerConfig.for_model("fake-model-1", verbose=False)
    summarizer = PaperSummarizer(fake_llm, config)

    result = await summarizer.summarize(sample_paper)

    # Top-level summary structure mirrors the source paper.
    assert result.title == sample_paper.title
    assert result.abstract == sample_paper.abstract
    assert result.executive_summary == "fake executive summary"
    assert result.key_contributions == ["contribution a", "contribution b"]

    # One section summary per flattened section (2 top-level + 1 subsection).
    assert len(result.section_summaries) == 3
    assert {s.section_title for s in result.section_summaries} == {
        "Introduction",
        "Background",
        "Method",
    }
    for section_summary in result.section_summaries:
        assert section_summary.summary == "fake section summary"

    # The pipeline issued exactly the expected structured calls.
    schemas_called = [name for name, _ in fake_llm.calls]
    assert schemas_called.count("SectionSummaryOutput") == 3
    assert schemas_called.count("ExecutiveSummaryOutput") == 1
    assert schemas_called.count("ContributionsOutput") == 1


async def test_long_section_is_chunked(fake_llm, sample_paper) -> None:
    # Force chunking by setting a tiny token budget.
    config = SummarizerConfig.for_model("fake-model-1", verbose=False)
    config.max_chunk_tokens = 1

    summarizer = PaperSummarizer(fake_llm, config)
    await summarizer.summarize(sample_paper)

    schemas_called = [name for name, _ in fake_llm.calls]
    # Long sections route through the chunk-summary path.
    assert "ChunkSummaryOutput" in schemas_called
