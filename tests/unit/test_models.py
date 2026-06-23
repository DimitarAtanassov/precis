"""Characterization tests for domain model behavior."""

from precis.models import Paper, PaperSummary, Section, SectionSummary


def test_section_word_count() -> None:
    assert Section(title="t", content="one two three").word_count == 3


def test_paper_full_text_and_word_count() -> None:
    paper = Paper(
        title="Title",
        abstract="An abstract here.",
        sections=[Section(title="S1", content="body words go here")],
    )
    assert "Title" in paper.full_text
    assert "An abstract here." in paper.full_text
    assert "body words go here" in paper.full_text
    # title(1) + abstract(3) + section(4)
    assert paper.word_count == 8


def test_paper_summary_to_markdown() -> None:
    summary = PaperSummary(
        title="Paper",
        executive_summary="Exec summary.",
        key_contributions=["c1", "c2"],
        section_summaries=[
            SectionSummary(
                section_title="Intro",
                level=1,
                summary="Intro summary.",
                key_points=["kp1"],
            )
        ],
    )
    md = summary.to_markdown()

    assert md.startswith("# Paper")
    assert "## Executive Summary" in md
    assert "Exec summary." in md
    assert "## Key Contributions" in md
    assert "- c1" in md
    assert "### Intro" in md
    assert "- kp1" in md
