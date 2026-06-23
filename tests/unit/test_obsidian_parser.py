"""Characterization tests for the Obsidian markdown parser."""

from pathlib import Path

from precis.parsers.markdown import ObsidianParser

NOTE = """\
---
tags: [research, ml]
title: My Note
---

This is the body with a #inline-tag and #nested/tag.

A link to [[Other Note]] and an aliased [[Target|Alias]] link.
An embed ![[Image]] and a heading link [[Doc#Section]].
"""


def test_parse_content_extracts_frontmatter_tags_and_links() -> None:
    parser = ObsidianParser()
    note = parser.parse_content(NOTE, title="My Note")

    # Frontmatter preserved.
    assert note.frontmatter["title"] == "My Note"

    # Tags merged from frontmatter and body, sorted, de-duplicated.
    assert note.tags == sorted({"research", "ml", "inline-tag", "nested/tag"})

    # Links: targets, alias, embed flag, and heading stripping.
    targets = {link.target for link in note.links}
    assert {"Other Note", "Target", "Image", "Doc"} == targets

    aliased = next(link for link in note.links if link.target == "Target")
    assert aliased.alias == "Alias"

    embed = next(link for link in note.links if link.target == "Image")
    assert embed.is_embed is True


def test_parse_reads_file(tmp_path: Path) -> None:
    note_path = tmp_path / "Sample.md"
    note_path.write_text(NOTE, encoding="utf-8")

    note = ObsidianParser().parse(note_path)

    assert note.title == "Sample"
    assert note.path == note_path
    assert note.word_count > 0


def test_no_frontmatter_returns_empty_dict() -> None:
    note = ObsidianParser().parse_content("Just body, no frontmatter.")
    assert note.frontmatter == {}
    assert note.body == "Just body, no frontmatter."
