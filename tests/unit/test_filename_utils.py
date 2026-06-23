"""Characterization tests for filename utilities.

These pin the current sanitization/naming behavior so the Phase 2 move into
adapters/sink does not silently change output filenames.
"""

import pytest

from precis.core.filename_utils import (
    get_model_suffix,
    get_name_from_source,
    sanitize_filename,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("My Paper: A Study (2024)", "My_Paper_A_Study_2024"),
        ("a//b\\c", "a_b_c"),
        ("__leading_trailing__", "leading_trailing"),
        ("keep-dot.name", "keep-dot.name"),
        ("weird***chars???", "weird_chars"),
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_get_model_suffix() -> None:
    assert get_model_suffix("openai", "gpt-4o") == "_openai_gpt-4o"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("/path/to/my_paper.pdf", "my_paper"),
        ("https://arxiv.org/pdf/1234.5678.pdf", "1234.5678"),
        ("https://example.com/articles/great-post", "great-post"),
        ("relative/file.md", "file"),
    ],
)
def test_get_name_from_source(source: str, expected: str) -> None:
    assert get_name_from_source(source) == expected
