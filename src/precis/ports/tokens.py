"""Port for token-counting strategies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """Strategy for estimating the token length of text.

    Single canonical definition shared by the summarizer and any future
    chunking logic (previously duplicated in core.interfaces and the summarizer).
    """

    def count(self, text: str) -> int:
        """Return the estimated number of tokens in ``text``."""
        ...
