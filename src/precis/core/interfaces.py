"""
Protocols and abstract interfaces for dependency injection.

These interfaces define contracts that concrete implementations must follow,
enabling loose coupling and easier testing.

Note: Protocols are defined here for future extensibility but may not all
be actively used in the current implementation. They serve as documentation
for expected interfaces.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """
    Protocol for token counting strategies.

    Used by the summarizer to handle different tokenization approaches.
    """

    def count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens in the text.
        """
        ...
