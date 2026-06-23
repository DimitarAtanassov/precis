"""
Base classes and protocols for parsers.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from precis.models import ParsedPaper, Section


class BaseParser(ABC):
    """
    Abstract base class for document parsers.

    Defines the interface that all parsers must implement.
    """

    @abstractmethod
    def can_parse(self, source: Path | bytes) -> bool:
        """
        Check if this parser can handle the given source.

        Args:
            source: File path or bytes to check.

        Returns:
            True if this parser can handle the source.
        """
        ...

    @abstractmethod
    def parse(self, source: Path | bytes, source_id: str = "") -> ParsedPaper:
        """
        Parse a document into a structured ParsedPaper object.

        Args:
            source: File path or bytes to parse.
            source_id: Optional identifier for the source.

        Returns:
            Parsed document structure.
        """
        ...

    @abstractmethod
    def extract_section_content(
        self, source: Path | bytes, start_page: int, end_page: int
    ) -> str:
        """
        Extract text content from a page range.

        Args:
            source: File path or bytes.
            start_page: 1-indexed start page.
            end_page: 1-indexed end page (inclusive).

        Returns:
            Extracted text content.
        """
        ...

    def load_section_content(self, source: Path | bytes, section: Section) -> str:
        """
        Load content for a specific section.

        Args:
            source: File path or bytes.
            section: Section to load content for.

        Returns:
            Section content text.
        """
        content = self.extract_section_content(
            source, section.page_start, section.page_end or section.page_start
        )
        section.content = content
        return content
