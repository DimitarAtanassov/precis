"""
High-level facade for parsing research papers.

Handles PDFs (local and remote) and orchestrates parsing strategies.
"""

from pathlib import Path

import httpx

from modelo_kit.models import ParsedPaper, Section
from modelo_kit.parsers.base import BaseParser
from modelo_kit.parsers.pdf import PDFFontParser, PDFTocParser


class PaperParser:
    """
    Facade for parsing research papers from various sources.

    Automatically selects the best parsing strategy:
    1. TOC-based parsing (if PDF has embedded outline)
    2. Font-based parsing (fallback)

    Supports:
    - Local PDF files
    - Remote PDFs via URL
    - In-memory PDF bytes

    Usage:
        parser = PaperParser()
        paper = parser.parse("/path/to/paper.pdf")
        paper = parser.parse("https://arxiv.org/pdf/1234.5678.pdf")
    """

    def __init__(self) -> None:
        self._parsers: list[BaseParser] = [
            PDFTocParser(),  # Preferred: uses embedded TOC
            PDFFontParser(),  # Fallback: analyzes fonts
        ]
        self._active_parser: BaseParser | None = None
        self._pdf_bytes: bytes | None = None

    def parse(self, path_or_url: str, load_content: bool = False) -> ParsedPaper:
        """
        Parse a research paper into structured format.

        Args:
            path_or_url: Local file path or URL to PDF.
            load_content: Whether to eagerly load all section content.

        Returns:
            Parsed paper structure.
        """
        source, source_id = self._resolve_source(path_or_url)
        self._active_parser = self._select_parser(source)

        if self._active_parser is None:
            raise ValueError(f"No parser available for: {path_or_url}")

        paper = self._active_parser.parse(source, source_id)

        if load_content:
            self._load_all_content(paper)

        return paper

    def get_section_content(self, paper: ParsedPaper, section: Section) -> str:
        """Get content for a section, loading if necessary."""
        source = self._pdf_bytes if self._pdf_bytes else Path(paper.source_path)
        if self._active_parser and not section.content:
            return self._active_parser.load_section_content(source, section)
        return section.content

    def print_structure(self, paper: ParsedPaper) -> None:
        """Print the paper structure to console."""
        print(f"\n{'=' * 60}")
        print(f"Title: {paper.title}")
        print(f"Source: {paper.source_path}")
        print(f"Total Pages: {paper.total_pages}")
        print(f"{'=' * 60}")

        if paper.abstract:
            print("\n📄 Abstract found")

        print("\n📑 Sections:")
        self._print_sections(paper.sections)

    def _resolve_source(self, path_or_url: str) -> tuple[Path | bytes, str]:
        """Resolve input to a parseable source."""
        if path_or_url.startswith(("http://", "https://")):
            return self._download_pdf(path_or_url), path_or_url

        path = Path(path_or_url)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path, str(path)

    def _download_pdf(self, url: str) -> bytes:
        """Download PDF from URL."""
        print(f"📥 Downloading: {url}")
        response = httpx.get(url, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
        self._pdf_bytes = response.content
        return self._pdf_bytes

    def _select_parser(self, source: Path | bytes) -> BaseParser | None:
        """Select the best parser for the source."""
        for parser in self._parsers:
            if parser.can_parse(source):
                return parser
        return None

    def _load_all_content(self, paper: ParsedPaper) -> None:
        """Recursively load content for all sections."""
        source = self._pdf_bytes if self._pdf_bytes else Path(paper.source_path)

        def load_recursive(sections: list[Section]) -> None:
            for section in sections:
                if not section.content and self._active_parser:
                    self._active_parser.load_section_content(source, section)
                if section.subsections:
                    load_recursive(section.subsections)

        load_recursive(paper.sections)

    def _print_sections(self, sections: list[Section], indent: int = 0) -> None:
        """Recursively print section structure."""
        for section in sections:
            prefix = "  " * indent
            page_info = f" (p.{section.page_start}-{section.page_end})"
            print(f"{prefix}• {section.title}{page_info}")
            if section.subsections:
                self._print_sections(section.subsections, indent + 1)
