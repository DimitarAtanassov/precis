"""
PDF parsing strategies using PyMuPDF.

Provides two strategies:
- TOC-based: Uses embedded table of contents (preferred)
- Font-based: Analyzes font sizes to detect headers (fallback)
"""

from collections import Counter
from pathlib import Path

import fitz  # type: ignore[import-untyped]

from modelo_kit.models import ParsedPaper, Section
from modelo_kit.parsers.base import BaseParser

# =============================================================================
# Shared Utilities
# =============================================================================


def open_document(source: Path | bytes) -> fitz.Document:
    """Open a PDF from path or bytes."""
    if isinstance(source, bytes):
        return fitz.open(stream=source, filetype="pdf")
    return fitz.open(source)


def extract_title_from_first_page(doc: fitz.Document) -> str:
    """Extract title from the first page."""
    if len(doc) == 0:
        return ""
    first_page = doc[0]
    text = first_page.get_text("text")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines[0] if lines else ""


def extract_abstract(doc: fitz.Document) -> str:
    """Extract abstract from the first few pages."""
    for page_num in range(min(2, len(doc))):
        text = doc[page_num].get_text("text")
        lower_text = text.lower()

        if "abstract" in lower_text:
            start_idx = lower_text.find("abstract")
            end_markers = ["introduction", "1.", "1 "]

            end_idx = len(text)
            for marker in end_markers:
                marker_idx = lower_text.find(marker, start_idx + 8)
                if marker_idx != -1 and marker_idx < end_idx:
                    end_idx = marker_idx

            abstract = text[start_idx:end_idx].strip()
            if abstract.lower().startswith("abstract"):
                abstract = abstract[8:].strip()
            return str(abstract)

    return ""


# =============================================================================
# TOC-Based Parser
# =============================================================================


class PDFTocParser(BaseParser):
    """
    Parser for PDFs with an embedded Table of Contents.

    This is the preferred strategy when available, as it provides
    accurate section boundaries defined by the document author.
    """

    def can_parse(self, source: Path | bytes) -> bool:
        """Check if PDF has an embedded TOC."""
        if isinstance(source, Path) and source.suffix.lower() != ".pdf":
            return False
        try:
            with open_document(source) as doc:
                toc = doc.get_toc()
                return len(toc) > 0
        except Exception:
            return False

    def extract_toc(self, source: Path | bytes) -> list[tuple[int, str, int]]:
        """Extract TOC as list of (level, title, page_number) tuples."""
        with open_document(source) as doc:
            toc: list[tuple[int, str, int]] = doc.get_toc()
            return toc

    def extract_section_content(
        self, source: Path | bytes, start_page: int, end_page: int
    ) -> str:
        """Extract text from page range (1-indexed)."""
        with open_document(source) as doc:
            text_parts = []
            for page_num in range(start_page - 1, min(end_page, len(doc))):
                page = doc[page_num]
                text_parts.append(page.get_text("text"))
            return "\n".join(text_parts)

    def parse(self, source: Path | bytes, source_id: str = "") -> ParsedPaper:
        """Parse PDF using embedded TOC."""
        if isinstance(source, Path):
            source_id = source_id or str(source)

        with open_document(source) as doc:
            toc = doc.get_toc()
            total_pages = len(doc)
            title = extract_title_from_first_page(doc)
            abstract = extract_abstract(doc)

        sections = self._build_section_tree(toc, total_pages)

        return ParsedPaper(
            title=title,
            authors=[],
            abstract=abstract,
            sections=sections,
            total_pages=total_pages,
            source_path=source_id,
        )

    def _build_section_tree(
        self, toc: list[tuple[int, str, int]], total_pages: int
    ) -> list[Section]:
        """Build hierarchical section tree from flat TOC."""
        if not toc:
            return []

        # Add end pages to each entry
        toc_with_ends = []
        for i, (level, title, page) in enumerate(toc):
            end_page = toc[i + 1][2] - 1 if i + 1 < len(toc) else total_pages
            toc_with_ends.append((level, title, page, max(end_page, page)))

        def build_recursive(
            entries: list[tuple[int, str, int, int]], parent_level: int = 0
        ) -> tuple[list[Section], int]:
            result = []
            i = 0

            while i < len(entries):
                level, title, start_page, end_page = entries[i]

                if level <= parent_level and parent_level > 0:
                    break

                section = Section(
                    title=title,
                    level=level,
                    page_start=start_page,
                    page_end=end_page,
                    content="",
                )
                i += 1

                if i < len(entries) and entries[i][0] > level:
                    subsections, consumed = build_recursive(entries[i:], level)
                    section.subsections = subsections
                    i += consumed

                result.append(section)

            return result, i

        sections, _ = build_recursive(toc_with_ends)
        return sections


# =============================================================================
# Font-Based Parser
# =============================================================================


class PDFFontParser(BaseParser):
    """
    Parser that detects sections by analyzing font sizes.

    Used as a fallback when no TOC is available. Identifies headers
    by finding text with larger-than-body font sizes.
    """

    # Font size thresholds
    MIN_HEADER_RATIO = 1.15  # Header must be 15% larger than body
    MIN_BODY_FONT = 8.0
    MAX_BODY_FONT = 14.0

    def can_parse(self, source: Path | bytes) -> bool:
        """Check if source is a PDF (always returns True for PDFs)."""
        if isinstance(source, Path) and source.suffix.lower() != ".pdf":
            return False
        try:
            with open_document(source) as doc:
                return len(doc) > 0
        except Exception:
            return False

    def extract_section_content(
        self, source: Path | bytes, start_page: int, end_page: int
    ) -> str:
        """Extract text from page range."""
        with open_document(source) as doc:
            text_parts = []
            for page_num in range(start_page - 1, min(end_page, len(doc))):
                page = doc[page_num]
                text_parts.append(page.get_text("text"))
            return "\n".join(text_parts)

    def parse(self, source: Path | bytes, source_id: str = "") -> ParsedPaper:
        """Parse PDF using font analysis."""
        if isinstance(source, Path):
            source_id = source_id or str(source)

        with open_document(source) as doc:
            total_pages = len(doc)
            title = extract_title_from_first_page(doc)
            abstract = extract_abstract(doc)

            # Extract text blocks with font info
            blocks = self._extract_blocks(doc)
            header_sizes, body_size = self._analyze_fonts(blocks)
            sections = self._build_sections(blocks, header_sizes, body_size)

        return ParsedPaper(
            title=title,
            authors=[],
            abstract=abstract,
            sections=sections,
            total_pages=total_pages,
            source_path=source_id,
        )

    def _extract_blocks(self, doc: fitz.Document) -> list[dict[str, str | float | int]]:
        """Extract text blocks with font information."""
        blocks = []
        for page_num, page in enumerate(doc):
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:  # Text block
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            blocks.append(
                                {
                                    "text": text,
                                    "size": span.get("size", 12.0),
                                    "page": page_num + 1,
                                    "flags": span.get("flags", 0),
                                }
                            )
        return blocks

    def _analyze_fonts(
        self, blocks: list[dict[str, str | float | int]]
    ) -> tuple[set[float], float]:
        """Analyze font sizes to identify headers vs body text."""
        if not blocks:
            return set(), 12.0

        # Count font sizes weighted by text length
        size_counts: Counter[float] = Counter()
        for block in blocks:
            text = str(block["text"])
            size = float(block["size"])
            if self.MIN_BODY_FONT <= size <= self.MAX_BODY_FONT:
                size_counts[size] += len(text)

        if not size_counts:
            return set(), 12.0

        # Body size is the most common
        body_size = size_counts.most_common(1)[0][0]

        # Headers are significantly larger than body
        header_sizes = {
            float(block["size"])
            for block in blocks
            if float(block["size"]) > body_size * self.MIN_HEADER_RATIO
        }

        return header_sizes, body_size

    def _build_sections(
        self,
        blocks: list[dict[str, str | float | int]],
        header_sizes: set[float],
        body_size: float,
    ) -> list[Section]:
        """Build sections from analyzed blocks."""
        if not blocks:
            return []

        sections: list[Section] = []
        current_section: Section | None = None
        content_parts: list[str] = []

        for block in blocks:
            text = str(block["text"])
            size = float(block["size"])
            page = int(block["page"])

            is_header = size in header_sizes and len(text) < 100

            if is_header:
                # Save previous section
                if current_section is not None:
                    current_section.content = "\n".join(content_parts)
                    sections.append(current_section)
                    content_parts = []

                # Determine level based on font size
                sorted_headers = sorted(header_sizes, reverse=True)
                level = sorted_headers.index(size) + 1 if size in sorted_headers else 1

                current_section = Section(
                    title=text,
                    level=level,
                    page_start=page,
                    page_end=page,
                    content="",
                )
            elif current_section is not None:
                content_parts.append(text)
                current_section.page_end = page

        # Save last section
        if current_section is not None:
            current_section.content = "\n".join(content_parts)
            sections.append(current_section)

        return sections
