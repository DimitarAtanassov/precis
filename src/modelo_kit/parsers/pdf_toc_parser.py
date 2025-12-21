import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple, Optional, Union
import io

from modelo_kit.parsers.parser_base import BaseParser
from modelo_kit.models.paper_model import ParsedPaper, Section


class PDFTocParser(BaseParser):
    """
    Parser for PDFs that have an embedded Table of Contents (outline).
    
    Uses PyMuPDF (fitz) to extract the document outline and section content.
    Supports both file paths and in-memory bytes.
    """

    def _open_document(self, source: Union[Path, bytes]) -> fitz.Document:
        """Open a PDF from path or bytes."""
        if isinstance(source, bytes):
            return fitz.open(stream=source, filetype="pdf")
        return fitz.open(source)

    def can_parse(self, source: Union[Path, bytes]) -> bool:
        """Check if source is a PDF with an embedded TOC."""
        if isinstance(source, Path) and source.suffix.lower() != ".pdf":
            return False
        
        try:
            with self._open_document(source) as doc:
                toc = doc.get_toc()
                return len(toc) > 0
        except Exception:
            return False

    def extract_toc(self, source: Union[Path, bytes]) -> List[Tuple[int, str, int]]:
        """
        Extract TOC from PDF outline.
        
        Args:
            source: Path to PDF file or PDF bytes.
            
        Returns:
            List of (level, title, page_number) tuples.
        """
        with self._open_document(source) as doc:
            return doc.get_toc()

    def extract_section_content(
        self, 
        source: Union[Path, bytes], 
        start_page: int, 
        end_page: int
    ) -> str:
        """
        Extract text from specified page range.
        
        Args:
            source: Path to PDF file or PDF bytes.
            start_page: 1-indexed start page.
            end_page: 1-indexed end page (inclusive).
        """
        with self._open_document(source) as doc:
            text_parts = []
            # Convert to 0-indexed for fitz
            for page_num in range(start_page - 1, min(end_page, len(doc))):
                page = doc[page_num]
                text_parts.append(page.get_text("text"))
            return "\n".join(text_parts)

    def _extract_title_and_authors(self, doc: fitz.Document) -> Tuple[str, List[str]]:
        """Extract paper title and authors from first page."""
        if len(doc) == 0:
            return "", []
        
        first_page = doc[0]
        text = first_page.get_text("text")
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        title = lines[0] if lines else ""
        return title, []

    def _extract_abstract(self, doc: fitz.Document) -> str:
        """Extract abstract from the paper."""
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
                return abstract
        
        return ""

    def _build_section_tree(
        self, 
        toc: List[Tuple[int, str, int]], 
        total_pages: int
    ) -> List[Section]:
        """Build hierarchical section tree from flat TOC."""
        if not toc:
            return []

        toc_with_ends = []
        for i, (level, title, page) in enumerate(toc):
            if i + 1 < len(toc):
                end_page = toc[i + 1][2] - 1
            else:
                end_page = total_pages
            toc_with_ends.append((level, title, page, max(end_page, page)))

        def build_recursive(
            entries: List[Tuple[int, str, int, int]], 
            parent_level: int = 0
        ) -> Tuple[List[Section], int]:
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
                    content=""
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

    def parse(self, source: Union[Path, bytes], source_id: str = "") -> ParsedPaper:
        """
        Parse PDF into structured ParsedPaper object.
        
        Args:
            source: Path to the PDF file or PDF bytes.
            source_id: Identifier for the source (URL or path string).
            
        Returns:
            ParsedPaper with full structure and metadata.
        """
        if isinstance(source, Path):
            source_id = source_id or str(source)
        
        with self._open_document(source) as doc:
            toc = doc.get_toc()
            total_pages = len(doc)
            title, authors = self._extract_title_and_authors(doc)
            abstract = self._extract_abstract(doc)

        sections = self._build_section_tree(toc, total_pages)

        return ParsedPaper(
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            total_pages=total_pages,
            source_path=source_id
        )

    def load_section_content(
        self, 
        source: Union[Path, bytes], 
        section: Section
    ) -> str:
        """Lazily load content for a specific section."""
        content = self.extract_section_content(
            source,
            section.page_start,
            section.page_end or section.page_start
        )
        section.content = content
        return content