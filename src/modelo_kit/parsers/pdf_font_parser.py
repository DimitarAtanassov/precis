"""
PDF parser that identifies sections by font size.
Section headers are typically larger than body text.
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import Counter

from modelo_kit.parsers.parser_base import BaseParser
from modelo_kit.models.paper_model import ParsedPaper, Section


@dataclass
class TextBlock:
    """A block of text with its font properties."""
    text: str
    font_size: float
    page_num: int
    y_position: float  # Vertical position on page


class PDFFontParser(BaseParser):
    """
    Parser for PDFs without TOC.
    Identifies section headers by detecting larger font sizes.
    """

    def _open_document(self, source: Union[Path, bytes]) -> fitz.Document:
        """Open a PDF from path or bytes."""
        if isinstance(source, bytes):
            return fitz.open(stream=source, filetype="pdf")
        return fitz.open(source)

    def can_parse(self, source: Union[Path, bytes]) -> bool:
        """This parser can handle any valid PDF."""
        if isinstance(source, Path) and source.suffix.lower() != ".pdf":
            return False
        try:
            with self._open_document(source) as doc:
                return len(doc) > 0
        except Exception:
            return False

    def _extract_text_blocks(self, doc: fitz.Document) -> List[TextBlock]:
        """Extract all text blocks with font size info."""
        blocks = []
        
        for page_num, page in enumerate(doc, start=1):
            # Get detailed text info with font sizes
            text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:  # Skip non-text blocks
                    continue
                
                for line in block.get("lines", []):
                    line_text = ""
                    max_font_size = 0
                    
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                        max_font_size = max(max_font_size, span.get("size", 0))
                    
                    line_text = line_text.strip()
                    if line_text and max_font_size > 0:
                        blocks.append(TextBlock(
                            text=line_text,
                            font_size=round(max_font_size, 1),
                            page_num=page_num,
                            y_position=line.get("bbox", [0, 0, 0, 0])[1]
                        ))
        
        return blocks

    def _detect_body_font_size(self, blocks: List[TextBlock]) -> float:
        """Detect the most common font size (body text)."""
        if not blocks:
            return 10.0
        
        # Count font sizes, weighted by text length
        size_counts = Counter()
        for block in blocks:
            # Weight by text length to favor body text
            size_counts[block.font_size] += len(block.text)
        
        # Most common size is likely body text
        return size_counts.most_common(1)[0][0]

    def _detect_header_sizes(
        self, 
        blocks: List[TextBlock], 
        body_size: float
    ) -> List[float]:
        """Find font sizes larger than body text (potential headers)."""
        header_sizes = set()
        
        for block in blocks:
            # Headers are typically 1.2x+ larger than body
            if block.font_size > body_size * 1.15:
                header_sizes.add(block.font_size)
        
        # Sort descending (largest = level 1)
        return sorted(header_sizes, reverse=True)

    def _is_section_header(
        self, 
        block: TextBlock, 
        header_sizes: List[float],
        body_size: float
    ) -> bool:
        """Check if a text block is likely a section header."""
        if block.font_size not in header_sizes:
            return False
        
        text = block.text.strip()
        
        # Skip very long lines (likely body text)
        if len(text) > 100:
            return False
        
        # Skip lines that are just numbers or punctuation
        if not any(c.isalpha() for c in text):
            return False
        
        return True

    def _get_header_level(
        self, 
        font_size: float, 
        header_sizes: List[float]
    ) -> int:
        """Map font size to header level (1 = largest)."""
        try:
            return header_sizes.index(font_size) + 1
        except ValueError:
            return 1

    def _build_sections(
        self, 
        blocks: List[TextBlock],
        header_sizes: List[float],
        body_size: float
    ) -> List[Section]:
        """Build sections from text blocks based on font sizes."""
        sections = []
        current_section: Optional[Section] = None
        current_content: List[str] = []
        
        for block in blocks:
            if self._is_section_header(block, header_sizes, body_size):
                # Save previous section
                if current_section:
                    current_section.content = "\n".join(current_content).strip()
                    sections.append(current_section)
                
                # Start new section
                level = self._get_header_level(block.font_size, header_sizes)
                current_section = Section(
                    title=block.text.strip(),
                    level=level,
                    page_start=block.page_num,
                    page_end=block.page_num,
                    content="",
                    subsections=[]
                )
                current_content = []
            else:
                # Add to current section content
                if current_section:
                    current_content.append(block.text)
                    current_section.page_end = block.page_num
        
        # Don't forget last section
        if current_section:
            current_section.content = "\n".join(current_content).strip()
            sections.append(current_section)
        
        return sections

    def _extract_title(self, blocks: List[TextBlock]) -> str:
        """Extract paper title (usually largest text on first page)."""
        first_page_blocks = [b for b in blocks if b.page_num == 1]
        
        if not first_page_blocks:
            return "Untitled"
        
        # Find largest font on first page
        max_size = max(b.font_size for b in first_page_blocks)
        
        # Get all text with that font size (title might span lines)
        title_parts = []
        for block in first_page_blocks:
            if block.font_size == max_size:
                title_parts.append(block.text)
            elif title_parts:  # Stop after title ends
                break
        
        title = " ".join(title_parts).strip()
        
        # Clean up common artifacts
        if len(title) > 200:
            title = title[:200] + "..."
        
        return title or "Untitled"

    def _extract_abstract(self, blocks: List[TextBlock]) -> str:
        """Extract abstract from the paper."""
        abstract_started = False
        abstract_parts = []
        
        for block in blocks:
            if block.page_num > 2:  # Abstract is usually on first 2 pages
                break
            
            text_lower = block.text.lower().strip()
            
            if "abstract" in text_lower and len(text_lower) < 20:
                abstract_started = True
                continue
            
            if abstract_started:
                # Stop at introduction or numbered section
                if any(marker in text_lower for marker in ["introduction", "1 introduction", "1. introduction"]):
                    break
                abstract_parts.append(block.text)
        
        return " ".join(abstract_parts).strip()

    def extract_toc(self, source: Union[Path, bytes]) -> List[Tuple[int, str, int]]:
        """Extract TOC by analyzing font sizes."""
        with self._open_document(source) as doc:
            blocks = self._extract_text_blocks(doc)
        
        body_size = self._detect_body_font_size(blocks)
        header_sizes = self._detect_header_sizes(blocks, body_size)
        
        toc = []
        for block in blocks:
            if self._is_section_header(block, header_sizes, body_size):
                level = self._get_header_level(block.font_size, header_sizes)
                toc.append((level, block.text.strip(), block.page_num))
        
        return toc

    def extract_section_content(
        self, 
        source: Union[Path, bytes], 
        start_page: int, 
        end_page: int
    ) -> str:
        """Extract text from page range."""
        with self._open_document(source) as doc:
            text_parts = []
            for page_num in range(start_page - 1, min(end_page, len(doc))):
                text_parts.append(doc[page_num].get_text("text"))
            return "\n".join(text_parts)

    def parse(self, source: Union[Path, bytes], source_id: str = "") -> ParsedPaper:
        """Parse PDF into structured format using font analysis."""
        if isinstance(source, Path):
            source_id = source_id or str(source)
        
        with self._open_document(source) as doc:
            total_pages = len(doc)
            blocks = self._extract_text_blocks(doc)
        
        body_size = self._detect_body_font_size(blocks)
        header_sizes = self._detect_header_sizes(blocks, body_size)
        
        title = self._extract_title(blocks)
        abstract = self._extract_abstract(blocks)
        sections = self._build_sections(blocks, header_sizes, body_size)
        
        return ParsedPaper(
            title=title,
            authors=[],
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
        """Section content is already loaded during parse."""
        return section.content