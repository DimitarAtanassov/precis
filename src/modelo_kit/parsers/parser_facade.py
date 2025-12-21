from pathlib import Path
from typing import List, Optional, Type, Union

import httpx

from modelo_kit.parsers.parser_base import BaseParser
from modelo_kit.parsers.pdf_toc_parser import PDFTocParser
from modelo_kit.parsers.pdf_font_parser import PDFFontParser
from modelo_kit.models.paper_model import ParsedPaper, Section


def is_url(path: str) -> bool:
    """Check if a string is a URL."""
    return path.startswith(("http://", "https://"))


def normalize_arxiv_url(url: str) -> str:
    """Convert arXiv abstract URL to PDF URL if needed."""
    if "arxiv.org/abs/" in url:
        url = url.replace("/abs/", "/pdf/")
    if "arxiv.org/pdf/" in url and not url.endswith(".pdf"):
        url += ".pdf"
    return url


class PaperParser:
    """
    Facade for parsing research papers.
    
    Supports both local files and URLs (streams directly to memory).
    Tries TOC-based parsing first, falls back to font-based parsing.
    
    Usage:
        parser = PaperParser()
        
        # From local file
        paper = parser.parse("path/to/paper.pdf")
        
        # From URL (no download to disk)
        paper = parser.parse("https://arxiv.org/pdf/2407.21783")
    """

    # Parsers in priority order: TOC first, then font-based fallback
    _parsers: List[Type[BaseParser]] = [
        PDFTocParser,
        PDFFontParser,
    ]

    def __init__(self):
        self._parser_instances: List[BaseParser] = [
            parser_cls() for parser_cls in self._parsers
        ]
        self._active_parser: Optional[BaseParser] = None
        self._pdf_bytes: Optional[bytes] = None
        self._source_id: str = ""

    def _fetch_pdf_bytes(self, url: str) -> bytes:
        """Fetch PDF bytes from URL into memory."""
        url = normalize_arxiv_url(url)
        
        print(f"⬇️  Fetching PDF from: {url}")
        
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
        
        print(f"✅ Loaded {len(response.content) / 1024 / 1024:.1f} MB into memory")
        return response.content

    def _get_source(self, path_or_url: str) -> Union[Path, bytes]:
        """Resolve input to either a Path or bytes."""
        if is_url(path_or_url):
            self._source_id = path_or_url
            self._pdf_bytes = self._fetch_pdf_bytes(path_or_url)
            return self._pdf_bytes
        else:
            path = Path(path_or_url)
            self._source_id = str(path)
            self._pdf_bytes = None
            return path

    def _select_parser(self, source: Union[Path, bytes]) -> BaseParser:
        """Select the best parser for the given source."""
        for parser in self._parser_instances:
            if parser.can_parse(source):
                parser_name = parser.__class__.__name__
                print(f"📄 Using {parser_name}")
                return parser
        
        raise ValueError(
            f"No suitable parser found. "
            f"Supported formats: PDF with TOC or font-based sections."
        )

    def parse(
        self, 
        path_or_url: str, 
        load_content: bool = False
    ) -> ParsedPaper:
        """
        Parse a research paper into structured format.
        
        Args:
            path_or_url: Local file path or URL to PDF.
            load_content: If True, load all section content immediately.
        
        Returns:
            ParsedPaper object with structure and optionally content.
        """
        source = self._get_source(path_or_url)
        
        if isinstance(source, Path) and not source.exists():
            raise FileNotFoundError(f"File not found: {source}")
        
        self._active_parser = self._select_parser(source)
        paper = self._active_parser.parse(source, source_id=self._source_id)
        
        if load_content:
            self._load_all_content(paper)
        
        return paper

    def _load_all_content(self, paper: ParsedPaper) -> None:
        """Recursively load content for all sections."""
        source = self._pdf_bytes if self._pdf_bytes else Path(paper.source_path)
        
        def load_recursive(sections: List[Section]):
            for section in sections:
                if not section.content:
                    self._active_parser.load_section_content(source, section)
                if section.subsections:
                    load_recursive(section.subsections)
        
        load_recursive(paper.sections)

    def get_section_content(self, paper: ParsedPaper, section: Section) -> str:
        """Get content for a specific section."""
        if section.content:
            return section.content
        
        source = self._pdf_bytes if self._pdf_bytes else Path(paper.source_path)
        return self._active_parser.load_section_content(source, section)

    def get_toc_preview(self, path_or_url: str) -> List[str]:
        """Get a quick preview of the paper's structure."""
        source = self._get_source(path_or_url)
        parser = self._select_parser(source)
        toc = parser.extract_toc(source)
        
        return [
            "  " * (level - 1) + title 
            for level, title, _ in toc
        ]

    def print_structure(self, paper: ParsedPaper) -> None:
        """Print the paper structure to console."""
        print(f"\n{'='*60}")
        print(f"Title: {paper.title}")
        print(f"Pages: {paper.total_pages}")
        print(f"{'='*60}\n")
        
        if paper.abstract:
            print("Abstract:")
            print(f"  {paper.abstract[:200]}...")
            print()
        
        print("Structure:")
        for title in paper.get_all_section_titles():
            print(f"  {title}")