"""
Parsers for various document formats.

- PaperParser: High-level facade for research papers (PDF)
- ObsidianParser: Obsidian markdown notes
- PDFTocParser: PDF parsing using embedded TOC
- PDFFontParser: PDF parsing using font analysis
"""

from precis.parsers.base import BaseParser
from precis.parsers.markdown import ObsidianParser
from precis.parsers.paper import PaperParser
from precis.parsers.pdf import PDFFontParser, PDFTocParser

__all__ = [
    "PaperParser",
    "BaseParser",
    "PDFTocParser",
    "PDFFontParser",
    "ObsidianParser",
]
