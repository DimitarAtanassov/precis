"""
Parsers for various document formats.

- PaperParser: High-level facade for research papers (PDF)
- ObsidianParser: Obsidian markdown notes
- PDFTocParser: PDF parsing using embedded TOC
- PDFFontParser: PDF parsing using font analysis
"""

from modelo_kit.parsers.base import BaseParser
from modelo_kit.parsers.markdown import ObsidianParser
from modelo_kit.parsers.paper import PaperParser
from modelo_kit.parsers.pdf import PDFFontParser, PDFTocParser

__all__ = [
    "PaperParser",
    "BaseParser",
    "PDFTocParser",
    "PDFFontParser",
    "ObsidianParser",
]
