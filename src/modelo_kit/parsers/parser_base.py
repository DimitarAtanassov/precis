from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Union

from modelo_kit.models.paper_model import ParsedPaper, Section


class BaseParser(ABC):
    """
    Abstract base class for paper parsers (Strategy Pattern).
    
    Supports both file paths and in-memory bytes.
    """

    @abstractmethod
    def can_parse(self, source: Union[Path, bytes]) -> bool:
        """
        Check if this parser can handle the given source.
        
        Args:
            source: Path to the document file or in-memory bytes.
            
        Returns:
            True if this parser can process the source, False otherwise.
        """
        pass

    @abstractmethod
    def extract_toc(self, source: Union[Path, bytes]) -> List[Tuple[int, str, int]]:
        """
        Extract the table of contents from the document.
        
        Args:
            source: Path to the document file or in-memory bytes.
            
        Returns:
            List of tuples: (level, title, page_number)
        """
        pass

    @abstractmethod
    def extract_section_content(
        self, 
        source: Union[Path, bytes], 
        start_page: int, 
        end_page: int
    ) -> str:
        """
        Extract text content between specified pages.
        
        Args:
            source: Path to the document file or in-memory bytes.
            start_page: Starting page number (1-indexed).
            end_page: Ending page number (1-indexed, inclusive).
            
        Returns:
            Extracted text content.
        """
        pass

    @abstractmethod
    def parse(self, source: Union[Path, bytes], source_id: str = "") -> ParsedPaper:
        """
        Fully parse a document into a structured ParsedPaper object.
        
        Args:
            source: Path to the document file or in-memory bytes.
            source_id: Optional identifier for the source.
            
        Returns:
            ParsedPaper object with sections and content.
        """
        pass