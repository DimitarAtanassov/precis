import requests
from abc import ABC, abstractmethod
from typing import Dict, Optional
from pdfminer.high_level import extract_text

class PaperService(ABC):
    @abstractmethod
    def get_paper(self, paper_id: str) -> Dict[str, Optional[str]]:
        """
        Returns a dictionary with keys: title, authors, abstract, content, url.
        """
        pass

class ArxivPaperService(PaperService):
    def get_paper(self, paper_id: str) -> Dict[str, Optional[str]]:
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        abs_url = f"https://arxiv.org/abs/{paper_id}"

        # Download PDF
        response = requests.get(pdf_url)
        response.raise_for_status()
        with open(f"/tmp/{paper_id}.pdf", "wb") as f:
            f.write(response.content)

        # Extract text from PDF
        content = extract_text(f"/tmp/{paper_id}.pdf")

        # Optionally, fetch metadata from arXiv API (for title, authors, abstract)
        # For now, just return the PDF content
        return {
            "title": None,      # Can be parsed from PDF or arXiv API
            "authors": None,    # Can be parsed from arXiv API
            "abstract": None,   # Can be parsed from arXiv API
            "content": content,
            "url": abs_url,
        }