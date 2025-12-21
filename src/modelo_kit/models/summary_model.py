from typing import List, Optional
from pydantic import BaseModel


class SectionSummary(BaseModel):
    """Summary of a single section."""
    section_title: str
    level: int
    summary: str
    key_points: List[str] = []
    page_range: str = ""


class PaperSummary(BaseModel):
    """Complete paper summary with hierarchical structure."""
    title: str
    abstract: str = ""
    executive_summary: str = ""
    section_summaries: List[SectionSummary] = []
    key_contributions: List[str] = []
    methodology_summary: str = ""
    results_summary: str = ""
    limitations: List[str] = []
    future_work: List[str] = []
    
    def to_markdown(self) -> str:
        """Export summary as formatted markdown."""
        lines = [
            f"# {self.title}",
            "",
            "## Executive Summary",
            self.executive_summary,
            "",
        ]
        
        if self.key_contributions:
            lines.extend([
                "## Key Contributions",
                *[f"- {c}" for c in self.key_contributions],
                "",
            ])
        
        if self.methodology_summary:
            lines.extend([
                "## Methodology",
                self.methodology_summary,
                "",
            ])
        
        if self.results_summary:
            lines.extend([
                "## Results",
                self.results_summary,
                "",
            ])
        
        if self.section_summaries:
            lines.extend(["## Section Summaries", ""])
            for sec in self.section_summaries:
                indent = "  " * (sec.level - 1)
                lines.append(f"{indent}### {sec.section_title}")
                lines.append(f"{indent}{sec.summary}")
                if sec.key_points:
                    for point in sec.key_points:
                        lines.append(f"{indent}- {point}")
                lines.append("")
        
        if self.limitations:
            lines.extend([
                "## Limitations",
                *[f"- {l}" for l in self.limitations],
                "",
            ])
        
        if self.future_work:
            lines.extend([
                "## Future Work",
                *[f"- {f}" for f in self.future_work],
            ])
        
        return "\n".join(lines)