from pydantic import BaseModel


class SectionSummary(BaseModel):
    """Summary of a single section."""

    section_title: str
    level: int
    summary: str
    key_points: list[str] = []
    page_range: str = ""


class PaperSummary(BaseModel):
    """Complete paper summary with hierarchical structure."""

    title: str
    abstract: str = ""
    executive_summary: str = ""
    section_summaries: list[SectionSummary] = []
    key_contributions: list[str] = []
    methodology_summary: str = ""
    results_summary: str = ""
    limitations: list[str] = []
    future_work: list[str] = []

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
            lines.extend(
                [
                    "## Key Contributions",
                    *[f"- {c}" for c in self.key_contributions],
                    "",
                ]
            )

        if self.methodology_summary:
            lines.extend(
                [
                    "## Methodology",
                    self.methodology_summary,
                    "",
                ]
            )

        if self.results_summary:
            lines.extend(
                [
                    "## Results",
                    self.results_summary,
                    "",
                ]
            )

        if self.section_summaries:
            lines.extend(["## Section Summaries", ""])
            for sec in self.section_summaries:
                indent = "  " * (sec.level - 1)
                lines.append(f"{indent}### {sec.section_title}")
                lines.append(f"{indent}{sec.summary}")
                if sec.key_points:
                    lines.extend([f"{indent}- {point}" for point in sec.key_points])
                lines.append("")

        if self.limitations:
            lines.extend(
                [
                    "## Limitations",
                    *[f"- {lim}" for lim in self.limitations],
                ]
            )

        if self.future_work:
            lines.extend(
                [
                    "## Future Work",
                    *[f"- {f}" for f in self.future_work],
                ]
            )

        return "\n".join(lines)
