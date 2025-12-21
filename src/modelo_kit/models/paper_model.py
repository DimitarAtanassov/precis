from pydantic import BaseModel


class Section(BaseModel):
    """Represents a section within a research paper."""

    title: str
    level: int  # Hierarchy level (1 = top-level, 2 = subsection, etc.)
    page_start: int
    page_end: int | None = None
    content: str = ""
    subsections: list["Section"] = []

    class Config:
        # Allow recursive model
        arbitrary_types_allowed = True


class ParsedPaper(BaseModel):
    """Represents a fully parsed research paper."""

    title: str
    authors: list[str] = []
    abstract: str = ""
    sections: list[Section] = []
    total_pages: int = 0
    source_path: str = ""

    def get_section_by_title(self, title: str) -> Section | None:
        """Find a section by its title (case-insensitive partial match)."""
        title_lower = title.lower()

        def search(sections: list[Section]) -> Section | None:
            for section in sections:
                if title_lower in section.title.lower():
                    return section
                if section.subsections:
                    result = search(section.subsections)
                    if result:
                        return result
            return None

        return search(self.sections)

    def get_all_section_titles(self) -> list[str]:
        """Returns a flat list of all section titles with indentation."""
        titles = []

        def collect(sections: list[Section], indent: int = 0) -> None:
            for section in sections:
                prefix = "  " * indent
                titles.append(f"{prefix}{section.title}")
                if section.subsections:
                    collect(section.subsections, indent + 1)

        collect(self.sections)
        return titles


# Required for recursive model
Section.model_rebuild()
