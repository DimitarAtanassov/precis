"""
CLI Handlers for different content modes.

Each handler is responsible for orchestrating the user workflow
for a specific content type, keeping business logic separate from presentation.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from precis.cli.menu import OBSIDIAN_MENU
from precis.cli.prompts import (
    confirm,
    print_error,
    print_header,
    print_section,
    prompt_input,
    select_llm,
)
from precis.core.filename_utils import get_name_from_source
from precis.core.output import OutputWriter, ParsedPaperData
from precis.models import ObsidianNote, Paper, PaperSummary, Section
from precis.parsers import PaperParser
from precis.services.llm_service import LLMRequest, LLMService
from precis.services.obsidian_vault import ObsidianVault
from precis.services.summarizer_service import PaperSummarizer, SummarizerConfig
from precis.services.web_service import WebService


class BaseHandler(ABC):
    """Base class for content handlers."""

    def __init__(self) -> None:
        self._llm_service = LLMService()
        self._output = OutputWriter()

    @abstractmethod
    def run(self) -> None:
        """Execute the handler workflow."""
        ...

    def _configure_llm(self) -> None:
        """Configure the LLM via user selection."""
        provider, model = select_llm()
        self._llm_service.configure(provider, model)


class PaperHandler(BaseHandler):
    """Handler for research paper parsing and summarization."""

    def run(self) -> None:
        """Execute paper workflow."""
        print_header("📄 Research Paper Mode", "-", 40)

        path_or_url = prompt_input("Enter PDF path or URL")
        if not path_or_url:
            return

        paper = self._parse_paper(path_or_url)
        self._print_structure(paper)

        if not confirm("\nGenerate LLM summary?"):
            self._save_parsed(paper, path_or_url)
            return

        self._configure_llm()
        summary = self._summarize(paper)

        self._save_summary(summary, path_or_url)
        self._print_executive_summary(summary)
        self._save_parsed(paper, path_or_url)

    def _parse_paper(self, path_or_url: str) -> Paper:
        """Parse the paper."""
        print("\n🔍 Parsing paper...")
        parser = PaperParser()
        return parser.parse(path_or_url, load_content=True)

    def _print_structure(self, paper: Paper) -> None:
        """Print paper structure."""
        parser = PaperParser()
        parser.print_structure(paper)

    def _summarize(self, paper: Paper) -> PaperSummary:
        """Generate summary using LLM."""
        llm = self._llm_service.get_llm()
        model = self._llm_service.model

        config = SummarizerConfig.for_model(model, verbose=True)
        print(f"\n🤖 Generating summary with {model}...")
        if config.max_chunk_tokens:
            print(f"   Context window: {config.max_chunk_tokens * 3:,} tokens")
            print(f"   Max chunk size: {config.max_chunk_tokens:,} tokens")

        summarizer = PaperSummarizer(llm, config)
        return summarizer.summarize(paper)

    def _save_summary(self, summary: PaperSummary, path_or_url: str) -> None:
        """Save summary to file."""
        name = get_name_from_source(path_or_url)
        self._output.save_summary(
            name=name,
            title=summary.title,
            content=summary.to_markdown(),
            provider=self._llm_service.provider,
            model=self._llm_service.model,
        )

    def _print_executive_summary(self, summary: PaperSummary) -> None:
        """Print executive summary."""
        print_section("Executive Summary")
        print(summary.executive_summary)

        if summary.key_contributions:
            print("\nKEY CONTRIBUTIONS:")
            for i, contribution in enumerate(summary.key_contributions, 1):
                print(f"  {i}. {contribution}")

    def _save_parsed(self, paper: Paper, path_or_url: str) -> None:
        """Save parsed paper content."""
        name = get_name_from_source(path_or_url)
        sections_content = self._format_sections(paper.sections)

        paper_data = ParsedPaperData(
            name=name,
            title=paper.title,
            source=paper.source_path,
            pages=paper.total_pages,
            abstract=paper.abstract,
            sections_content=sections_content,
        )
        self._output.save_parsed_paper(paper_data)

    def _format_sections(self, sections: list[Section], indent: int = 0) -> str:
        """Format sections as text."""
        lines: list[str] = []

        for section in sections:
            prefix = "  " * indent
            lines.append("\n" + "=" * 80)
            lines.append(f"{prefix}SECTION: {section.title}")
            lines.append(f"{prefix}Pages: {section.page_start} - {section.page_end}")
            lines.append("-" * 80 + "\n")

            if section.content:
                lines.append(section.content)

            if section.subsections:
                lines.append(self._format_sections(section.subsections, indent + 1))

        return "\n".join(lines)


class ObsidianHandler(BaseHandler):
    """Handler for Obsidian vault operations."""

    def run(self) -> None:
        """Execute Obsidian workflow."""
        print_header("📓 Obsidian Vault Mode", "-", 40)

        vault_path = prompt_input("Enter vault path")
        if not vault_path:
            return

        path = Path(vault_path).expanduser()
        if not path.exists():
            print_error(f"Vault not found: {path}")
            return

        vault = ObsidianVault(path)
        self._print_stats(vault)

        action = OBSIDIAN_MENU.display()

        actions = {
            "1": lambda: self._summarize_note(vault),
            "2": lambda: self._summarize_folder(vault),
            "3": lambda: self._list_notes(vault),
        }

        if handler := actions.get(action):
            handler()

    def _print_stats(self, vault: ObsidianVault) -> None:
        """Print vault statistics."""
        stats = vault.get_stats()
        print("\n📊 Vault Statistics:")
        print(f"   Notes: {stats.total_notes}")
        print(f"   Words: {stats.total_words:,}")
        print(f"   Links: {stats.total_links}")
        print(f"   Tags:  {len(stats.unique_tags)}")

        if stats.orphan_notes:
            print(f"   Orphans: {len(stats.orphan_notes)}")

    def _summarize_note(self, vault: ObsidianVault) -> None:
        """Summarize a single note."""
        note_name = prompt_input("\nEnter note name (without .md)")
        if not note_name:
            return

        note = vault.get_note(note_name)
        if not note:
            print_error(f"Note not found: {note_name}")
            return

        self._print_note_info(note)

        if not confirm("\nGenerate summary?"):
            return

        self._configure_llm()

        print(f"\n🤖 Summarizing with {self._llm_service.model}...")
        response = self._llm_service.generate(
            LLMRequest(
                prompt_key="obsidian_note_summary",
                variables={
                    "title": note.title,
                    "tags": ", ".join(note.tags) if note.tags else "none",
                    "word_count": str(note.word_count),
                    "content": note.body,
                },
            )
        )

        print_section("Summary")
        print(response)

        self._output.save_summary(
            note.title,
            f"Summary: {note.title}",
            response,
            provider=self._llm_service.provider,
            model=self._llm_service.model,
        )

    def _summarize_folder(self, vault: ObsidianVault) -> None:
        """Summarize all notes in a specific folder."""
        # List available folders
        self._list_folders(vault)

        folder_path = prompt_input("\nEnter folder path (relative to vault root)")
        if not folder_path:
            return

        notes = vault.notes_in_folder(folder_path)
        if not notes:
            print_error(f"No notes found in folder: {folder_path}")
            return

        total_words = sum(note.word_count for note in notes)
        all_tags: set[str] = set()
        for note in notes:
            all_tags.update(note.tags)

        print(f"\n📂 Folder: {folder_path}")
        print(f"   Notes: {len(notes)}")
        print(f"   Words: {total_words:,}")
        print(f"   Tags:  {len(all_tags)}")

        print(f"\n⚠️  This will summarize {len(notes)} notes in '{folder_path}'.")
        print("   This may take a while and use many API tokens.")

        if not confirm("\nContinue?"):
            return

        self._configure_llm()

        notes_overview = "\n".join(
            f"- **{note.title}**: {note.word_count} words, "
            f"tags: {', '.join(note.tags) or 'none'}"
            for note in notes
        )

        print(f"\n🤖 Creating folder summary with {self._llm_service.model}...")
        response = self._llm_service.generate(
            LLMRequest(
                prompt_key="obsidian_folder_summary",
                variables={
                    "folder_name": folder_path,
                    "total_notes": str(len(notes)),
                    "total_words": str(total_words),
                    "tag_count": str(len(all_tags)),
                    "notes_overview": notes_overview,
                },
            )
        )

        print_section(f"Folder Summary: {folder_path}")
        print(response)

        # Sanitize folder name for filename
        folder_name_safe = folder_path.replace("/", "_").replace("\\", "_")
        self._output.save_summary(
            f"obsidian_{folder_name_safe}",
            f"Obsidian Folder Summary: {folder_path}",
            f"*{len(notes)} notes in '{folder_path}' analyzed*\n\n{response}",
            provider=self._llm_service.provider,
            model=self._llm_service.model,
        )

    def _list_folders(self, vault: ObsidianVault) -> None:
        """List all folders in the vault."""
        folders: set[str] = set()
        for note in vault.notes():
            # Get relative path from vault root
            rel_path = note.path.parent.relative_to(vault.path)
            if str(rel_path) != ".":
                # Add all parent folders too
                parts = rel_path.parts
                for i in range(len(parts)):
                    folders.add("/".join(parts[: i + 1]))

        if folders:
            print("\n📁 Folders in vault:")
            for folder in sorted(folders):
                # Count notes in this folder
                count = len(vault.notes_in_folder(folder))
                print(f"   {folder}/ ({count} notes)")
        else:
            print("\n📁 No subfolders found (all notes in vault root)")

    def _list_notes(self, vault: ObsidianVault) -> None:
        """List all notes in vault."""
        notes = list(vault.notes())
        print(f"\n📋 Notes in vault ({len(notes)} total):\n")

        for note in sorted(notes, key=lambda n: n.title.lower()):
            tags = f" [{', '.join(note.tags)}]" if note.tags else ""
            print(f"  • {note.title}{tags}")

    def _print_note_info(self, note: ObsidianNote) -> None:
        """Print information about a note."""
        print(f"\n📝 Note: {note.title}")
        print(f"   Words: {note.word_count}")
        print(f"   Tags: {', '.join(note.tags) or 'none'}")
        print(f"   Links: {len(note.links)}")

        if note.backlinks:
            print(f"   Backlinks: {len(note.backlinks)}")


class WebHandler(BaseHandler):
    """Handler for web page parsing and summarization."""

    def run(self) -> None:
        """Execute web workflow."""
        print_header("🌐 Web Page Mode", "-", 40)

        url = prompt_input("Enter URL")
        if not url:
            return

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        print(f"\n🔍 Fetching content from: {url}")
        content = WebService.get_web_content(url)

        if not content:
            print_error("Could not fetch content from URL.")
            return

        word_count = len(content.split())
        print(f"✅ Fetched {word_count:,} words")

        # Show preview
        preview = content[:500] + "..." if len(content) > 500 else content
        print(f"\n📄 Preview:\n{preview}")

        if not confirm("\nGenerate summary?"):
            self._save_content(url, content)
            return

        self._configure_llm()

        print(f"\n🤖 Summarizing with {self._llm_service.model}...")
        response = self._llm_service.generate(
            LLMRequest(
                prompt_key="webpage_summary",
                variables={"content": content},
            )
        )

        print_section("Summary")
        print(response)

        name = get_name_from_source(url)
        self._output.save_summary(
            name,
            f"Summary: {url}",
            response,
            provider=self._llm_service.provider,
            model=self._llm_service.model,
        )
        self._save_content(url, content)

    def _save_content(self, url: str, content: str) -> None:
        """Save web content to file."""
        name = get_name_from_source(url)
        self._output.save_content(name, content, header=f"URL: {url}\n")
