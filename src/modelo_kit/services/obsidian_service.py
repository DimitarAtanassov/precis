"""
High-level service for using Obsidian notes as LLM context.
"""

from pathlib import Path

from modelo_kit.models import ObsidianNote
from modelo_kit.services.obsidian_vault import ObsidianVault


class ContextConfig:
    """Configuration for context retrieval."""

    def __init__(
        self,
        max_notes: int = 10,
        max_tokens: int = 8000,
        include_backlinks: bool = True,
        include_metadata: bool = True,
        chars_per_token: int = 4,
    ) -> None:
        self.max_notes = max_notes
        self.max_tokens = max_tokens
        self.include_backlinks = include_backlinks
        self.include_metadata = include_metadata
        self.chars_per_token = chars_per_token


class ObsidianService:
    """
    Facade for using Obsidian vault as LLM context.

    Provides methods to:
    - Retrieve relevant notes for a query
    - Format notes as context for LLM prompts
    - Build context from specific notes or tags

    Usage:
        service = ObsidianService("/path/to/vault")

        # Get context for a topic
        context = service.get_context_for_query("machine learning")

        # Get context from specific notes
        context = service.get_context_from_notes(["Note 1", "Note 2"])

        # Get context by tag
        context = service.get_context_by_tag("project")

        # Use with LLM
        prompt = f"Based on these notes:\\n{context}\\n\\nAnswer: {question}"
    """

    def __init__(
        self,
        vault_path: str | Path,
        config: ContextConfig | None = None,
    ) -> None:
        """
        Initialize the Obsidian service.

        Args:
            vault_path: Path to the Obsidian vault.
            config: Optional configuration for context retrieval.
        """
        self.vault = ObsidianVault(vault_path)
        self.config = config or ContextConfig()

    def get_context_for_query(self, query: str) -> str:
        """
        Get formatted context for a search query.

        Args:
            query: Search string to find relevant notes.

        Returns:
            Formatted context string for LLM consumption.
        """
        notes = self.vault.search(query)
        return self._format_context(notes[: self.config.max_notes])

    def get_context_from_notes(self, note_names: list[str]) -> str:
        """
        Get formatted context from specific notes.

        Args:
            note_names: List of note names to include.

        Returns:
            Formatted context string.
        """
        notes = []
        for name in note_names:
            note = self.vault.get_note(name)
            if note:
                notes.append(note)

        return self._format_context(notes)

    def get_context_by_tag(self, tag: str) -> str:
        """
        Get formatted context from notes with a specific tag.

        Args:
            tag: Tag to filter by (with or without #).

        Returns:
            Formatted context string.
        """
        notes = self.vault.notes_with_tag(tag)
        return self._format_context(notes[: self.config.max_notes])

    def get_context_from_folder(self, folder: str) -> str:
        """
        Get formatted context from notes in a folder.

        Args:
            folder: Folder path relative to vault root.

        Returns:
            Formatted context string.
        """
        notes = self.vault.notes_in_folder(folder)
        return self._format_context(notes[: self.config.max_notes])

    def get_note_with_context(self, note_name: str, depth: int = 1) -> str:
        """
        Get a note with its linked notes as context.

        Args:
            note_name: Name of the primary note.
            depth: How many levels of links to follow (1 = direct links only).

        Returns:
            Formatted context including the note and its linked notes.
        """
        primary = self.vault.get_note(note_name)
        if not primary:
            return ""

        notes = [primary]
        seen = {note_name.lower()}

        # Gather linked notes
        if depth >= 1:
            for link in primary.links:
                if link.target.lower() not in seen:
                    linked_note = self.vault.get_note(link.target)
                    if linked_note:
                        notes.append(linked_note)
                        seen.add(link.target.lower())

            # Also include backlinks if configured
            if self.config.include_backlinks:
                self.vault.build_backlinks()
                for backlink_name in primary.backlinks:
                    if backlink_name.lower() not in seen:
                        backlink_note = self.vault.get_note(backlink_name)
                        if backlink_note:
                            notes.append(backlink_note)
                            seen.add(backlink_name.lower())

        return self._format_context(notes[: self.config.max_notes])

    def list_all_tags(self) -> list[str]:
        """Get all unique tags in the vault."""
        return self.vault.get_stats().unique_tags

    def list_all_notes(self) -> list[str]:
        """Get all note titles in the vault."""
        return [note.title for note in self.vault.notes()]

    def _format_context(self, notes: list[ObsidianNote]) -> str:
        """
        Format notes as a context string, respecting token limits.

        Args:
            notes: List of notes to format.

        Returns:
            Formatted context string.
        """
        if not notes:
            return ""

        parts = []
        total_chars = 0
        max_chars = self.config.max_tokens * self.config.chars_per_token

        for note in notes:
            note_context = note.to_context_string(
                include_metadata=self.config.include_metadata
            )

            # Check if adding this note would exceed limit
            if total_chars + len(note_context) > max_chars:
                # Try to add truncated version
                remaining = max_chars - total_chars - 100  # Leave room for separator
                if remaining > 500:  # Only add if meaningful content can fit
                    truncated = note_context[:remaining] + "\n\n[Content truncated...]"
                    parts.append(truncated)
                break

            parts.append(note_context)
            total_chars += len(note_context)

        return "\n\n---\n\n".join(parts)

    def get_vault_summary(self) -> str:
        """Get a summary of the vault for LLM context."""
        stats = self.vault.get_stats()

        return f"""Obsidian Vault Summary:
- Total Notes: {stats.total_notes}
- Total Words: {stats.total_words:,}
- Total Links: {stats.total_links}
- Unique Tags: {len(stats.unique_tags)}
- Top Tags: {", ".join(stats.unique_tags[:10])}
"""
