"""
Obsidian vault management and note discovery.
"""

from collections.abc import Iterator
from pathlib import Path

from precis.models import ObsidianNote, VaultStats
from precis.parsers import ObsidianParser


class ObsidianVault:
    """
    Manages access to an Obsidian vault.

    Provides methods for:
    - Discovering notes
    - Reading individual notes
    - Filtering by tags, folders, or links
    - Building the link graph (backlinks)

    Usage:
        vault = ObsidianVault("/path/to/vault")

        # Get all notes
        for note in vault.notes():
            print(note.title)

        # Get specific note
        note = vault.get_note("My Note")

        # Filter by tag
        tagged = vault.notes_with_tag("project")
    """

    EXCLUDED_FOLDERS = {".obsidian", ".trash", ".git", "node_modules"}

    def __init__(self, vault_path: str | Path) -> None:
        """
        Initialize vault connection.

        Args:
            vault_path: Path to the Obsidian vault root directory.

        Raises:
            ValueError: If path doesn't exist or isn't a directory.
        """
        self.path = Path(vault_path).resolve()

        if not self.path.exists():
            raise ValueError(f"Vault path does not exist: {self.path}")
        if not self.path.is_dir():
            raise ValueError(f"Vault path is not a directory: {self.path}")

        self._parser = ObsidianParser()
        self._cache: dict[str, ObsidianNote] = {}
        self._backlinks_built = False

    def notes(self, use_cache: bool = True) -> Iterator[ObsidianNote]:
        """
        Iterate over all notes in the vault.

        Args:
            use_cache: Whether to use cached notes if available.

        Yields:
            ObsidianNote objects for each .md file.
        """
        for md_path in self._discover_notes():
            yield self._get_or_parse(md_path, use_cache)

    def get_note(self, name: str) -> ObsidianNote | None:
        """
        Get a specific note by name (without .md extension).

        Args:
            name: Note name (case-insensitive).

        Returns:
            ObsidianNote if found, None otherwise.
        """
        name_lower = name.lower()

        # Check cache first
        if name_lower in self._cache:
            return self._cache[name_lower]

        # Search for the note
        for md_path in self._discover_notes():
            if md_path.stem.lower() == name_lower:
                return self._get_or_parse(md_path, use_cache=True)

        return None

    def notes_with_tag(self, tag: str) -> list[ObsidianNote]:
        """Get all notes containing a specific tag."""
        tag_lower = tag.lower().lstrip("#")
        return [
            note
            for note in self.notes()
            if any(t.lower() == tag_lower for t in note.tags)
        ]

    def notes_in_folder(self, folder: str) -> list[ObsidianNote]:
        """Get all notes in a specific folder (relative to vault root)."""
        folder_path = self.path / folder
        if not folder_path.exists():
            return []

        return [
            note
            for note in self.notes()
            if folder_path in note.path.parents or note.path.parent == folder_path
        ]

    def notes_linking_to(self, target_name: str) -> list[ObsidianNote]:
        """Get all notes that link to the specified note."""
        target_lower = target_name.lower()
        return [
            note
            for note in self.notes()
            if any(link.target.lower() == target_lower for link in note.links)
        ]

    def search(self, query: str) -> list[ObsidianNote]:
        """
        Search notes by content or title.

        Args:
            query: Search string (case-insensitive).

        Returns:
            List of matching notes, sorted by relevance (title matches first).
        """
        query_lower = query.lower()
        title_matches = []
        content_matches = []

        for note in self.notes():
            if query_lower in note.title.lower():
                title_matches.append(note)
            elif query_lower in note.body.lower():
                content_matches.append(note)

        return title_matches + content_matches

    def build_backlinks(self) -> None:
        """
        Build backlink graph for all notes.

        Updates each note's backlinks field with names of notes linking to it.
        """
        if self._backlinks_built:
            return

        # First pass: ensure all notes are cached
        all_notes = list(self.notes())

        # Build link map
        link_map: dict[str, list[str]] = {}  # target -> [sources]

        for note in all_notes:
            for link in note.links:
                target_lower = link.target.lower()
                if target_lower not in link_map:
                    link_map[target_lower] = []
                link_map[target_lower].append(note.title)

        # Update backlinks
        for note in all_notes:
            note.backlinks = link_map.get(note.title.lower(), [])

        self._backlinks_built = True

    def get_stats(self) -> VaultStats:
        """Get statistics about the vault."""
        all_notes = list(self.notes())
        all_tags: set[str] = set()
        total_words = 0
        total_links = 0
        notes_with_links: set[str] = set()
        linked_notes: set[str] = set()

        for note in all_notes:
            all_tags.update(note.tags)
            total_words += note.word_count
            total_links += len(note.links)

            if note.links:
                notes_with_links.add(note.title.lower())
                for link in note.links:
                    linked_notes.add(link.target.lower())

        orphans = [
            note.title
            for note in all_notes
            if note.title.lower() not in linked_notes
            and note.title.lower() not in notes_with_links
        ]

        return VaultStats(
            total_notes=len(all_notes),
            total_words=total_words,
            total_links=total_links,
            unique_tags=sorted(all_tags),
            orphan_notes=orphans,
        )

    def clear_cache(self) -> None:
        """Clear the note cache."""
        self._cache.clear()
        self._backlinks_built = False

    def _discover_notes(self) -> Iterator[Path]:
        """Discover all markdown files in the vault."""
        for md_path in self.path.rglob("*.md"):
            # Skip excluded folders
            if any(excluded in md_path.parts for excluded in self.EXCLUDED_FOLDERS):
                continue
            yield md_path

    def _get_or_parse(self, path: Path, use_cache: bool) -> ObsidianNote:
        """Get note from cache or parse it."""
        cache_key = path.stem.lower()

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        note: ObsidianNote = self._parser.parse(path)
        self._cache[cache_key] = note
        return note
