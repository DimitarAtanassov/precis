"""
Markdown parsing for Obsidian notes.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from precis.models import ObsidianLink, ObsidianNote


class ObsidianParser:
    """
    Parser for Obsidian markdown files.

    Handles:
    - YAML frontmatter
    - Wiki-style links: [[note]] and [[note|alias]]
    - Embedded links: ![[note]]
    - Tags: #tag and nested #tag/subtag
    """

    # Regex patterns
    FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
    WIKI_LINK_RE = re.compile(r"(!?)\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
    TAG_RE = re.compile(r"(?:^|\s)#([a-zA-Z0-9_/-]+)")

    def parse(self, path: Path) -> ObsidianNote:
        """
        Parse an Obsidian markdown file.

        Args:
            path: Path to the .md file.

        Returns:
            Parsed ObsidianNote object.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Note not found: {path}")

        content = path.read_text(encoding="utf-8")
        return self.parse_content(content, title=path.stem, path=path)

    def parse_content(
        self,
        content: str,
        title: str = "Untitled",
        path: Path | None = None,
    ) -> ObsidianNote:
        """
        Parse markdown content directly.

        Useful for testing or processing content from other sources.
        """
        frontmatter, body = self._extract_frontmatter(content)
        tags = self._extract_tags(body, frontmatter)
        links = self._extract_links(body)

        # Get file timestamps if path provided
        created_at = None
        modified_at = None
        if path and path.exists():
            stat = path.stat()
            created_at = datetime.fromtimestamp(stat.st_ctime)
            modified_at = datetime.fromtimestamp(stat.st_mtime)

        return ObsidianNote(
            title=title,
            path=path or Path(f"{title}.md"),
            content=content,
            body=body,
            frontmatter=frontmatter,
            tags=tags,
            links=links,
            created_at=created_at,
            modified_at=modified_at,
        )

    def _extract_frontmatter(self, content: str) -> tuple[dict[str, Any], str]:
        """Extract YAML frontmatter and return (frontmatter_dict, body)."""
        match = self.FRONTMATTER_RE.match(content)

        if not match:
            return {}, content.strip()

        yaml_content = match.group(1)
        body = content[match.end() :].strip()

        try:
            frontmatter = yaml.safe_load(yaml_content) or {}
        except yaml.YAMLError:
            frontmatter = {}

        return frontmatter, body

    def _extract_tags(self, body: str, frontmatter: dict[str, Any]) -> list[str]:
        """Extract tags from body text and frontmatter."""
        tags: set[str] = set()

        # Tags from body (#tag format)
        for match in self.TAG_RE.finditer(body):
            tags.add(match.group(1))

        # Tags from frontmatter
        fm_tags = frontmatter.get("tags", [])
        if isinstance(fm_tags, str):
            tags.add(fm_tags)
        elif isinstance(fm_tags, list):
            for tag in fm_tags:
                if isinstance(tag, str):
                    tags.add(tag)

        return sorted(tags)

    def _extract_links(self, body: str) -> list[ObsidianLink]:
        """Extract wiki-style links from body text."""
        links = []

        for match in self.WIKI_LINK_RE.finditer(body):
            is_embed = match.group(1) == "!"
            target = match.group(2).strip()
            alias = match.group(3).strip() if match.group(3) else None

            # Handle heading links: [[note#heading]]
            if "#" in target:
                target = target.split("#")[0]

            if target:
                links.append(
                    ObsidianLink(target=target, alias=alias, is_embed=is_embed)
                )

        return links
