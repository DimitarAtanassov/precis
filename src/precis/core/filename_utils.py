"""
Filename utilities for sanitizing and formatting output filenames.

These are pure functions with no side effects, extracted from OutputWriter
to be reusable across the codebase.
"""

import re
from pathlib import Path
from urllib.parse import urlparse


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be safe for use as a filename.

    Removes or replaces characters that are invalid in filenames
    across different operating systems.

    Args:
        name: The raw filename string.

    Returns:
        A sanitized filename safe for all platforms.

    Example:
        >>> sanitize_filename("My Paper: A Study (2024)")
        'My_Paper_A_Study_2024'
    """
    # Replace spaces and problematic characters with underscores
    sanitized = re.sub(r'[\s:;,/\\<>"|?*]+', "_", name)
    # Remove any remaining non-alphanumeric chars except underscore, dash, dot
    sanitized = re.sub(r"[^\w\-.]", "", sanitized)
    # Collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    return sanitized.strip("_")


def get_model_suffix(provider: str, model: str) -> str:
    """
    Generate a filename suffix from provider and model.

    Creates a readable suffix for distinguishing outputs from
    different LLM providers/models.

    Args:
        provider: LLM provider name (e.g., "claude", "openai").
        model: Model name (e.g., "claude-sonnet-4-5-20250929").

    Returns:
        Formatted suffix string.

    Example:
        >>> get_model_suffix("openai", "gpt-4o")
        '_openai_gpt-4o'
    """
    return f"_{provider}_{model}"


def get_name_from_source(source: str) -> str:
    """
    Extract a clean name from a file path or URL.

    Handles both local paths and URLs, extracting just the
    meaningful part of the filename without extension.

    Args:
        source: File path or URL string.

    Returns:
        Clean name suitable for use in output filenames.

    Example:
        >>> get_name_from_source("/path/to/my_paper.pdf")
        'my_paper'
        >>> get_name_from_source("https://arxiv.org/pdf/1234.5678.pdf")
        '1234.5678'
    """
    if source.startswith(("http://", "https://")):
        parsed = urlparse(source)
        path_part = parsed.path
        name = Path(path_part).stem
        # Handle arxiv and similar URLs
        if not name or name == "pdf":
            # Try to get meaningful part from path
            parts = [p for p in path_part.split("/") if p and p != "pdf"]
            name = parts[-1] if parts else "web_content"
    else:
        name = Path(source).stem

    return sanitize_filename(name)
