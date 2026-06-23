"""
CLI prompt utilities for user input.

Provides reusable functions for common user interactions.
"""

from precis.cli.menu import LLM_MENU
from precis.core.enums import LLMProvider


def confirm(message: str) -> bool:
    """
    Prompt user for yes/no confirmation.

    Args:
        message: The confirmation message to display.

    Returns:
        True if user confirmed, False otherwise.
    """
    response = input(f"{message} (y/n): ").strip().lower()
    return response == "y"


def prompt_input(message: str, required: bool = True) -> str | None:
    """
    Prompt user for text input.

    Args:
        message: The prompt message.
        required: If True, prints error when empty.

    Returns:
        User input or None if empty and required.
    """
    value = input(f"{message}: ").strip()
    if not value and required:
        print("❌ No input provided.")
        return None
    return value


def select_llm() -> tuple[str, str]:
    """
    Prompt user to select LLM provider and model.

    Returns:
        Tuple of (provider_name, model_name).
    """
    choice = LLM_MENU.display()
    provider = LLMProvider.from_choice(choice)

    print(f"\nDefault model: {provider.default_model}")
    custom_model = input("Enter model name (or press Enter for default): ").strip()
    model = custom_model if custom_model else provider.default_model

    return provider.provider_name, model


def print_header(title: str, char: str = "=", width: int = 50) -> None:
    """Print a formatted header."""
    print(f"\n{title}")
    print(char * width)


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✅ {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"❌ {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"ℹ️  {message}")


def print_section(title: str, char: str = "=", width: int = 60) -> None:
    """
    Print a section header with separators.

    Args:
        title: The section title to display.
        char: Character to use for the separator line.
        width: Width of the separator line.

    Example:
        >>> print_section("Executive Summary")
        ============================================================
        EXECUTIVE SUMMARY
        ============================================================
    """
    print("\n" + char * width)
    print(title.upper())
    print(char * width)
