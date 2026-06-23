"""
Reusable menu components for CLI interaction.

Provides a consistent interface for displaying menus and getting user input.
"""

from dataclasses import dataclass


@dataclass
class MenuItem:
    """A single menu option."""

    key: str
    label: str
    icon: str = ""

    def display(self) -> str:
        """Format the menu item for display."""
        prefix = f"{self.icon} " if self.icon else ""
        return f"  {self.key}. {prefix}{self.label}"


class Menu:
    """
    Interactive CLI menu.

    Usage:
        menu = Menu("Select option:", [
            MenuItem("1", "First option", "📄"),
            MenuItem("2", "Second option", "📓"),
        ])
        choice = menu.display()
    """

    def __init__(self, title: str, items: list[MenuItem]) -> None:
        """
        Initialize the menu.

        Args:
            title: Menu title/prompt.
            items: List of menu items.
        """
        self.title = title
        self.items = items
        self._valid_keys = {item.key for item in items}

    def display(self) -> str:
        """
        Display menu and get user choice.

        Returns:
            The key of the selected menu item.
        """
        print(f"\n{self.title}")
        for item in self.items:
            print(item.display())

        max_key = max(int(k) for k in self._valid_keys if k.isdigit())

        while True:
            choice = input(f"\nChoice (1-{max_key}): ").strip()
            if choice in self._valid_keys:
                return choice
            print("Invalid choice, try again.")


# Pre-defined menus
MAIN_MENU = Menu(
    "Select content source:",
    [
        MenuItem("1", "Research Paper (PDF file or URL)", "📄"),
        MenuItem("2", "Obsidian Vault (markdown notes)", "📓"),
        MenuItem("3", "Web Page (URL)", "🌐"),
        MenuItem("4", "Exit", "❌"),
    ],
)

OBSIDIAN_MENU = Menu(
    "Select action:",
    [
        MenuItem("1", "Summarize a single note", "📝"),
        MenuItem("2", "Summarize notes in folder", "�"),
        MenuItem("3", "List all notes", "📋"),
        MenuItem("4", "Back to main menu", "↩️"),
    ],
)

LLM_MENU = Menu(
    "Select LLM provider:",
    [
        MenuItem("1", "Claude", ""),
        MenuItem("2", "OpenAI", ""),
        MenuItem("3", "Gemini", ""),
        MenuItem("4", "DeepSeek", ""),
    ],
)
