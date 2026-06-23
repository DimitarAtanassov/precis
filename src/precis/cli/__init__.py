"""
CLI module for user interaction.

Separates presentation logic from business logic.
"""

from precis.cli.handlers import ObsidianHandler, PaperHandler, WebHandler
from precis.cli.menu import Menu, MenuItem

__all__ = [
    "Menu",
    "MenuItem",
    "PaperHandler",
    "ObsidianHandler",
    "WebHandler",
]
