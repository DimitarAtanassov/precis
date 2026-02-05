"""
CLI module for user interaction.

Separates presentation logic from business logic.
"""

from modelo_kit.cli.handlers import ObsidianHandler, PaperHandler, WebHandler
from modelo_kit.cli.menu import Menu, MenuItem

__all__ = [
    "Menu",
    "MenuItem",
    "PaperHandler",
    "ObsidianHandler",
    "WebHandler",
]
