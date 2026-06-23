"""
Main CLI application entry point.

This module provides a thin entry point that delegates to handlers.
All business logic is contained in the handlers module.
"""

import sys

from dotenv import load_dotenv

from precis.cli.handlers import ObsidianHandler, PaperHandler, WebHandler
from precis.cli.menu import MAIN_MENU

# Load environment variables
load_dotenv()


def main() -> None:
    """Main entry point for the CLI."""
    print("\n🧠 Précis - Content Parser & Summarizer")
    print("=" * 50)

    handlers = {
        "1": PaperHandler(),
        "2": ObsidianHandler(),
        "3": WebHandler(),
    }

    choice = MAIN_MENU.display()

    if handler := handlers.get(choice):
        handler.run()
    else:
        print("👋 Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
