"""CLI adapter (Typer) over the shared application services.

Note: the Typer instance lives in ``precis.cli.app`` and is intentionally not
re-exported here, to avoid shadowing the ``app`` submodule name.
"""

from precis.cli.app import main

__all__ = ["main"]
