"""CAPP GPT Serve - HTTP API for GPT-2 based manufacturing process planning."""

__version__ = "0.1.0"


def main() -> None:
    """Main entry point."""
    from .main import cli

    cli()
