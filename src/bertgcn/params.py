"""
Parameter handling for BertGCN using typer for modern CLI interface.

This module provides command-line argument parsing for BertGCN using typer,
which offers a more modern and intuitive CLI experience compared to argparse.
"""

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from bertgcn.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DOCUMENT_LEVEL,
    DEFAULT_MIN_PMI,
    DEFAULT_MODEL_PATH,
    DEFAULT_SEED,
    DEFAULT_USE_BIDIRECTIONAL_TFIDF,
    DEFAULT_WINDOW_SIZE,
    MODEL_PATHS,
)

# Initialize console for rich output
console = Console()


class ModelType(str, Enum):
    """Enum for available BERT models."""

    MEDBERT = "medbert"
    ROBERTA = "roberta"
    BERT = "bert"


@dataclass
class BertGCNParameters:
    """Container for all parameters used in BertGCN."""

    # Dataset configuration
    doclevel: str = DEFAULT_DOCUMENT_LEVEL
    bertmodel: str = "medbert"
    model_path: str = field(default="", init=False)
    testunklar: bool = False

    # Graph building parameters
    window_size: int = DEFAULT_WINDOW_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    bidirectional_tfidf: bool = DEFAULT_USE_BIDIRECTIONAL_TFIDF
    min_pmi: float = DEFAULT_MIN_PMI

    # General parameters
    seed: int = DEFAULT_SEED

    def __post_init__(self):
        """Set derived attributes after initialization."""
        # Ensure model_path is always set based on bertmodel
        self.model_path = MODEL_PATHS.get(self.bertmodel, DEFAULT_MODEL_PATH)

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary for display."""
        return {
            "doclevel": self.doclevel,
            "bertmodel": self.bertmodel,
            "model_path": self.model_path,
            "testunklar": self.testunklar,
            "window_size": self.window_size,
            "batch_size": self.batch_size,
            "bidirectional_tfidf": self.bidirectional_tfidf,
            "min_pmi": self.min_pmi,
            "seed": self.seed,
        }


def create_cli_app(name: str, help_text: str) -> typer.Typer:
    """
    Create a typer app with common settings.

    Args:
        name: Name of the CLI app
        help_text: Help text for the CLI app

    Returns:
        Configured typer app
    """
    return typer.Typer(
        name=name,
        help=help_text,
        add_completion=False,
        rich_markup_mode="rich",
    )


def parse_args(args: Optional[List[str]] = None) -> BertGCNParameters:
    """
    Parse command line arguments for BertGCN using typer.

    Args:
        args: Optional list of command line arguments to parse.
              If None, sys.argv[1:] is used.

    Returns:
        BertGCNParameters object with all parsed arguments
    """
    # Create parameters object with default values
    parameters = BertGCNParameters()

    # Create a typer app for CLI
    app = create_cli_app(
        name="BertGCN",
        help_text="Build document-word graph with TF-IDF and PMI edges for BertGCN",
    )

    # Define the main command (build) with all parameter options
    @app.command()
    def build(
        # Dataset configuration
        doclevel: str = typer.Option(
            DEFAULT_DOCUMENT_LEVEL,
            "--doclevel",
            "-d",
            help="Document level (letter, sentence, etc.)",
        ),
        bertmodel: ModelType = typer.Option(
            ModelType.MEDBERT,
            "--bertmodel",
            "-b",
            case_sensitive=False,
            help=f"BERT model to use. Available models: {', '.join(MODEL_PATHS.keys())}",
        ),
        testunklar: bool = typer.Option(
            False,
            "--testunklar",
            help="Test unclear samples",
        ),
        # Graph building parameters
        window_size: int = typer.Option(
            DEFAULT_WINDOW_SIZE,
            "--window-size",
            "-w",
            help="Size of sliding window for word co-occurrence",
        ),
        batch_size: int = typer.Option(
            DEFAULT_BATCH_SIZE,
            "--batch-size",
            "-bs",
            help="Number of documents to process at once",
        ),
        bidirectional_tfidf: bool = typer.Option(
            DEFAULT_USE_BIDIRECTIONAL_TFIDF,
            "--bidirectional-tfidf",
            "-bt",
            help="Whether to add bidirectional TF-IDF edges",
        ),
        min_pmi: float = typer.Option(
            DEFAULT_MIN_PMI,
            "--min-pmi",
            "-p",
            help="Minimum PMI value threshold",
        ),
        seed: int = typer.Option(
            DEFAULT_SEED,
            "--seed",
            "-s",
            help="Random seed for reproducibility",
        ),
    ):
        """Build document-word graph with TF-IDF and PMI edges."""
        # Update parameters
        parameters.doclevel = doclevel
        parameters.bertmodel = bertmodel.value
        parameters.testunklar = testunklar
        parameters.window_size = window_size
        parameters.batch_size = batch_size
        parameters.bidirectional_tfidf = bidirectional_tfidf
        parameters.min_pmi = min_pmi
        parameters.seed = seed
        parameters.model_path = MODEL_PATHS.get(bertmodel.value, DEFAULT_MODEL_PATH)

        # Display parameter summary
        show_parameter_summary(parameters.to_dict())

    # Run the app with specified args or with sys.argv
    if args is not None:
        # Save original argv
        sys_argv = sys.argv.copy()
        # Set new argv for typer to parse
        sys.argv = [sys.argv[0]] + (args if isinstance(args, list) else [args])
        try:
            app(standalone_mode=False)
        except SystemExit:
            # Suppress SystemExit when --help is called
            pass
        finally:
            # Restore original argv
            sys.argv = sys_argv
    else:
        # For direct CLI usage, suppress SystemExit
        try:
            app(standalone_mode=False)
        except SystemExit:
            pass

    # Always return parameters
    return parameters


def show_parameter_summary(params: Dict[str, Any]) -> None:
    """
    Display a summary of the parameters being used.

    Args:
        params: Dictionary of parameter name-value pairs
    """
    # Format parameters for display with nice formatting
    param_lines = []
    for k, v in params.items():
        # Format boolean values as colored text
        if isinstance(v, bool):
            color = "green" if v else "red"
            v_str = f"[{color}]{v}[/{color}]"
        else:
            v_str = str(v)
        param_lines.append(f"  [bold]{k}[/bold]: {v_str}")

    param_text = "\n".join(param_lines)

    # Display in a nice panel
    console.print(
        Panel(
            param_text,
            title="[bold blue]BertGCN Parameters[/bold blue]",
            border_style="blue",
            expand=False,
        )
    )


if __name__ == "__main__":
    """Run the CLI directly for testing."""
    # Create a main typer app with subcommands
    cli = create_cli_app(name="BertGCN CLI", help_text="BertGCN command line interface")

    # Add build command
    @cli.command()
    def build(
        doclevel: str = typer.Option(DEFAULT_DOCUMENT_LEVEL, "--doclevel", "-d"),
        window_size: int = typer.Option(DEFAULT_WINDOW_SIZE, "--window-size", "-w"),
    ):
        """Build the graph."""
        console.print(f"Building graph with [bold]doclevel={doclevel}[/bold]")
        console.print(f"Window size: [bold]{window_size}[/bold]")

    # Add train command
    @cli.command()
    def train(epochs: int = typer.Option(10, "--epochs", "-e")):
        """Train the model."""
        console.print(f"Training for [bold]{epochs}[/bold] epochs")

    # Run the CLI
    cli()
