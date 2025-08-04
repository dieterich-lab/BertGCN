#!/usr/bin/env python3
"""
Enhanced CLI for graph building in BertGCN.
"""

import logging
import sys
from pathlib import Path

import click

from .config_enhanced import get_graph_config
from .graph_builder_enhanced import build_graph_enhanced

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@click.command()
@click.option(
    "--doclevel", default="letter", help="Document level (letter, diagnosis, etc.)"
)
@click.option("--testunklar", is_flag=True, help="Use testunklar mode")
@click.option(
    "--vocab-min-freq",
    type=int,
    default=None,
    help="Minimum frequency for words to be included in vocabulary",
)
@click.option(
    "--max-vocab-size",
    type=int,
    default=None,
    help="Maximum vocabulary size (most frequent words)",
)
@click.option(
    "--train-ratio",
    type=float,
    default=None,
    help="Training data ratio (e.g., 0.7 for 70%)",
)
@click.option(
    "--val-ratio",
    type=float,
    default=None,
    help="Validation data ratio (e.g., 0.1 for 10%)",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def main(
    doclevel: str,
    testunklar: bool,
    vocab_min_freq: int,
    max_vocab_size: int,
    train_ratio: float,
    val_ratio: float,
    verbose: bool,
):
    """Build document-word heterogeneous graph for clinical text classification."""

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Build configuration overrides
        config_overrides = {}
        if vocab_min_freq is not None:
            config_overrides["vocab_min_freq"] = vocab_min_freq
        if max_vocab_size is not None:
            config_overrides["max_vocab_size"] = max_vocab_size
        if train_ratio is not None:
            config_overrides["train_ratio"] = train_ratio
        if val_ratio is not None:
            config_overrides["val_ratio"] = val_ratio
            # Adjust test ratio
            test_ratio = 1.0 - train_ratio - val_ratio
            if test_ratio < 0:
                raise ValueError("Train ratio + Val ratio cannot exceed 1.0")
            config_overrides["test_ratio"] = test_ratio

        # Show configuration
        default_config = get_graph_config()
        default_config.update(config_overrides)

        click.echo("Configuration:")
        for key, value in default_config.items():
            click.echo(f"  {key}: {value}")
        click.echo()

        logging.info(f"Building enhanced graph for {doclevel}")
        result = build_graph_enhanced(
            doclevel=doclevel, testunklar=testunklar, **config_overrides
        )

        logging.info(f"✅ Graph building completed: {result['graph_name']}")

        # Display results
        metadata = result["metadata"]
        click.echo(f"✅ Graph saved to: {result['graph_dir']}")
        click.echo(f"📊 Graph statistics:")
        click.echo(f"  - Total nodes: {metadata['total_nodes']}")
        click.echo(f"  - Documents: {metadata['num_docs']}")
        click.echo(f"  - Words: {metadata['num_words']}")
        click.echo(f"  - Total edges: {metadata['total_edges']}")
        click.echo(f"  - Classes: {metadata['num_classes']}")
        click.echo(
            f"  - Train/Val/Test: {metadata['train_size']}/{metadata['val_size']}/{metadata['test_size']}"
        )

        # Show summary file location
        summary_file = result["graph_dir"] / f"{result['graph_name']}_summary.txt"
        click.echo(f"📋 Detailed summary: {summary_file}")

    except Exception as e:
        logging.error(f"❌ Graph building failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
