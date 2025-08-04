#!/usr/bin/env python3
"""
Simple CLI for graph building in BertGCN.
"""

import logging
import sys
from pathlib import Path

import click

from .graph_builder import build_graph

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@click.command()
@click.option(
    "--doclevel", default="letter", help="Document level (letter, diagnosis, etc.)"
)
@click.option("--testunklar", is_flag=True, help="Use testunklar mode")
def main(doclevel: str, testunklar: bool):
    """Build document-word heterogeneous graph for clinical text classification."""
    try:
        logging.info(f"Building graph for {doclevel}")
        result = build_graph(doclevel=doclevel, testunklar=testunklar)
        logging.info(f"✅ Graph building completed: {result['graph_name']}")
        click.echo(f"✅ Graph saved to: {result['graph_dir']}")
    except Exception as e:
        logging.error(f"❌ Graph building failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
