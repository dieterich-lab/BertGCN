import argparse
import sys
from typing import Optional, Union

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


def parse_args(args: Optional[list] = None):
    """
    Parse command line arguments for BertGCN graph building.

    Args:
        args: Optional list of command line arguments to parse.
              If None, sys.argv[1:] is used.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Build document-word graph with TF-IDF and PMI edges for BertGCN"
    )

    # Dataset configuration
    parser.add_argument(
        "--doclevel",
        type=str,
        default=DEFAULT_DOCUMENT_LEVEL,
        help=f"Document level (letter, sentence, etc.) (default: {DEFAULT_DOCUMENT_LEVEL})",
    )
    parser.add_argument(
        "--bertmodel",
        type=str,
        default="medbert",
        choices=list(MODEL_PATHS.keys()),
        help=f"BERT model to use (default: medbert). Available models: {', '.join(MODEL_PATHS.keys())}",
    )
    parser.add_argument(
        "--testunklar", action="store_true", help="Test unclear samples"
    )

    # Graph building parameters
    parser.add_argument(
        "--window_size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Size of sliding window for word co-occurrence (default: {DEFAULT_WINDOW_SIZE})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of documents to process at once (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--bidirectional_tfidf",
        action="store_true",
        default=DEFAULT_USE_BIDIRECTIONAL_TFIDF,
        help=f"Whether to add bidirectional TF-IDF edges (default: {DEFAULT_USE_BIDIRECTIONAL_TFIDF})",
    )
    parser.add_argument(
        "--min_pmi",
        type=float,
        default=DEFAULT_MIN_PMI,
        help=f"Minimum PMI value threshold (default: {DEFAULT_MIN_PMI})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )

    parsed_args = parser.parse_args(args)

    # Set model path based on the selected model name
    if hasattr(parsed_args, "bertmodel"):
        parsed_args.model_path = MODEL_PATHS.get(
            parsed_args.bertmodel, DEFAULT_MODEL_PATH
        )

    return parsed_args
