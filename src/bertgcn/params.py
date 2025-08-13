import argparse


def parse_args():
    """
    Parse command line arguments for BertGCN graph building.

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
        default="letter",
        help="Document level (letter, sentence, etc.)",
    )
    parser.add_argument(
        "--bertmodel", type=str, default="medbert", help="BERT model to use"
    )
    parser.add_argument(
        "--testunklar", action="store_true", help="Test unclear samples"
    )

    # Graph building parameters
    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="Size of sliding window for word co-occurrence (default: 20)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of documents to process at once (default: 1000)",
    )
    parser.add_argument(
        "--bidirectional_tfidf",
        action="store_true",
        default=True,
        help="Whether to add bidirectional TF-IDF edges (default: True)",
    )
    parser.add_argument(
        "--min_pmi",
        type=float,
        default=0.0,
        help="Minimum PMI value threshold (default: 0.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )

    return parser.parse_args()
