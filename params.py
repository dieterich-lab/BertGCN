import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Build graph for BertGCN")
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
    return parser.parse_args()
