#!/usr/bin/env python3
"""
Minimal Document-Word Graph Builder for Clinical Text Classification
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

from scipy.sparse import csr_matrix
from transformers import AutoTokenizer

from data_manager import (
    create_data_matrices,
    get_embedding_dim,
    load_or_create_dataset,
    save_graph_files,
)
from entry import PRETRAINEDMODEL
from graph_algorithms import (
    build_adjacency_matrix,
    build_vocabulary,
    calculate_pmi_edges,
    calculate_tfidf_edges,
    calculate_word_doc_counts,
    create_splits,
    generate_windows,
)
from params import parse_args


@dataclass
class GraphConfig:
    """Graph building configuration."""

    window_size: int = 20
    min_word_freq: int = 1
    test_split: float = 0.2
    val_split: float = 0.1
    random_seed: int = 0


def build_graph(doclevel: str, testunklar: bool = False) -> Tuple[csr_matrix, Dict]:
    """Build complete document-word graph."""
    # Setup
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINEDMODEL)
    config = GraphConfig(random_seed=0)

    # Load data
    dataset = load_or_create_dataset(tokenizer, doclevel, clean=True)
    tokenized_texts = [text.split() for text in dataset.texts]
    embed_dim = get_embedding_dim(tokenizer)

    # Create splits
    if testunklar:
        train_indices, test_indices = [], []
        for i, example in enumerate(dataset):
            if "unklar" in dataset.LE.classes_[example["labels"]]:
                test_indices.append(i)
            else:
                train_indices.append(i)

        import numpy as np

        np.random.seed(config.random_seed)
        np.random.shuffle(train_indices)
        val_size = int(len(train_indices) * 0.1)
        val_indices = train_indices[:val_size]
        train_indices = train_indices[val_size:]
    else:
        train_indices, val_indices, test_indices = create_splits(
            len(dataset), config.test_split, config.val_split, config.random_seed
        )

    # Build vocabulary and calculate statistics
    vocab, word2id = build_vocabulary(tokenized_texts, config.min_word_freq)
    word_doc_counts = calculate_word_doc_counts(tokenized_texts, vocab)

    # Generate edges
    windows = generate_windows(tokenized_texts, config.window_size)
    pmi_edges = calculate_pmi_edges(windows, word2id, len(train_indices))
    tfidf_edges = calculate_tfidf_edges(
        tokenized_texts,
        word2id,
        word_doc_counts,
        train_indices,
        val_indices,
        test_indices,
        len(dataset),
        len(vocab),
    )

    # Build adjacency matrix
    node_size = len(dataset) + len(vocab)
    adj_matrix = build_adjacency_matrix(pmi_edges, tfidf_edges, node_size)

    # Create metadata
    metadata = {
        "vocab_size": len(vocab),
        "embed_dim": embed_dim,
        "node_size": node_size,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "test_size": len(test_indices),
        "vocab": vocab,
        "word2id": word2id,
        "label_classes": dataset.LE.classes_,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }

    return adj_matrix, metadata, dataset


def main():
    """Main function."""
    args = parse_args()

    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

    # Build graph
    adj_matrix, metadata, dataset = build_graph(args.doclevel, args.testunklar)

    # Save graph data
    data_matrices = create_data_matrices(dataset, metadata, metadata["embed_dim"])
    dataset_name = f"medindcls_{args.doclevel}"
    save_graph_files(
        adj_matrix,
        data_matrices,
        metadata,
        dataset_name,
        args.doclevel,
        args.testunklar,
    )

    print(f"Graph built: {metadata['node_size']} nodes, {adj_matrix.nnz} edges")
    print(
        f"Train/Val/Test: {metadata['train_size']}/{metadata['val_size']}/{metadata['test_size']}"
    )


if __name__ == "__main__":
    main()
