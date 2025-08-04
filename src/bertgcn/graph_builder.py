"""
Enhanced graph building functionality for BertGCN.

Builds document-word heterogeneous graphs for clinical text classification.
"""

import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix, save_npz
from transformers import AutoTokenizer

from .config_enhanced import get_graph_config, get_model_path, get_paths
from .datasets import CleanClinicDataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def build_graph_enhanced(
    doclevel: str = "letter", testunklar: bool = False, **kwargs
) -> Dict:
    """Build document-word graph for clinical text classification with enhanced features."""

    logging.info(
        f"Building enhanced graph for doclevel: {doclevel}, testunklar: {testunklar}"
    )

    # Get configurations
    paths = get_paths()
    model_path = get_model_path()
    graph_config = get_graph_config()

    # Override config with any provided kwargs
    graph_config.update(kwargs)

    # Initialize tokenizer
    logging.info(f"Loading tokenizer from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        logging.error(f"Failed to load tokenizer from {model_path}: {e}")
        logging.info("Attempting to use fallback model...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load dataset
    logging.info("Loading dataset...")
    dataset = CleanClinicDataset(tokenizer, doclevel=doclevel, clean=True)

    # Build vocabulary with frequency filtering
    logging.info("Building vocabulary with frequency filtering...")
    word_counts = Counter()
    processed_texts = []

    for text in dataset.texts:
        words = text.lower().split()
        processed_texts.append(words)
        word_counts.update(words)

    # Filter vocabulary by frequency
    vocab = [
        word
        for word, count in word_counts.items()
        if count >= graph_config["vocab_min_freq"]
    ]

    # Limit vocabulary size if specified
    if graph_config["max_vocab_size"] and len(vocab) > graph_config["max_vocab_size"]:
        # Keep most frequent words
        vocab = [
            word
            for word, _ in word_counts.most_common(graph_config["max_vocab_size"])
            if word_counts[word] >= graph_config["vocab_min_freq"]
        ]

    word2id = {word: i for i, word in enumerate(vocab)}

    logging.info(
        f"Vocabulary size: {len(vocab)} (filtered from {len(word_counts)} total words)"
    )

    # Create document-word connections
    num_docs = len(dataset)
    num_words = len(vocab)
    total_nodes = num_docs + num_words

    logging.info(
        f"Creating graph with {num_docs} documents and {num_words} words ({total_nodes} total nodes)"
    )

    # Build adjacency matrix with document-word edges
    row_indices = []
    col_indices = []
    data = []

    # Add document-word edges
    for doc_idx, words in enumerate(processed_texts):
        word_counts_in_doc = Counter(words)
        total_words_in_doc = len(words)

        for word in set(words):
            if word in word2id:
                word_idx = word2id[word] + num_docs  # Offset by number of documents

                # TF-IDF-like weighting
                tf = word_counts_in_doc[word] / total_words_in_doc

                # Add bidirectional edges
                row_indices.extend([doc_idx, word_idx])
                col_indices.extend([word_idx, doc_idx])
                data.extend([tf, tf])

    # Create sparse adjacency matrix
    adj_matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(total_nodes, total_nodes),
        dtype=np.float32,
    )

    logging.info(f"Adjacency matrix created with {len(data)} edges")

    # Split data according to configuration
    train_size = int(graph_config["train_ratio"] * num_docs)
    val_size = int(graph_config["val_ratio"] * num_docs)
    test_size = num_docs - train_size - val_size

    # Ensure we don't have empty sets
    if test_size <= 0:
        test_size = 1
        val_size = max(1, val_size - 1)
        train_size = num_docs - val_size - test_size

    train_labels = dataset.ohe_labels[:train_size]
    val_labels = dataset.ohe_labels[train_size : train_size + val_size]
    test_labels = dataset.ohe_labels[train_size + val_size :]

    logging.info(
        f"Data split - Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}"
    )

    # Create graph name
    graph_name = f"medindcls_{doclevel}"
    if testunklar:
        graph_name += "_testunklar"

    graph_dir = paths["graphs"] / graph_name
    graph_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving graph files to: {graph_dir}")

    # Save adjacency matrix
    save_npz(graph_dir / f"ind.{graph_name}.adj", adj_matrix)

    # Save feature matrices (labels as features for now)
    with open(graph_dir / f"ind.{graph_name}.x", "wb") as f:
        pickle.dump(train_labels, f)

    with open(graph_dir / f"ind.{graph_name}.vx", "wb") as f:
        pickle.dump(val_labels, f)

    with open(graph_dir / f"ind.{graph_name}.tx", "wb") as f:
        pickle.dump(test_labels, f)

    # Save vocabulary
    with open(graph_dir / f"ind.{graph_name}.vocab", "wb") as f:
        pickle.dump({"word2id": word2id, "vocab": vocab}, f)

    # Save processed texts for debugging
    with open(graph_dir / f"ind.{graph_name}.texts", "wb") as f:
        pickle.dump(processed_texts, f)

    # Enhanced metadata with more information
    metadata = {
        "num_docs": num_docs,
        "num_words": num_words,
        "total_nodes": total_nodes,
        "vocab_size": len(vocab),
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "num_classes": len(dataset.class_names),
        "class_names": list(dataset.class_names),
        "graph_config": graph_config,
        "model_path": model_path,
        "total_edges": len(data),
        "avg_edges_per_doc": len(data)
        / (2 * num_docs),  # Divided by 2 because edges are bidirectional
        "doclevel": doclevel,
        "testunklar": testunklar,
    }

    with open(graph_dir / f"ind.{graph_name}.metadata", "wb") as f:
        pickle.dump(metadata, f)

    # Save human-readable summary
    summary_text = f"""
Graph Summary for {graph_name}
=====================================
Documents: {num_docs}
Words: {num_words}
Total nodes: {total_nodes}
Total edges: {len(data)}
Classes: {len(dataset.class_names)} ({', '.join(dataset.class_names)})

Data Split:
- Training: {train_size} ({train_size/num_docs*100:.1f}%)
- Validation: {val_size} ({val_size/num_docs*100:.1f}%)
- Test: {test_size} ({test_size/num_docs*100:.1f}%)

Vocabulary Statistics:
- Original word count: {len(word_counts)}
- Filtered vocabulary: {len(vocab)}
- Min frequency threshold: {graph_config["vocab_min_freq"]}
- Max vocabulary size: {graph_config["max_vocab_size"] or "unlimited"}

Model: {model_path}
"""

    with open(graph_dir / f"{graph_name}_summary.txt", "w") as f:
        f.write(summary_text)

    logging.info(
        f"Graph built successfully: {total_nodes} nodes, {len(dataset.class_names)} classes, {len(data)} edges"
    )

    return {
        "adj_matrix": adj_matrix,
        "metadata": metadata,
        "graph_dir": graph_dir,
        "graph_name": graph_name,
        "vocab": vocab,
        "word2id": word2id,
    }
