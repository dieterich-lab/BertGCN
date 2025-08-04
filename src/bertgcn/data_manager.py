"""
Data management utilities for graph construction.

Handles dataset loading, saving, and matrix creation for the graph building pipeline.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bertgcn.config import get_paths
from bertgcn.data import CleanClinicDataset


def load_or_create_dataset(
    tokenizer: AutoTokenizer, doclevel: str, clean: bool = True
) -> CleanClinicDataset:
    """Load existing dataset or create new one."""
    paths = get_paths()
    dataset_file = paths.get_dataset_path(doclevel, "medbert", clean=clean)

    if dataset_file.exists():
        with open(dataset_file, "rb") as f:
            return pickle.load(f)

    dataset = CleanClinicDataset(tokenizer=tokenizer, doclevel=doclevel, clean=clean)
    with open(dataset_file, "wb") as f:
        pickle.dump(dataset, f)

    return dataset


def get_embedding_dim(tokenizer: AutoTokenizer) -> int:
    """Get embedding dimension from pretrained model."""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            tokenizer.name_or_path
        )
        embed_dim = model.bert.embeddings.word_embeddings.embedding_dim
        del model
        return embed_dim
    except:
        return 768  # Fallback to common BERT dimension


def create_data_matrices(
    dataset: CleanClinicDataset, metadata: Dict, embed_dim: int
) -> Dict[str, np.ndarray]:
    """Create data matrices for graph neural network training."""
    node_size = metadata["node_size"]
    vocab_size = metadata["vocab_size"]
    doc_size = len(dataset)

    # Initialize feature matrix
    feature_matrix = np.zeros((node_size, embed_dim), dtype=np.float32)

    # Document features (BERT embeddings or TF-IDF)
    # For now, we'll use random initialization - you can replace with actual BERT embeddings
    feature_matrix[:doc_size] = np.random.normal(0, 0.1, (doc_size, embed_dim))

    # Word features (word embeddings)
    # For now, random initialization - replace with word2vec/GloVe embeddings
    feature_matrix[doc_size:] = np.random.normal(0, 0.1, (vocab_size, embed_dim))

    # Create label matrix
    labels = np.array([dataset[i]["labels"] for i in range(len(dataset))])

    # Create train/val/test masks
    train_mask = np.zeros(node_size, dtype=bool)
    val_mask = np.zeros(node_size, dtype=bool)
    test_mask = np.zeros(node_size, dtype=bool)

    train_mask[metadata["train_indices"]] = True
    val_mask[metadata["val_indices"]] = True
    test_mask[metadata["test_indices"]] = True

    return {
        "features": feature_matrix,
        "labels": labels,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }


def save_graph_files(
    adj_matrix: csr_matrix,
    data_matrices: Dict[str, np.ndarray],
    metadata: Dict,
    dataset_name: str,
    doclevel: str,
    testunklar: bool = False,
) -> None:
    """Save all graph files to disk."""
    paths = get_paths()
    graph_dir = paths.get_graph_path(dataset_name, doclevel, testunklar)
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Save adjacency matrix
    np.savez_compressed(
        graph_dir / "adj_matrix.npz",
        data=adj_matrix.data,
        indices=adj_matrix.indices,
        indptr=adj_matrix.indptr,
        shape=adj_matrix.shape,
    )

    # Save data matrices
    for name, matrix in data_matrices.items():
        np.save(graph_dir / f"{name}.npy", matrix)

    # Save metadata
    with open(graph_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Graph saved to: {graph_dir}")


def load_graph_files(
    dataset_name: str, doclevel: str, testunklar: bool = False
) -> Tuple[csr_matrix, Dict[str, np.ndarray], Dict]:
    """Load graph files from disk."""
    paths = get_paths()
    graph_dir = paths.get_graph_path(dataset_name, doclevel, testunklar)

    # Load adjacency matrix
    adj_data = np.load(graph_dir / "adj_matrix.npz")
    adj_matrix = csr_matrix(
        (adj_data["data"], adj_data["indices"], adj_data["indptr"]),
        shape=adj_data["shape"],
    )

    # Load data matrices
    data_matrices = {}
    for file in graph_dir.glob("*.npy"):
        if file.stem != "adj_matrix":
            data_matrices[file.stem] = np.load(file)

    # Load metadata
    with open(graph_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return adj_matrix, data_matrices, metadata
