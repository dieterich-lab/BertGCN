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

from clinic_datasets import CleanClinicDataset
from config import get_paths


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
    """Create all required data matrices for graph training."""
    train_size = metadata["train_size"]
    val_size = metadata["val_size"]
    test_size = metadata["test_size"]
    vocab_size = metadata["vocab_size"]

    # Empty feature matrices (will be filled by BERT)
    x = csr_matrix((train_size, embed_dim), dtype=np.float32)
    vx = csr_matrix((val_size, embed_dim), dtype=np.float32)
    tx = csr_matrix((test_size, embed_dim), dtype=np.float32)
    allx = csr_matrix((train_size + vocab_size, embed_dim), dtype=np.float32)

    # Label matrices
    y = dataset.ohe_labels[metadata["train_indices"]]
    vy = dataset.ohe_labels[metadata["val_indices"]]
    ty = dataset.ohe_labels[metadata["test_indices"]]
    ally = np.concatenate(
        [y.toarray(), np.zeros((vocab_size, len(metadata["label_classes"])))]
    )

    return {
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "tx": tx,
        "ty": ty,
        "allx": allx,
        "ally": ally,
    }


def save_graph_files(
    adj_matrix: csr_matrix,
    data_matrices: Dict,
    metadata: Dict,
    dataset_name: str,
    doclevel: str,
    testunklar: bool = False,
) -> None:
    """Save all graph files to organized directories."""
    paths = get_paths()
    output_dir = paths.get_graph_path(dataset_name, doclevel, testunklar)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"ind.{dataset_name}_{doclevel}"
    if testunklar:
        base_name += "_testunklar"

    # Save all data matrices and adjacency matrix
    files_to_save = {**data_matrices, "adj": adj_matrix}

    for suffix, data in files_to_save.items():
        file_path = output_dir / f"{base_name}.{suffix}"
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    # Save metadata
    with open(output_dir / f"{base_name}.metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
