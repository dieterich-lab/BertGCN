"""
Utility functions for the BertGCN package.

Common utilities used throughout the package.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Default paths
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_DATA_DIR = DEFAULT_OUTPUT_DIR / "data"
DEFAULT_MODELS_DIR = DEFAULT_OUTPUT_DIR / "models"
DEFAULT_GRAPHS_DIR = DEFAULT_DATA_DIR / "graphs"
DEFAULT_DATASETS_DIR = DEFAULT_DATA_DIR / "datasets"
DEFAULT_LOGS_DIR = DEFAULT_OUTPUT_DIR / "logs"


def create_directories():
    """Create necessary directories if they don't exist."""
    for directory in [
        DEFAULT_OUTPUT_DIR,
        DEFAULT_DATA_DIR,
        DEFAULT_MODELS_DIR,
        DEFAULT_GRAPHS_DIR,
        DEFAULT_DATASETS_DIR,
        DEFAULT_LOGS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


class Paths:
    """Class for managing file paths in the project."""

    def __init__(self):
        """Initialize paths and create directories."""
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.data_dir = DEFAULT_DATA_DIR
        self.models_dir = DEFAULT_MODELS_DIR
        self.graphs_dir = DEFAULT_GRAPHS_DIR
        self.datasets_dir = DEFAULT_DATASETS_DIR
        self.logs_dir = DEFAULT_LOGS_DIR

        # Create directories if they don't exist
        create_directories()

    def get_dataset_path(self, doclevel, model_name, clean=True):
        """Get path for processed dataset."""
        clean_suffix = "_clean" if clean else ""
        return self.datasets_dir / f"{doclevel}_{model_name}{clean_suffix}.pkl"

    def get_graph_path(self, dataset_name, doclevel, testunklar=False):
        """Get path for graph files."""
        graph_dir = self.graphs_dir / f"{dataset_name}"
        graph_dir.mkdir(parents=True, exist_ok=True)
        return graph_dir

    def get_finetuned_model_path(self, doclevel, clean=True):
        """Get path for fine-tuned BERT model."""
        clean_suffix = "_clean" if clean else ""
        return self.models_dir / "finetuned" / f"{doclevel}{clean_suffix}"

    def get_gcn_model_path(self, doclevel, clean=True):
        """Get path for trained GCN model."""
        clean_suffix = "_clean" if clean else ""
        return self.models_dir / "gcn" / f"{doclevel}{clean_suffix}"


def get_paths():
    """Get paths singleton instance."""
    return Paths()


def create_data_matrices(dataset, metadata, embed_dim):
    """Create data matrices for the graph."""
    logging.info("Creating data matrices...")

    # Extract sizes from metadata
    train_size = metadata["train_size"]
    val_size = metadata["val_size"]
    test_size = metadata["test_size"]
    vocab_size = metadata["vocab_size"]

    # Empty feature matrices (will be filled by BERT)
    x = csr_matrix((train_size, embed_dim), dtype=np.float32)
    vx = csr_matrix((val_size, embed_dim), dtype=np.float32)
    tx = csr_matrix((test_size, embed_dim), dtype=np.float32)
    allx = csr_matrix((train_size + vocab_size, embed_dim), dtype=np.float32)

    # Label matrices - check if dataset.ohe_labels is sparse or dense
    y = dataset.ohe_labels[metadata["train_indices"]]
    vy = dataset.ohe_labels[metadata["val_indices"]]
    ty = dataset.ohe_labels[metadata["test_indices"]]

    # Handle both sparse and dense label arrays
    if hasattr(y, "toarray"):
        y_array = y.toarray()
    else:
        y_array = y

    ally = np.concatenate(
        [y_array, np.zeros((vocab_size, len(metadata["label_classes"])))]
    )

    logging.info(
        f"Created matrices: x={x.shape}, y={y.shape}, vx={vx.shape}, vy={vy.shape}"
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
    adj_matrix, data_matrices, metadata, dataset_name, doclevel, testunklar=False
):
    """Save all graph files to organized directories."""
    logging.info(f"Saving graph files for {dataset_name}_{doclevel}...")

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

    logging.info(f"✅ Graph files saved at {output_dir}")
    return output_dir
