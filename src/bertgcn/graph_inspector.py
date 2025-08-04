"""
Graph inspection and validation utilities for BertGCN.
"""

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.sparse import csr_matrix, load_npz


def load_graph(graph_dir: Path, graph_name: str) -> Dict[str, Any]:
    """Load a complete graph from disk."""
    graph_files = {}

    # Load adjacency matrix
    adj_file = graph_dir / f"ind.{graph_name}.adj.npz"
    if adj_file.exists():
        graph_files["adj_matrix"] = load_npz(adj_file)

    # Load metadata
    metadata_file = graph_dir / f"ind.{graph_name}.metadata"
    if metadata_file.exists():
        with open(metadata_file, "rb") as f:
            graph_files["metadata"] = pickle.load(f)

    # Load vocabulary
    vocab_file = graph_dir / f"ind.{graph_name}.vocab"
    if vocab_file.exists():
        with open(vocab_file, "rb") as f:
            graph_files["vocab_data"] = pickle.load(f)

    # Load feature matrices
    for split in ["x", "vx", "tx"]:
        split_file = graph_dir / f"ind.{graph_name}.{split}"
        if split_file.exists():
            with open(split_file, "rb") as f:
                graph_files[f"{split}_features"] = pickle.load(f)

    # Load texts
    texts_file = graph_dir / f"ind.{graph_name}.texts"
    if texts_file.exists():
        with open(texts_file, "rb") as f:
            graph_files["texts"] = pickle.load(f)

    return graph_files


def validate_graph(graph_dir: Path, graph_name: str) -> bool:
    """Validate that a graph has all required components."""
    required_files = [
        f"ind.{graph_name}.adj.npz",
        f"ind.{graph_name}.metadata",
        f"ind.{graph_name}.x",
        f"ind.{graph_name}.vx",
        f"ind.{graph_name}.tx",
    ]

    missing_files = []
    for file_name in required_files:
        file_path = graph_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    try:
        # Load and validate adjacency matrix
        adj_matrix = load_npz(graph_dir / f"ind.{graph_name}.adj.npz")
        if not isinstance(adj_matrix, csr_matrix):
            print("❌ Adjacency matrix is not in CSR format")
            return False

        # Load and validate metadata
        with open(graph_dir / f"ind.{graph_name}.metadata", "rb") as f:
            metadata = pickle.load(f)

        expected_keys = ["num_docs", "num_words", "total_nodes", "num_classes"]
        for key in expected_keys:
            if key not in metadata:
                print(f"❌ Missing metadata key: {key}")
                return False

        # Validate matrix dimensions
        if adj_matrix.shape[0] != metadata["total_nodes"]:
            print(
                f"❌ Adjacency matrix dimension mismatch: {adj_matrix.shape[0]} != {metadata['total_nodes']}"
            )
            return False

        print("✅ Graph validation passed")
        return True

    except Exception as e:
        print(f"❌ Graph validation failed: {e}")
        return False


def inspect_graph(graph_dir: Path, graph_name: str) -> None:
    """Provide detailed inspection of a graph."""
    print(f"Inspecting graph: {graph_name}")
    print("=" * 50)

    if not validate_graph(graph_dir, graph_name):
        return

    # Load all components
    graph_data = load_graph(graph_dir, graph_name)

    # Display basic info
    metadata = graph_data["metadata"]
    adj_matrix = graph_data["adj_matrix"]

    print(f"📊 Graph Statistics:")
    print(f"  Total nodes: {metadata['total_nodes']}")
    print(f"  Documents: {metadata['num_docs']}")
    print(f"  Words: {metadata['num_words']}")
    print(f"  Classes: {metadata['num_classes']}")
    print(f"  Adjacency matrix shape: {adj_matrix.shape}")
    print(f"  Non-zero entries: {adj_matrix.nnz}")
    print(
        f"  Sparsity: {1 - adj_matrix.nnz / (adj_matrix.shape[0] * adj_matrix.shape[1]):.4f}"
    )

    # Data split info
    print(f"\n📈 Data Split:")
    print(f"  Training: {metadata['train_size']}")
    print(f"  Validation: {metadata['val_size']}")
    print(f"  Test: {metadata['test_size']}")

    # Vocabulary info
    if "vocab_data" in graph_data:
        vocab_data = graph_data["vocab_data"]
        print(f"\n📚 Vocabulary:")
        print(f"  Size: {len(vocab_data['vocab'])}")
        print(f"  Sample words: {vocab_data['vocab'][:10]}")

    # Class distribution
    if "x_features" in graph_data:
        train_labels = graph_data["x_features"]
        class_counts = np.sum(train_labels, axis=0)
        print(f"\n🏷️  Training Class Distribution:")
        for i, (class_name, count) in enumerate(
            zip(metadata["class_names"], class_counts)
        ):
            print(f"  {class_name}: {count} ({count/len(train_labels)*100:.1f}%)")


if __name__ == "__main__":
    import sys

    from .config import get_paths

    paths = get_paths()

    if len(sys.argv) > 1:
        graph_name = sys.argv[1]
    else:
        graph_name = "medindcls_letter"

    graph_dir = paths["graphs"] / graph_name

    if not graph_dir.exists():
        print(f"❌ Graph directory not found: {graph_dir}")
        sys.exit(1)

    inspect_graph(graph_dir, graph_name)
