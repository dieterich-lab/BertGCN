#!/usr/bin/env python3
"""
Unit tests for graph building functionality.
"""

import os
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from transformers import AutoTokenizer, BertTokenizer

from bertgcn.data_manager import create_data_matrices, save_graph_files
from bertgcn.datasets import CleanClinicDataset

# Import the modules from bertgcn package
from bertgcn.graph_builder import build_graph
from bertgcn.utils import get_paths


class TestGraphBuilding(unittest.TestCase):
    """Test cases for graph building."""

    def setUp(self):
        """Set up test environment."""
        # Use a small test model
        self.model_name = "bert-base-uncased"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.doclevel = "letter"
        self.testunklar = False

    def test_build_graph(self):
        """Test building a document-word graph."""
        # Build graph
        adj_matrix, metadata, dataset = build_graph(self.doclevel, self.testunklar)

        # Check graph properties
        self.assertIsNotNone(adj_matrix)
        self.assertIsNotNone(metadata)
        self.assertIsNotNone(dataset)

        # Check that it's a proper adjacency matrix
        self.assertIsInstance(adj_matrix, csr_matrix)

        # Check metadata contains required keys
        required_keys = [
            "node_size",
            "doc_size",
            "vocab_size",
            "embed_dim",
            "train_size",
            "val_size",
            "test_size",
            "label_classes",
        ]
        for key in required_keys:
            self.assertIn(key, metadata)

        # Check dataset attributes
        self.assertTrue(hasattr(dataset, "texts"))
        self.assertTrue(hasattr(dataset, "labels"))
        self.assertTrue(hasattr(dataset, "ohe_labels"))

    def test_create_data_matrices(self):
        """Test creating data matrices from graph metadata."""
        # First build the graph
        adj_matrix, metadata, dataset = build_graph(self.doclevel, self.testunklar)

        # Create data matrices
        data_matrices = create_data_matrices(dataset, metadata, metadata["embed_dim"])

        # Check matrices
        required_matrices = ["x", "y", "vx", "vy", "tx", "ty", "allx", "ally"]
        for key in required_matrices:
            self.assertIn(key, data_matrices)

        # Check shapes
        self.assertEqual(data_matrices["x"].shape[0], metadata["train_size"])
        self.assertEqual(data_matrices["vx"].shape[0], metadata["val_size"])
        self.assertEqual(data_matrices["tx"].shape[0], metadata["test_size"])

    def test_save_graph_files(self):
        """Test saving graph files to disk."""
        # Build graph and create matrices
        adj_matrix, metadata, dataset = build_graph(self.doclevel, self.testunklar)
        data_matrices = create_data_matrices(dataset, metadata, metadata["embed_dim"])

        # Save graph files
        dataset_name = f"test_medindcls_{self.doclevel}"
        output_dir = save_graph_files(
            adj_matrix,
            data_matrices,
            metadata,
            dataset_name,
            self.doclevel,
            self.testunklar,
        )

        # Check output directory exists
        self.assertTrue(output_dir.exists())

        # Check base filename pattern
        base_name = f"ind.{dataset_name}_{self.doclevel}"

        # Check that files were created
        files_to_check = [
            f"{base_name}.{suffix}"
            for suffix in ["x", "y", "vx", "vy", "tx", "ty", "adj", "metadata.pkl"]
        ]

        for filename in files_to_check:
            file_path = output_dir / filename
            self.assertTrue(file_path.exists(), f"File {filename} not created")

        # Clean up test files
        for filename in files_to_check:
            file_path = output_dir / filename
            if file_path.exists():
                try:
                    os.remove(file_path)
                except:
                    pass


if __name__ == "__main__":
    unittest.main()
