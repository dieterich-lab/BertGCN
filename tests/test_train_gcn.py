"""Tests for the minimal BERT-GCN training script."""

import shutil

# Import the functions we want to test
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import LabelEncoder
from torch_geometric import nn as pyg_nn

sys.path.insert(0, "/home/pwiesenbach/BertGCN/src")
from bertgcn.train_gcn import SimpleGCN, evaluate, load_processed_dataset, train_epoch


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    config = {
        "seed": 42,
        "mlflow_experiment_name": "test_experiment",
        "data": {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
        },
        "model": {
            "n_features": 16,
            "n_hidden": 32,
            "dropout": 0.5,
        },
        "training": {
            "lr": 0.01,
            "epochs": 2,
        },
    }
    return OmegaConf.create(config)


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    # Create sample data
    n_samples = 100
    data = {
        "input_ids": [torch.randint(0, 1000, (512,)) for _ in range(n_samples)],
        "attention_mask": [torch.ones(512, dtype=torch.long) for _ in range(n_samples)],
        "labels": torch.randint(0, 5, (n_samples,)).tolist(),
        "med_id": torch.randint(0, 3, (n_samples,)).tolist(),
    }
    return Dataset.from_dict(data)


@pytest.fixture
def mock_label_encoder():
    """Create a mock label encoder."""
    le = LabelEncoder()
    le.classes_ = np.array(["class_0", "class_1", "class_2", "class_3", "class_4"])
    return le


class TestSimpleGCN:
    """Test the SimpleGCN model."""

    def test_forward_pass(self):
        """Test forward pass of SimpleGCN."""
        n_features, n_hidden, n_classes = 16, 32, 5
        n_nodes = 10

        model = SimpleGCN(n_features, n_hidden, n_classes)
        x = torch.randn(n_nodes, n_features)
        # Create a simple edge_index (COO format)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        output = model(x, edge_index)

        assert output.shape == (n_nodes, n_classes)
        # Check that output is log probabilities (negative values)
        assert torch.all(output <= 0)


class TestDataFunctions:
    """Test data loading and processing functions."""

    @patch("bertgcn.train_gcn.load_from_disk")
    @patch("bertgcn.train_gcn.joblib.load")
    @patch("bertgcn.train_gcn.get_original_cwd")
    def test_load_processed_dataset(
        self, mock_cwd, mock_joblib_load, mock_load_from_disk, sample_config
    ):
        """Test loading processed dataset."""
        mock_cwd.return_value = "/fake/path"
        mock_dataset = Mock()
        mock_dataset.column_names = ["input_ids", "attention_mask", "labels", "med_id"]
        mock_load_from_disk.return_value = mock_dataset
        mock_joblib_load.return_value = Mock()

        with patch("pathlib.Path.exists", return_value=True):
            dataset, le = load_processed_dataset(sample_config)

            mock_load_from_disk.assert_called_once()
            mock_joblib_load.assert_called_once()


class TestTrainingFunctions:
    """Test training and evaluation functions."""

    def test_train_epoch(self, sample_config):
        """Test training for one epoch."""
        n_nodes, n_features, n_classes = 50, 16, 5

        # Create model and data
        model = SimpleGCN(n_features, 32, n_classes)
        data = {
            "features": torch.randn(n_nodes, n_features),
            "adj": torch.randn(n_nodes, n_nodes),
            "labels": torch.randint(0, n_classes, (n_nodes,)),
            "train_mask": torch.rand(n_nodes) > 0.5,
            "val_mask": torch.rand(n_nodes) > 0.5,
            "test_mask": torch.rand(n_nodes) > 0.5,
            "input_ids": torch.randint(0, 1000, (n_nodes, 128)),  # dummy BERT inputs
            "attention_mask": torch.ones(n_nodes, 128),
        }

        optimizer = torch.optim.Adam(model.parameters(), lr=sample_config.training.lr)
        criterion = torch.nn.NLLLoss()
        device = torch.device("cpu")

        loss = train_epoch(model, data, optimizer, criterion, device)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_evaluate(self, sample_config):
        """Test model evaluation."""
        n_nodes, n_features, n_classes = 50, 16, 5

        # Create model and data
        model = SimpleGCN(n_features, 32, n_classes)
        data = {
            "features": torch.randn(n_nodes, n_features),
            "adj": torch.randn(n_nodes, n_nodes),
            "labels": torch.randint(0, n_classes, (n_nodes,)),
            "train_mask": torch.rand(n_nodes) > 0.5,
            "val_mask": torch.rand(n_nodes) > 0.5,
            "test_mask": torch.rand(n_nodes) > 0.5,
            "input_ids": torch.randint(0, 1000, (n_nodes, 128)),  # dummy BERT inputs
            "attention_mask": torch.ones(n_nodes, 128),
        }

        mask = torch.rand(n_nodes) > 0.5
        device = torch.device("cpu")

        acc = evaluate(model, data, mask, device)

        assert isinstance(acc, float)
        assert 0 <= acc <= 1

        assert isinstance(acc, float)
        assert 0 <= acc <= 1


class TestIntegration:
    """Integration tests for the training pipeline."""

    @patch("bertgcn.train_gcn.load_processed_dataset")
    @patch("bertgcn.train_gcn.load_graph_data_from_disk")
    @patch("bertgcn.train_gcn.mlflow")
    def test_main_function_structure(
        self, mock_mlflow, mock_load_graph, mock_load_dataset, sample_config
    ):
        """Test that main function can be called (mocked)."""
        # Setup mocks
        mock_dataset = Mock()
        mock_le = Mock()
        mock_le.classes_ = ["class_0", "class_1"]
        mock_load_dataset.return_value = (mock_dataset, mock_le)

        mock_data = {
            "adj": torch.randn(10, 10),
            "features": torch.randn(10, 16),
            "labels": torch.randint(0, 2, (10,)),
            "train_mask": torch.rand(10) > 0.5,
            "val_mask": torch.rand(10) > 0.5,
            "test_mask": torch.rand(10) > 0.5,
        }
        mock_load_graph.return_value = mock_data

        # Mock MLflow context manager
        mock_run = Mock()
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = Mock(return_value=None)

        # This would normally run the full training, but we're just testing structure
        # In a real test, we'd call main() but that would require more complex mocking
        from bertgcn.train_gcn import main

        # Just test that the function exists and can be imported
        assert callable(main)
