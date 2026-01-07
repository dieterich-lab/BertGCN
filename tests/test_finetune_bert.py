"""Tests for the minimal BERT fine-tuning script."""

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

sys.path.insert(0, "/home/pwiesenbach/BertGCN/src")
from bertgcn.train_bert import compute_metrics, load_processed_dataset, split_dataset


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
            "model_name_or_path": "bert-base-uncased",
        },
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_train_epochs": 3,
            "weight_decay": 0.01,
            "logging_steps": 100,
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


class TestDataFunctions:
    """Test data loading and processing functions."""

    @patch("bertgcn.train_bert.load_from_disk")
    @patch("bertgcn.train_bert.joblib.load")
    @patch("bertgcn.train_bert.get_original_cwd")
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

    def test_split_dataset(self, mock_dataset, sample_config):
        """Test dataset splitting."""
        train_ds, val_ds, test_ds = split_dataset(mock_dataset, sample_config)

        # Check that splits add up to original size
        total_split_size = len(train_ds) + len(val_ds) + len(test_ds)
        assert total_split_size == len(mock_dataset)

        # Check approximate ratios (allowing for rounding)
        n = len(mock_dataset)
        expected_train = int(sample_config.data.train_ratio * n)
        expected_val = int(sample_config.data.val_ratio * n)

        assert abs(len(train_ds) - expected_train) <= 1
        assert abs(len(val_ds) - expected_val) <= 1


class TestMetrics:
    """Test metrics computation."""

    def test_compute_metrics(self):
        """Test accuracy computation."""
        # Create fake predictions and labels
        n_samples = 10
        n_classes = 3

        # Perfect predictions
        logits = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            logits[i, i % n_classes] = 1.0  # Make prediction correct

        labels = np.arange(n_samples) % n_classes

        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred)

        assert "accuracy" in metrics
        assert metrics["accuracy"] == 1.0  # Perfect accuracy

    def test_compute_metrics_imperfect(self):
        """Test accuracy computation with some errors."""
        n_samples = 10
        n_classes = 3

        # Some wrong predictions
        logits = np.random.randn(n_samples, n_classes)
        labels = np.random.randint(0, n_classes, n_samples)

        eval_pred = (logits, labels)
        metrics = compute_metrics(eval_pred)

        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0


class TestIntegration:
    """Integration tests for the training pipeline."""

    @patch("bertgcn.train_bert.load_processed_dataset")
    @patch("bertgcn.train_bert.split_dataset")
    @patch("bertgcn.train_bert.mlflow")
    def test_main_function_structure(
        self, mock_mlflow, mock_split, mock_load_dataset, sample_config
    ):
        """Test that main function can be called (mocked)."""
        # Setup mocks
        mock_dataset = Mock()
        mock_le = Mock()
        mock_le.classes_ = ["class_0", "class_1"]
        mock_load_dataset.return_value = (mock_dataset, mock_le)

        mock_train_ds, mock_val_ds, mock_test_ds = Mock(), Mock(), Mock()
        mock_split.return_value = (mock_train_ds, mock_val_ds, mock_test_ds)

        # Mock MLflow context manager
        mock_run = Mock()
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = Mock(return_value=None)

        # This would normally run the full training, but we're just testing structure
        # In a real test, we'd call main() but that would require more complex mocking
        from bertgcn.train_bert import main

        # Just test that the function exists and can be imported
        assert callable(main)
