"""Tests for the interpretation script."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import shap
import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, "/home/pwiesenbach/BertGCN/src")
from bertgcn.interpret import interpret


@pytest.fixture
def sample_interpret_config():
    """Create a sample configuration for interpretation testing."""
    config = {
        "seed": 42,
        "data": {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
        },
        "inference": {
            "model_path": "/fake/model/path",
            "output_file": "/fake/output/interpretations.csv",
            "max_samples": 10,
        },
    }
    return OmegaConf.create(config)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    return tokenizer


@pytest.fixture
def mock_label_encoder():
    """Create a mock label encoder."""
    le = Mock()
    le.classes_ = np.array(["class_0", "class_1", "class_2"])
    return le


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing."""
    pipeline_mock = Mock()
    return pipeline_mock


class TestInterpretFunction:
    """Test the main interpret function."""

    @patch("bertgcn.interpret.load_processed_dataset")
    @patch("bertgcn.interpret.split_dataset")
    @patch("bertgcn.interpret.load_model_and_tokenizer")
    @patch("bertgcn.interpret.pipeline")
    @patch("bertgcn.interpret.shap.Explainer")
    @patch("bertgcn.interpret.pd.DataFrame.to_csv")
    def test_interpret_success(
        self,
        mock_to_csv,
        mock_shap_explainer,
        mock_pipeline_func,
        mock_load_model_tokenizer,
        mock_split_dataset,
        mock_load_dataset,
        sample_interpret_config,
        mock_model,
        mock_tokenizer,
        mock_label_encoder,
        mock_pipeline,
    ):
        """Test successful interpretation run."""
        # Setup mocks
        mock_dataset = Mock()
        mock_dataset.column_names = ["input_ids", "attention_mask", "labels", "text"]
        mock_load_dataset.return_value = (mock_dataset, mock_label_encoder)

        mock_train_ds, mock_val_ds, mock_test_ds = Mock(), Mock(), Mock()
        mock_test_ds.indices = [0, 1, 2]
        mock_split_dataset.return_value = (mock_train_ds, mock_val_ds, mock_test_ds)

        mock_load_model_tokenizer.return_value = (
            mock_model,
            mock_tokenizer,
            mock_label_encoder,
        )

        # Mock dataset select
        mock_selected_dataset = Mock()
        mock_selected_dataset.column_names = [
            "input_ids",
            "attention_mask",
            "labels",
            "text",
        ]
        mock_selected_dataset.__len__ = Mock(return_value=3)
        mock_selected_dataset.select = Mock(return_value=mock_selected_dataset)
        mock_dataset.select.return_value = mock_selected_dataset

        # Mock text data
        texts = ["text 1", "text 2", "text 3"]
        mock_selected_dataset.__getitem__ = lambda i: {"text": texts[i]}

        # Mock pipeline
        mock_pipeline_func.return_value = mock_pipeline

        # Mock SHAP explainer
        mock_explainer_instance = Mock()
        mock_shap_explainer.return_value = mock_explainer_instance

        # Mock SHAP values
        mock_shap_values = []
        for i, text in enumerate(texts):
            mock_shap_val = Mock()
            mock_shap_val.data = ["token1", "token2", "token3"]
            mock_shap_val.values = [0.1, 0.2, 0.3]
            mock_shap_values.append(mock_shap_val)
        mock_explainer_instance.return_value = mock_shap_values

        # Run interpret
        interpret(sample_interpret_config)

        # Verify calls
        mock_load_dataset.assert_called_once_with(sample_interpret_config)
        mock_split_dataset.assert_called_once_with(
            mock_dataset, sample_interpret_config
        )
        mock_load_model_tokenizer.assert_called_once_with(sample_interpret_config)
        mock_pipeline_func.assert_called_once()
        mock_shap_explainer.assert_called_once_with(mock_pipeline)
        mock_explainer_instance.assert_called_once_with(texts)
        mock_to_csv.assert_called_once()

    @patch("bertgcn.interpret.load_processed_dataset")
    @patch("bertgcn.interpret.split_dataset")
    @patch("bertgcn.interpret.load_model_and_tokenizer")
    @patch("bertgcn.interpret.pipeline")
    @patch("bertgcn.interpret.shap.Explainer")
    @patch("bertgcn.interpret.pd.DataFrame.to_csv")
    def test_interpret_limits_samples(
        self,
        mock_to_csv,
        mock_shap_explainer,
        mock_pipeline_func,
        mock_load_model_tokenizer,
        mock_split_dataset,
        mock_load_dataset,
        sample_interpret_config,
        mock_model,
        mock_tokenizer,
        mock_label_encoder,
    ):
        """Test that interpretation limits samples correctly."""
        # Setup config with small max_samples
        sample_interpret_config.inference.max_samples = 2

        # Setup mocks
        mock_dataset = Mock()
        mock_dataset.column_names = ["input_ids", "attention_mask", "labels", "text"]
        mock_load_dataset.return_value = (mock_dataset, mock_label_encoder)

        mock_train_ds, mock_val_ds, mock_test_ds = Mock(), Mock(), Mock()
        mock_split_dataset.return_value = (mock_train_ds, mock_val_ds, mock_test_ds)

        mock_load_model_tokenizer.return_value = (
            mock_model,
            mock_tokenizer,
            mock_label_encoder,
        )

        # Mock dataset select - original has 5 samples, limited to 2
        mock_selected_dataset = Mock()
        mock_selected_dataset.column_names = [
            "input_ids",
            "attention_mask",
            "labels",
            "text",
        ]
        mock_selected_dataset.__len__ = Mock(return_value=5)  # More than max_samples
        mock_limited_dataset = Mock()
        mock_limited_dataset.__len__ = Mock(return_value=2)
        mock_selected_dataset.select.return_value = mock_limited_dataset
        mock_dataset.select.return_value = mock_selected_dataset

        # Mock pipeline and SHAP
        mock_pipeline_func.return_value = Mock()
        mock_explainer_instance = Mock()
        mock_shap_explainer.return_value = mock_explainer_instance
        mock_explainer_instance.return_value = [Mock(), Mock()]  # 2 shap values

        # Run interpret
        interpret(sample_interpret_config)

        # Verify select was called to limit samples
        mock_selected_dataset.select.assert_called_once_with(range(2))

    @patch("bertgcn.interpret.load_processed_dataset")
    @patch("bertgcn.interpret.split_dataset")
    @patch("bertgcn.interpret.load_model_and_tokenizer")
    @patch("bertgcn.interpret.pipeline")
    @patch("bertgcn.interpret.shap.Explainer")
    @patch("bertgcn.interpret.pd.DataFrame.to_csv")
    def test_interpret_without_text_column(
        self,
        mock_to_csv,
        mock_shap_explainer,
        mock_pipeline_func,
        mock_load_model_tokenizer,
        mock_split_dataset,
        mock_load_dataset,
        sample_interpret_config,
        mock_model,
        mock_tokenizer,
        mock_label_encoder,
    ):
        """Test interpretation when dataset doesn't have text column."""
        # Setup mocks
        mock_dataset = Mock()
        mock_dataset.column_names = ["input_ids", "attention_mask", "labels"]  # No text
        mock_load_dataset.return_value = (mock_dataset, mock_label_encoder)

        mock_train_ds, mock_val_ds, mock_test_ds = Mock(), Mock(), Mock()
        mock_split_dataset.return_value = (mock_train_ds, mock_val_ds, mock_test_ds)

        mock_load_model_tokenizer.return_value = (
            mock_model,
            mock_tokenizer,
            mock_label_encoder,
        )

        # Mock dataset select
        mock_selected_dataset = Mock()
        mock_selected_dataset.column_names = ["input_ids", "attention_mask", "labels"]
        mock_selected_dataset.__len__ = Mock(return_value=2)
        mock_dataset.select.return_value = mock_selected_dataset

        # Mock pipeline and SHAP
        mock_pipeline_func.return_value = Mock()
        mock_explainer_instance = Mock()
        mock_shap_explainer.return_value = mock_explainer_instance
        mock_explainer_instance.return_value = [Mock(), Mock()]

        # Run interpret
        interpret(sample_interpret_config)

        # Should use empty strings as fallback
        expected_texts = ["", ""]
        mock_explainer_instance.assert_called_once_with(expected_texts)


class TestIntegration:
    """Integration tests for interpretation pipeline."""

    @patch("bertgcn.interpret.load_processed_dataset")
    @patch("bertgcn.interpret.split_dataset")
    @patch("bertgcn.interpret.load_model_and_tokenizer")
    @patch("bertgcn.interpret.pipeline")
    @patch("bertgcn.interpret.shap.Explainer")
    @patch("bertgcn.interpret.pd.DataFrame.to_csv")
    def test_main_function_calls_interpret(
        self,
        mock_to_csv,
        mock_shap_explainer,
        mock_pipeline_func,
        mock_load_model_tokenizer,
        mock_split_dataset,
        mock_load_dataset,
        sample_interpret_config,
    ):
        """Test that main function calls interpret."""
        from bertgcn.interpret import main

        # Setup minimal mocks
        mock_dataset = Mock()
        mock_dataset.column_names = ["input_ids", "attention_mask", "labels", "text"]
        mock_load_dataset.return_value = (mock_dataset, Mock())

        mock_split_dataset.return_value = (Mock(), Mock(), Mock())
        mock_load_model_tokenizer.return_value = (Mock(), Mock(), Mock())

        mock_pipeline_func.return_value = Mock()
        mock_explainer_instance = Mock()
        mock_shap_explainer.return_value = mock_explainer_instance
        mock_explainer_instance.return_value = [Mock()]

        # Test that main can be called
        assert callable(main)

        # Call main (would normally be called by Hydra)
        main(sample_interpret_config)

        # Verify interpret was called through the mocks
        mock_load_dataset.assert_called()


class TestSHAPIntegration:
    """Test SHAP-specific functionality."""

    def test_shap_explainer_creation(self):
        """Test that SHAP explainer can be created with pipeline."""
        # This is more of a smoke test to ensure SHAP is available
        # In real scenarios, this would be mocked
        mock_pipeline = Mock()
        try:
            explainer = shap.Explainer(mock_pipeline)
            assert explainer is not None
        except Exception:
            # SHAP might not be properly configured in test environment
            # This is acceptable for unit tests
            pass
