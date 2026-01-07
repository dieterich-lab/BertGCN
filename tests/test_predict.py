"""Tests for the prediction script."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, "/home/pwiesenbach/BertGCN/src")
from bertgcn.predict import load_model_and_tokenizer, predict


@pytest.fixture
def sample_predict_config():
    """Create a sample configuration for prediction testing."""
    config = {
        "seed": 42,
        "data": {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
        },
        "inference": {
            "model_path": "/fake/model/path",
            "output_file": "/fake/output/predictions.csv",
        },
    }
    return OmegaConf.create(config)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.eval = Mock()
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
    le.inverse_transform = Mock(return_value=["class_0", "class_1"])
    return le


class TestLoadModelAndTokenizer:
    """Test model and tokenizer loading functions."""

    @patch("bertgcn.predict.AutoTokenizer.from_pretrained")
    @patch("bertgcn.predict.AutoModelForSequenceClassification.from_pretrained")
    @patch("bertgcn.predict.joblib.load")
    @patch("pathlib.Path.exists")
    def test_load_model_and_tokenizer_success(
        self,
        mock_exists,
        mock_joblib_load,
        mock_model_from_pretrained,
        mock_tokenizer_from_pretrained,
        sample_predict_config,
        mock_model,
        mock_tokenizer,
        mock_label_encoder,
    ):
        """Test successful loading of model, tokenizer, and label encoder."""
        mock_exists.return_value = True
        mock_model_from_pretrained.return_value = mock_model
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_joblib_load.return_value = mock_label_encoder

        model, tokenizer, le = load_model_and_tokenizer(sample_predict_config)

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        assert le == mock_label_encoder

    @patch("pathlib.Path.exists")
    def test_load_model_path_not_exists(self, mock_exists, sample_predict_config):
        """Test error when model path doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(SystemExit):
            load_model_and_tokenizer(sample_predict_config)

    @patch("pathlib.Path.exists")
    @patch("bertgcn.predict.Path.__truediv__")
    def test_load_label_encoder_not_exists(
        self, mock_truediv, mock_exists, sample_predict_config
    ):
        """Test error when label encoder doesn't exist."""
        mock_exists.side_effect = [True, False]  # model path exists, le doesn't

        with pytest.raises(SystemExit):
            load_model_and_tokenizer(sample_predict_config)


class TestPredictFunction:
    """Test the main predict function."""

    @patch("bertgcn.predict.load_processed_dataset")
    @patch("bertgcn.predict.split_dataset")
    @patch("bertgcn.predict.load_model_and_tokenizer")
    @patch("bertgcn.predict.Trainer")
    @patch("bertgcn.predict.DataCollatorWithPadding")
    @patch("bertgcn.predict.pd.DataFrame.to_csv")
    def test_predict_success(
        self,
        mock_to_csv,
        mock_data_collator,
        mock_trainer,
        mock_load_model_tokenizer,
        mock_split_dataset,
        mock_load_dataset,
        sample_predict_config,
        mock_model,
        mock_tokenizer,
        mock_label_encoder,
    ):
        """Test successful prediction run."""
        # Setup mocks
        mock_dataset = Mock()
        mock_dataset.column_names = ["input_ids", "attention_mask", "labels"]
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
        mock_selected_dataset.column_names = ["input_ids", "attention_mask", "labels"]
        mock_selected_dataset.__len__ = Mock(return_value=3)
        mock_selected_dataset.__getitem__ = Mock(return_value={"labels": 0})
        mock_dataset.select.return_value = mock_selected_dataset

        # Mock trainer and predictions
        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance
        mock_predictions = Mock()
        mock_predictions.predictions = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        mock_trainer_instance.predict.return_value = mock_predictions

        # Run predict
        predict(sample_predict_config)

        # Verify calls
        mock_load_dataset.assert_called_once_with(sample_predict_config)
        mock_split_dataset.assert_called_once_with(mock_dataset, sample_predict_config)
        mock_load_model_tokenizer.assert_called_once_with(sample_predict_config)
        mock_trainer.assert_called_once()
        mock_trainer_instance.predict.assert_called_once()
        mock_to_csv.assert_called_once()

    @patch("bertgcn.predict.load_processed_dataset")
    @patch("bertgcn.predict.split_dataset")
    @patch("bertgcn.predict.load_model_and_tokenizer")
    def test_predict_without_labels(
        self,
        mock_load_model_tokenizer,
        mock_split_dataset,
        mock_load_dataset,
        sample_predict_config,
        mock_model,
        mock_tokenizer,
        mock_label_encoder,
    ):
        """Test prediction when dataset doesn't have labels."""
        # Setup mocks
        mock_dataset = Mock()
        mock_dataset.column_names = ["input_ids", "attention_mask"]  # No labels
        mock_load_dataset.return_value = (mock_dataset, mock_label_encoder)

        mock_train_ds, mock_val_ds, mock_test_ds = Mock(), Mock(), Mock()
        mock_split_dataset.return_value = (mock_train_ds, mock_val_ds, mock_test_ds)

        mock_load_model_tokenizer.return_value = (
            mock_model,
            mock_tokenizer,
            mock_label_encoder,
        )

        # Mock trainer and predictions
        with patch("bertgcn.predict.Trainer") as mock_trainer, patch(
            "bertgcn.predict.DataCollatorWithPadding"
        ), patch("bertgcn.predict.pd.DataFrame.to_csv") as mock_to_csv:

            mock_trainer_instance = Mock()
            mock_trainer.return_value = mock_trainer_instance
            mock_predictions = Mock()
            mock_predictions.predictions = np.array([[0.1, 0.9], [0.8, 0.2]])
            mock_trainer_instance.predict.return_value = mock_predictions

            # Run predict
            predict(sample_predict_config)

            # Verify CSV was still called (without true labels)
            mock_to_csv.assert_called_once()


class TestIntegration:
    """Integration tests for prediction pipeline."""

    @patch("bertgcn.predict.load_processed_dataset")
    @patch("bertgcn.predict.split_dataset")
    @patch("bertgcn.predict.load_model_and_tokenizer")
    @patch("bertgcn.predict.Trainer")
    @patch("bertgcn.predict.DataCollatorWithPadding")
    @patch("bertgcn.predict.pd.DataFrame.to_csv")
    def test_main_function_calls_predict(
        self,
        mock_to_csv,
        mock_data_collator,
        mock_trainer,
        mock_load_model_tokenizer,
        mock_split_dataset,
        mock_load_dataset,
        sample_predict_config,
    ):
        """Test that main function calls predict."""
        from bertgcn.predict import main

        # Setup minimal mocks
        mock_dataset = Mock()
        mock_dataset.column_names = ["input_ids", "attention_mask", "labels"]
        mock_load_dataset.return_value = (mock_dataset, Mock())

        mock_split_dataset.return_value = (Mock(), Mock(), Mock())
        mock_load_model_tokenizer.return_value = (Mock(), Mock(), Mock())

        mock_trainer_instance = Mock()
        mock_trainer.return_value = mock_trainer_instance
        mock_predictions = Mock()
        mock_predictions.predictions = np.array([[0.1, 0.9]])
        mock_trainer_instance.predict.return_value = mock_predictions

        # Test that main can be called
        assert callable(main)

        # Call main (would normally be called by Hydra)
        main(sample_predict_config)

        # Verify predict was called through the mocks
        mock_load_dataset.assert_called()
