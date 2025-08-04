#!/usr/bin/env python3
"""
Data Module for BertGCN

PyTorch Lightning data module for handling clinical text data.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class ClinicalTextDataset(Dataset):
    """Dataset for clinical text classification."""

    def __init__(self, texts, labels, tokenizer, max_length: int = 512):
        """
        Initialize dataset.

        Args:
            texts: List of text strings
            labels: List of labels
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class DataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for BertGCN."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data module.

        Args:
            config: Data configuration dictionary
        """
        super().__init__()
        self.config = config

        # Configuration parameters
        self.batch_size = config.get("batch_size", 8)
        self.max_length = config.get("max_length", 512)
        self.num_workers = config.get("num_workers", 4)
        self.doclevel = config.get("doclevel", "letter")

        # Data splits
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Tokenizer
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation, and testing."""
        logger.info(f"Setting up data module for stage: {stage}")

        try:
            # For this example, we'll create dummy data
            # In practice, you would load your actual clinical data here

            # Dummy data for demonstration
            dummy_texts = [
                "Patient presents with chest pain and shortness of breath.",
                "No acute distress noted. Vital signs stable.",
                "History of hypertension and diabetes.",
                "Prescribed medication as directed.",
                "Follow-up appointment scheduled.",
            ] * 100  # Replicate to have enough samples

            dummy_labels = [0, 1, 0, 1, 0] * 100  # Binary classification

            # Split data (70% train, 15% val, 15% test)
            total_size = len(dummy_texts)
            train_size = int(0.7 * total_size)
            val_size = int(0.15 * total_size)

            if stage == "fit" or stage is None:
                self.train_dataset = ClinicalTextDataset(
                    texts=dummy_texts[:train_size],
                    labels=dummy_labels[:train_size],
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                )

                self.val_dataset = ClinicalTextDataset(
                    texts=dummy_texts[train_size : train_size + val_size],
                    labels=dummy_labels[train_size : train_size + val_size],
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                )

            if stage == "test" or stage is None:
                self.test_dataset = ClinicalTextDataset(
                    texts=dummy_texts[train_size + val_size :],
                    labels=dummy_labels[train_size + val_size :],
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                )

            logger.info(
                f"Data setup completed. Train: {len(self.train_dataset) if self.train_dataset else 0}, "
                f"Val: {len(self.val_dataset) if self.val_dataset else 0}, "
                f"Test: {len(self.test_dataset) if self.test_dataset else 0}"
            )

        except Exception as e:
            logger.error(f"Failed to setup data: {str(e)}")
            raise

    def train_dataloader(self):
        """Create training data loader."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation data loader."""
        if self.val_dataset is None:
            raise RuntimeError(
                "Validation dataset not initialized. Call setup() first."
            )

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create test data loader."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        """Create prediction data loader."""
        # For prediction, we can use the test dataloader
        return self.test_dataloader()


def create_data_module(config: Dict[str, Any]) -> DataModule:
    """
    Factory function to create a data module.

    Args:
        config: Data configuration

    Returns:
        Configured DataModule instance
    """
    return DataModule(config)
