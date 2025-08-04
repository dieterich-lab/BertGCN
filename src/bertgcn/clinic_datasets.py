"""
Clinical datasets for BertGCN.

Handles loading and preprocessing of clinical text datasets.
"""

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Try to import stopwords, but don't fail if it's not available
try:
    from nltk.corpus import stopwords

    STOPWORDS = set(stopwords.words("german"))
except:
    logging.warning("NLTK stopwords not available. Using a minimal set.")
    STOPWORDS = {
        "der",
        "die",
        "das",
        "ein",
        "eine",
        "und",
        "in",
        "im",
        "mit",
        "für",
        "von",
        "zu",
        "auf",
        "ist",
        "sind",
        "war",
        "waren",
    }


class CleanClinicDataset:
    """Dataset for clinical documents with text cleaning."""

    def __init__(
        self, tokenizer: AutoTokenizer, doclevel: str = "letter", clean: bool = True
    ):
        """Initialize dataset with clinical documents."""
        self.tokenizer = tokenizer
        self.doclevel = doclevel
        self.clean = clean

        logging.info(f"Initializing dataset for {doclevel} with clean={clean}")

        # Load and prepare data
        self._load_data()

    def _load_data(self):
        """Load clinical data."""
        # In a real implementation, we would load data from files
        # For this minimal implementation, we'll create synthetic data

        logging.info("Creating synthetic clinical data for testing")

        # Generate synthetic data
        num_samples = 100

        # Sample clinical texts
        clinical_texts = [
            f"Patient presents with symptoms of {condition}. "
            + f"Medical history includes {random.choice(['diabetes', 'hypertension', 'asthma'])}"
            + f" and {random.choice(['COPD', 'coronary artery disease', 'arthritis'])}."
            for condition in [
                "pneumonia",
                "bronchitis",
                "influenza",
                "gastritis",
                "appendicitis",
                "diabetes",
            ]
            * (num_samples // 6 + 1)
        ][:num_samples]

        # Sample labels
        labels = ["positive", "negative", "uncertain"] * (num_samples // 3 + 1)
        labels = labels[:num_samples]

        # Create encoder for labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)

        # Convert labels to numeric
        label_ids = self.label_encoder.transform(labels)

        # Create one-hot encoding
        num_classes = len(self.label_encoder.classes_)
        self.ohe_labels = np.zeros((len(labels), num_classes))
        for i, label_id in enumerate(label_ids):
            self.ohe_labels[i, label_id] = 1

        # Store data
        self.texts = clinical_texts
        self.labels = label_ids
        self.class_names = self.label_encoder.classes_

        # Tokenize texts
        logging.info("Tokenizing texts...")
        self.tokenized = self._tokenize_texts()

        logging.info(
            f"Dataset created with {len(self.texts)} samples and {num_classes} classes"
        )

    def _tokenize_texts(self):
        """Tokenize texts using the BERT tokenizer."""
        # Tokenize all texts
        tokenized = self.tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return tokenized

    def __len__(self):
        """Get dataset length."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """Get item for PyTorch compatibility."""
        # Convert numpy int64 to regular Python int
        if isinstance(idx, np.integer):
            idx = int(idx)

        # Return a dictionary with all needed fields
        return {
            "input_ids": self.tokenized["input_ids"][idx],
            "attention_mask": self.tokenized["attention_mask"][idx],
            "label_id": self.labels[idx],
            "med_id": 0,  # Default medication ID (simplified)
        }
