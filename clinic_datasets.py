from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, Features, Value
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

STOPWORDS = set(stopwords.words("german"))


class CleanClinicDataset:
    """
    A modern dataset class for clinical discharge letters using Hugging Face datasets.
    Efficiently handles loading, preprocessing, and tokenization with caching support.
    """

    COLUMN_NAMES = [
        "_",
        "diagnosis",
        "anamnesis",
        "risk_factor",
        "discharge_letter",
        "medication_type",
        "medication_name",
        "label",
    ]

    DOCLEVEL_TO_COLUMN = {
        "letter": "discharge_letter",
        "diagnosis": "diagnosis",
        "riskfactor": "risk_factor",
        "anamnesis": "anamnesis",
    }

    def __init__(
        self,
        tokenizer,
        doclevel: str = "letter",
        dev_limit: Optional[int] = None,
        clean: bool = True,
        nomeds: bool = False,
    ):
        self.tokenizer = tokenizer
        self.file_path = "/prj/doctoral_letters/MIEdeep/corpus/annotated_gold500/med_indication_all_RF_diag.csv"
        self.doclevel_column = self.DOCLEVEL_TO_COLUMN.get(doclevel, "discharge_letter")
        self.clean = clean
        self.nomeds = nomeds

        # Load and prepare dataset
        self.dataset = self._create_dataset(dev_limit)
        self.LE = LabelEncoder()
        self.medsLE = LabelEncoder()

        # Apply preprocessing pipeline
        self.dataset = self._preprocess_dataset()

    def _create_dataset(self, dev_limit: Optional[int]) -> Dataset:
        """Load data and create HuggingFace Dataset."""
        df = pd.read_csv(
            self.file_path,
            sep=r"\|\|\|",
            header=None,
            names=self.COLUMN_NAMES,
            engine="python",
        )

        if dev_limit:
            df = df.head(dev_limit)

        # Clean string columns
        for col in df.select_dtypes(include=["object"]):
            df[col] = df[col].str.strip()

        return Dataset.from_pandas(df)

    def _preprocess_dataset(self) -> Dataset:
        """Apply all preprocessing steps to the dataset."""
        # Encode labels
        combined_labels = [
            f"{row['medication_type']}_{row['label']}" for row in self.dataset
        ]

        label_ids = self.LE.fit_transform(combined_labels)
        med_ids = self.medsLE.fit_transform(self.dataset["medication_name"])

        # Add encoded labels to dataset
        self.dataset = self.dataset.add_column("label_id", label_ids)
        self.dataset = self.dataset.add_column("med_id", med_ids)

        # Add processed text
        processed_texts = [self._get_text(row) for row in self.dataset]
        self.dataset = self.dataset.add_column("processed_text", processed_texts)

        # Add raw texts for graph building (needed for vocabulary extraction)
        raw_texts = [self._get_raw_text(row) for row in self.dataset]
        self.dataset = self.dataset.add_column("raw_text", raw_texts)

        # Tokenize with caching
        self.dataset = self.dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=[
                col
                for col in self.dataset.column_names
                if col
                not in ["label_id", "med_id", "input_ids", "attention_mask", "raw_text"]
            ],
        )

        return self.dataset

    def _get_text(self, row: dict) -> str:
        """Construct and clean text from row data."""
        text = row[self.doclevel_column]

        if not self.nomeds:
            text = f"Medikament {row['medication_name']} & {text}"

        if self.clean:
            text = " ".join(
                word for word in text.split() if word.lower() not in STOPWORDS
            )

        return text

    def _get_raw_text(self, row: dict) -> str:
        """Get raw text for graph building (no cleaning, just medication concatenation)."""
        text = row[self.doclevel_column]

        if not self.nomeds:
            text = f"Medikament {row['medication_name']} & {text}"

        return text

    def _tokenize_function(self, examples: dict) -> dict:
        """Tokenize texts efficiently in batches."""
        return self.tokenizer(
            examples["processed_text"],
            truncation=True,
            padding="max_length",
            return_tensors=None,  # Return lists for datasets compatibility
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """Get item for PyTorch compatibility."""
        item = self.dataset[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["label_id"],
            "meds": item["med_id"],
        }

    def to_torch_dataset(self):
        """Convert to PyTorch format with proper tensor types."""
        self.dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label_id", "med_id"]
        )
        return self.dataset

    @property
    def texts(self) -> List[str]:
        """Get raw texts for graph building."""
        return self.dataset["raw_text"]

    @property
    def ohe_labels(self) -> np.ndarray:
        """Get one-hot encoded labels."""
        if not hasattr(self, "_ohe_labels_cache"):
            from sklearn.preprocessing import LabelBinarizer

            lb = LabelBinarizer()
            self._ohe_labels_cache = lb.fit_transform(self.dataset["label_id"])
        return self._ohe_labels_cache

    def __str__(self) -> str:
        return self.file_path.stem
