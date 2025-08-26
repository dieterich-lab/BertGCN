from pathlib import Path
from typing import Optional, Union

import nltk
import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, Features, Value
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Download NLTK data
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

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
        file_path: Optional[Union[str, Path]] = None,
    ):
        self.tokenizer = tokenizer
        self.file_path = (
            file_path
            or "/prj/doctoral_letters/MIEdeep/corpus/annotated_gold500/med_indication_all_RF_diag.csv"
        )
        self.doclevel_column = self.DOCLEVEL_TO_COLUMN.get(doclevel, "discharge_letter")
        self.clean = clean

        # Load and prepare dataset
        self.dataset = self._create_dataset(dev_limit)
        self.LE = LabelEncoder()
        self.medsLE = LabelEncoder()

        # Apply preprocessing pipeline
        self.dataset = self._preprocess_dataset()

        # Create properties needed by build_graph.py
        self.texts = [
            self._get_text_from_processed(i) for i in range(len(self.dataset))
        ]

        # Create one-hot encoded labels
        self.ohe = OneHotEncoder(sparse_output=False)
        self.ohe_labels = self.ohe.fit_transform(
            np.array(
                [self.dataset[i]["labels"] for i in range(len(self.dataset))]
            ).reshape(-1, 1)
        )

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
        # First get the raw data before encoding
        raw_data = []
        for i in range(len(self.dataset)):
            row = self.dataset[i]
            raw_data.append(row)

        # Encode labels
        combined_labels = [
            f"{row['medication_type']}_{row['label']}" for row in raw_data
        ]

        label_ids = self.LE.fit_transform(combined_labels)
        med_ids = self.medsLE.fit_transform(
            [row["medication_name"] for row in raw_data]
        )

        # Add encoded labels to dataset
        self.dataset = self.dataset.add_column("labels", label_ids)
        self.dataset = self.dataset.add_column("med_id", med_ids)

        # Add processed text
        processed_texts = [self._get_text(row) for row in raw_data]
        self.dataset = self.dataset.add_column("processed_text", processed_texts)

        # Tokenize with caching
        self.dataset = self.dataset.map(
            self._tokenize_function,
            batched=True,
            load_from_cache_file=False,  # Force re-processing
            remove_columns=[
                col
                for col in self.dataset.column_names
                if col
                not in [
                    "labels",
                    "med_id",
                    "input_ids",
                    "attention_mask",
                    "processed_text",
                ]
            ],
        )

        return self.dataset

    def _get_text(self, row: dict) -> str:
        """Construct and clean text from row data."""
        text = row[self.doclevel_column]

        text = f"Medikament {row['medication_name']} & {text}"

        if self.clean:
            text = " ".join(
                word for word in text.split() if word.lower() not in STOPWORDS
            )

        return text

    def _get_text_from_processed(self, idx: int) -> str:
        """Get processed text from dataset by index."""
        return self.dataset[idx]["processed_text"]

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
            "labels": item["labels"],
            "meds": item["med_id"],
        }

    def to_torch_dataset(self):
        """Convert to PyTorch format with proper tensor types."""
        self.dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels", "med_id"]
        )
        return self.dataset

    def __str__(self) -> str:
        return self.file_path.stem
