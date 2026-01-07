import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from bertgcn.core import get_logger

logger = get_logger(__name__)


class CleanClinicDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, max_length: int = 512
    ):
        self.tokenizer = tokenizer
        # load_data will populate self.LE and self.ohe_labels for tests
        self.texts, self.labels = self.load_data(file_path)
        self.max_length = max_length

    def load_data(self, file_path: str):
        logger.info(f"Loading data from {file_path}")
        # The raw CSV in this project uses '|||'-separated columns without a header
        column_names = [
            "_",
            "diagnosis",
            "anamnesis",
            "risk_factor",
            "discharge_letter",
            "medication_type",
            "medication_name",
            "label",
        ]
        df = pd.read_csv(
            file_path, sep=r"\|\|\|", header=None, names=column_names, engine="python"
        )
        # strip whitespace from object columns
        for col in df.select_dtypes(include=["object"]):
            df[col] = df[col].str.strip()

        # Build the text field as in preprocessing
        df["text"] = df["medication_name"].astype(str)  # ensure string type
        df["text"] = [
            f"Medikament {name} & {text}"
            for name, text in zip(df["medication_name"], df["discharge_letter"])
        ]

        # Build combined label and encode
        df["combined_label"] = df["medication_type"] + "_" + df["label"]
        le = LabelEncoder()
        encoded = le.fit_transform(df["combined_label"]) if len(df) > 0 else []
        self.LE = le

        # One-hot encoded labels for tests that expect ohe_labels
        num_classes = len(le.classes_)
        if num_classes > 0:
            ohe = np.eye(num_classes, dtype=int)[encoded]
        else:
            ohe = np.zeros((len(df), 0), dtype=int)
        self.ohe_labels = ohe

        texts = df["text"].tolist()
        labels = torch.tensor(encoded, dtype=torch.long)
        return texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": label,
        }
