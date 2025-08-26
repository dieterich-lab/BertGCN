import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from bertgcn.core import get_logger

logger = get_logger(__name__)


class CleanClinicDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.texts, self.labels = self.load_data(file_path)
        self.max_length = max_length

    def load_data(self, file_path: str):
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        texts = df["text"].tolist()
        labels = torch.tensor(df["label"].tolist())
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
