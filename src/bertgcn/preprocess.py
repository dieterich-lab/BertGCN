"""
Preprocesses the raw clinical data into a Hugging Face Dataset.

This script handles the one-time task of cleaning, tokenizing, and saving the
dataset in the efficient Arrow format. It also saves the label encoders.
"""

import shutil
from pathlib import Path

import hydra
import joblib
import nltk
import pandas as pd
from datasets import Dataset
from nltk.corpus import stopwords
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

from bertgcn.core import get_logger

logger = get_logger(__name__)

# Ensure NLTK stopwords are available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
STOPWORDS = set(stopwords.words("german"))


def clean_text(text: str) -> str:
    """A simple text cleaning function to remove stopwords."""
    return " ".join(word for word in text.split() if word.lower() not in STOPWORDS)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def preprocess(cfg: DictConfig) -> None:
    """
    Runs the data preprocessing pipeline.

    1.  Deletes any existing processed data to ensure freshness.
    2.  Loads the raw data from the source CSV.
    3.  Encodes labels and cleans text.
    4.  Tokenizes the text data.
    5.  Saves the processed dataset to disk in Arrow format.
    6.  Saves the fitted label encoders for later use.
    """
    logger.info("Starting data preprocessing...")

    # --- 1. Define paths and clean output directory ---
    data_path = Path(cfg.dataset.get("path", Path.cwd() / "data"))
    raw_csv_path = data_path / "med_indication_all_RF_diag.csv"
    output_dir = Path(cfg.dataset.get("processed_path", data_path / "processed"))

    if output_dir.exists():
        logger.info(f"Removing existing processed data directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Raw data path: {raw_csv_path}")
    logger.info(f"Processed data will be saved to: {output_dir}")

    if not raw_csv_path.exists():
        logger.error(
            f"Raw data file not found at {raw_csv_path}. "
            "Please ensure the source CSV is available."
        )
        return

    # --- 2. Load raw data ---
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
        raw_csv_path, sep=r"\|\|\|", header=None, names=column_names, engine="python"
    )
    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].str.strip()

    # --- 3. Encode Labels ---
    logger.info("Encoding labels...")
    df["combined_label"] = df["medication_type"] + "_" + df["label"]

    le = LabelEncoder()
    df["labels"] = le.fit_transform(df["combined_label"])

    meds_le = LabelEncoder()
    df["med_id"] = meds_le.fit_transform(df["medication_name"])

    # --- 4. Create Hugging Face Dataset and Process Text ---
    doclevel_column = f"{cfg.dataset.doclevel.lower()}"
    if doclevel_column == "letter":
        doclevel_column = "discharge_letter"

    # Select relevant columns for the dataset
    df_for_dataset = df[[doclevel_column, "medication_name", "labels", "med_id"]].copy()
    df_for_dataset.rename(columns={doclevel_column: "text"}, inplace=True)

    hf_dataset = Dataset.from_pandas(df_for_dataset)

    def process_and_tokenize(batch):
        """Function to apply to the dataset for cleaning and tokenizing."""
        # Construct text as in the original script
        text = [
            f"Medikament {name} & {text}"
            for name, text in zip(batch["medication_name"], batch["text"])
        ]

        # Clean text if required
        if cfg.dataset.get("clean", True):
            text = [clean_text(t) for t in text]

        # Tokenize
        tokenized = tokenizer(text, truncation=True, padding="max_length")
        return tokenized

    logger.info("Tokenizing dataset...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.hparams.model_name_or_path)
    hf_dataset = hf_dataset.map(
        process_and_tokenize,
        batched=True,
        remove_columns=["text", "medication_name"],  # No longer needed
    )

    # --- 5. Save Processed Dataset ---
    dataset_path = output_dir / "tokenized_dataset"
    hf_dataset.save_to_disk(dataset_path)
    logger.info(
        f"Processed dataset with {len(hf_dataset)} rows saved to {dataset_path}"
    )
    logger.info(f"Final dataset columns: {hf_dataset.column_names}")

    # --- 6. Save Label Encoders ---
    le_path = output_dir / "label_encoder.joblib"
    meds_le_path = output_dir / "meds_label_encoder.joblib"
    joblib.dump(le, le_path)
    joblib.dump(meds_le, meds_le_path)
    logger.info(f"Label encoders saved to {le_path} and {meds_le_path}")

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    preprocess()
