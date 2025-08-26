"""
Preprocesses the raw clinical data into a Hugging Face Dataset.

This script handles the one-time task of cleaning, tokenizing, and saving the
dataset in the efficient Arrow format. It also saves the label encoders.
"""

import pickle
from pathlib import Path

import hydra
import joblib
from omegaconf import DictConfig
from transformers import AutoTokenizer

from bertgcn.clinic_datasets import CleanClinicDataset
from bertgcn.core import get_logger

logger = get_logger(__name__)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def preprocess(cfg: DictConfig) -> None:
    """
    Runs the data preprocessing pipeline.

    1. Loads the raw data from the source CSV.
    2. Cleans and tokenizes the text data.
    3. Saves the processed dataset to disk in Arrow format.
    4. Saves the label encoders for later use.
    """
    logger.info("Starting data preprocessing...")

    # Define paths
    data_path = Path(cfg.dataset.get("path", Path.cwd() / "data"))
    raw_csv_path = data_path / "med_indication_all_RF_diag.csv"
    output_dir = Path(cfg.dataset.get("processed_path", data_path / "processed"))
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Raw data path: {raw_csv_path}")
    logger.info(f"Processed data output path: {output_dir}")

    if not raw_csv_path.exists():
        logger.error(
            f"Raw data file not found at {raw_csv_path}. "
            "Please ensure the source CSV is available before running preprocessing."
        )
        return

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.hparams.model_name_or_path)

    # Use CleanClinicDataset to perform the preprocessing logic
    # Note: We are using this class as a processor, not as a PyTorch dataset wrapper
    clinic_data_processor = CleanClinicDataset(
        tokenizer=tokenizer,
        doclevel=cfg.dataset.doclevel,
        clean=cfg.dataset.get("clean", True),
        file_path=raw_csv_path,
    )

    # The processed data is in the `dataset` attribute
    processed_dataset = clinic_data_processor.dataset
    logger.info(f"Processed dataset has {len(processed_dataset)} rows.")

    # Rename the 'label_id' column to 'labels' to match what the Trainer expects
    if "label_id" in processed_dataset.column_names:
        processed_dataset = processed_dataset.rename_column("label_id", "labels")
        logger.info("Renamed column 'label_id' to 'labels'.")

    # Save the processed dataset to disk
    dataset_path = output_dir / "tokenized_dataset"
    processed_dataset.save_to_disk(dataset_path)
    logger.info(f"Processed dataset saved to {dataset_path}")

    # Save the label encoders
    le_path = output_dir / "label_encoder.joblib"
    meds_le_path = output_dir / "meds_label_encoder.joblib"
    joblib.dump(clinic_data_processor.LE, le_path)
    joblib.dump(clinic_data_processor.medsLE, meds_le_path)
    logger.info(f"Label encoders saved to {le_path} and {meds_le_path}")

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    preprocess()
