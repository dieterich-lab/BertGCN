"""Prediction script for BertGCN."""

import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import torch
from datasets import Dataset, load_from_disk
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .train_bert import load_processed_dataset, split_dataset


def _get_logger():
    import logging

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )
    return logging.getLogger("predict")


logger = _get_logger()


def load_model_and_tokenizer(
    cfg: DictConfig,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, LabelEncoder]:
    """Load the trained model, tokenizer, and label encoder."""
    model_path = Path(cfg.inference.model_path)
    if not model_path.exists():
        logger.error(f"Model path {model_path} does not exist.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

    le_path = model_path / "label_encoder.joblib"
    if not le_path.exists():
        logger.error(f"Label encoder not found at {le_path}")
        sys.exit(1)
    le: LabelEncoder = joblib.load(le_path)

    return model, tokenizer, le


def predict(cfg: DictConfig) -> None:
    """Run predictions on the test set."""
    logger.info("Starting prediction...")

    # Load processed dataset
    dataset, le = load_processed_dataset(cfg)
    train_ds, val_ds, test_ds = split_dataset(dataset, cfg)

    # Load model and tokenizer
    model, tokenizer, le = load_model_and_tokenizer(cfg)
    model.eval()

    # Prepare test dataset for prediction
    if isinstance(test_ds, Subset) and hasattr(dataset, "select"):
        predict_indices = [int(i) for i in test_ds.indices]
        predict_ds = dataset.select(predict_indices)
    else:
        predict_ds = test_ds

    # Create data collator
    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create trainer for prediction
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Predict
    logger.info("Running predictions...")
    predictions = trainer.predict(predict_ds)
    preds = np.argmax(predictions.predictions, axis=1)
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()

    # Prepare output
    output_data = []
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        row = {
            "index": i,
            "predicted_label": le.inverse_transform([pred])[0],
            "predicted_class": int(pred),
        }
        for j, p in enumerate(prob):
            row[f"prob_class_{j}"] = float(p)
        if "labels" in predict_ds.column_names:
            true_label = predict_ds[i]["labels"]
            row["true_label"] = le.inverse_transform([true_label])[0]
            row["true_class"] = int(true_label)
        output_data.append(row)

    # Save to CSV
    import pandas as pd

    df = pd.DataFrame(output_data)
    output_file = Path(cfg.inference.output_file)
    df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")


def main(cfg: DictConfig):
    predict(cfg)
