"""Interpretation script for BertGCN using SHAP."""

import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import shap
import torch
from datasets import Dataset, load_from_disk
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from .predict import load_model_and_tokenizer
from .train_bert import load_processed_dataset, split_dataset


def _get_logger():
    import logging

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )
    return logging.getLogger("interpret")


logger = _get_logger()


def interpret(cfg: DictConfig) -> None:
    """Run SHAP interpretations on the test set."""
    logger.info("Starting interpretation...")

    # Load processed dataset
    dataset, le = load_processed_dataset(cfg)
    train_ds, val_ds, test_ds = split_dataset(dataset, cfg)

    # Load model and tokenizer
    model, tokenizer, le = load_model_and_tokenizer(cfg)

    # Prepare test dataset for interpretation
    if isinstance(test_ds, Subset) and hasattr(dataset, "select"):
        predict_indices = [int(i) for i in test_ds.indices]
        predict_ds = dataset.select(predict_indices)
    else:
        predict_ds = test_ds

    # Limit samples
    max_samples = cfg.inference.max_samples
    if len(predict_ds) > max_samples:
        predict_ds = predict_ds.select(range(max_samples))
        logger.info(f"Limited to {max_samples} samples for interpretation")

    # Create pipeline for SHAP
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=0 if torch.cuda.is_available() else -1,
    )

    # SHAP explainer
    explainer = shap.Explainer(pipe)

    # Get texts
    texts = (
        predict_ds["text"]
        if "text" in predict_ds.column_names
        else [""] * len(predict_ds)
    )

    # Explain
    logger.info("Running SHAP explanations...")
    shap_values = explainer(texts)

    # Process results
    results = []
    for i, (shap_val, text) in enumerate(zip(shap_values, texts)):
        # Get top features
        feature_importance = {}
        if hasattr(shap_val, "data"):
            # For text, shap_val.data contains the tokens
            tokens = shap_val.data
            values = shap_val.values
            for j, (token, val) in enumerate(zip(tokens, values)):
                feature_importance[f"token_{j}"] = {
                    "token": token,
                    "importance": float(val),
                }

        results.append(
            {
                "index": i,
                "text": text,
                "shap_values": (
                    str(shap_val.values.tolist()) if hasattr(shap_val, "values") else ""
                ),
                "feature_importance": str(feature_importance),
            }
        )

    # Save to CSV
    df = pd.DataFrame(results)
    output_file = Path(cfg.inference.output_file)
    df.to_csv(output_file, index=False)
    logger.info(f"Interpretations saved to {output_file}")


def main(cfg: DictConfig):
    interpret(cfg)
