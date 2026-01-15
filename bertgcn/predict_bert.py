"""Prediction script for BERT model."""

import sys
from pathlib import Path
from typing import Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import hydra

from .train_bert import load_processed_dataset, split_dataset


def _get_logger():
    import logging

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )
    return logging.getLogger("predict_bert")


logger = _get_logger()


def load_model_and_tokenizer(
    cfg: DictConfig,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, LabelEncoder]:
    """Load the latest trained BERT model from MLflow artifact."""
    import logging

    logger = logging.getLogger("predict_bert")

    client = mlflow.tracking.MlflowClient()
    exp_name = "train_bert"  # Should match the BERT experiment name
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        raise RuntimeError(f"No MLflow experiment named {exp_name} found.")
    runs = client.search_runs(
        exp.experiment_id,
        "attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No finished BERT runs found in MLflow.")
    run = runs[0]

    # Log BERT run info
    run_date = run.info.start_time if hasattr(run.info, "start_time") else None
    import datetime

    if run_date:
        run_date_str = datetime.datetime.fromtimestamp(run_date / 1000).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    else:
        run_date_str = "unknown"

    bert_params = {}
    if run.data and run.data.params:
        for k, v in run.data.params.items():
            if any(
                param in k
                for param in [
                    "learning_rate",
                    "num_train_epochs",
                    "batch_size",
                    "weight_decay",
                ]
            ):
                bert_params[k] = v

    logger.info(
        f"Loaded latest BERT from MLflow run_id={run.info.run_id}, experiment={exp_name}"
    )
    logger.info(f"BERT training date: {run_date_str}")
    if bert_params:
        logger.info(f"BERT hyperparameters: {bert_params}")

    artifact_path = "final_model"
    model_dir = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path=artifact_path
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Load label encoder from processed dataset
    try:
        project_root = Path(get_original_cwd())
    except Exception:
        project_root = Path.cwd()
    processed_dir = project_root / "data" / "processed"
    le_path = processed_dir / "label_encoder.joblib"
    if not le_path.exists():
        raise FileNotFoundError(f"Label encoder not found at {le_path}")
    le: LabelEncoder = joblib.load(le_path)

    return model, tokenizer, le


def predict(cfg: DictConfig) -> None:
    """Run predictions on the test set."""
    logger.info("Starting BERT prediction...")

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

    # Calculate metrics if debug mode is enabled
    if getattr(cfg.inference, "debug", False):
        true_labels = [sample["labels"] for sample in predict_ds]
        accuracy = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds, average="weighted")
        logger.info("\n📊 DEBUG MODE - BERT Test Set Metrics:")
        logger.info(f"   • Accuracy: {accuracy:.1%}")
        logger.info(f"   • F1 Score (weighted): {f1:.3f}")
        logger.info("   • Reference: Compare with BERT training results")

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
    df = pd.DataFrame(output_data)
    output_file = Path(cfg.inference.output_file)
    df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Allow legacy overrides that address keys not present in the base config
    OmegaConf.set_struct(cfg, False)

    predict(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print("\n\n==== EXCEPTION OCCURRED ====")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
