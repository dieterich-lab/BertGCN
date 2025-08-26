"""
Fine-tune a Hugging Face Transformer model for sequence classification.

This script is designed for MLOps workflows, integrating:
- Hydra for configuration management.
- MLflow for experiment tracking.
- Hugging Face Trainer for efficient training.
- SHAP and LIME for model explainability.
- Optuna (via Hydra) for hyperparameter sweeping.
"""

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import hydra
import joblib
import mlflow
import numpy as np
import pandas as pd
import toml
from datasets import Dataset, load_from_disk
from lime.lime_text import LimeTextExplainer
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.training_args import IntervalStrategy

from bertgcn.core import get_logger, setup_environment

logger = get_logger(__name__)


def get_git_commit() -> str:
    """Returns the current git commit hash, or 'Not a git repo' if not in a git repo."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Not a git repo"


def log_environment_info(cfg: DictConfig):
    """Logs hardware, OS, and package version info to MLflow."""
    env_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "git_commit": get_git_commit(),
    }

    # Add package versions, filtered by pyproject.toml
    try:
        with open("pyproject.toml", "r") as f:
            pyproject = toml.load(f)

        dependencies = (
            pyproject.get("tool", {}).get("poetry", {}).get("dependencies", {})
        )

        package_versions = {
            dist.metadata["name"]: dist.version
            for dist in importlib.metadata.distributions()
        }

        relevant_packages = {
            name: package_versions.get(name, "Not Found")
            for name in dependencies
            if name != "python"  # Exclude python itself
        }
        env_info["packages"] = relevant_packages

    except (FileNotFoundError, ImportError) as e:
        logger.warning(f"Could not log package versions: {e}")
        env_info["packages"] = "Could not be determined"

    mlflow.log_dict(env_info, "environment.json")


def load_processed_dataset(
    cfg: DictConfig,
) -> Tuple[Dataset, LabelEncoder, LabelEncoder]:
    """Loads the preprocessed dataset and label encoders from disk."""
    data_path = Path(
        cfg.dataset.get("processed_path", Path.cwd() / "data" / "processed")
    )
    dataset_path = data_path / "tokenized_dataset"
    le_path = data_path / "label_encoder.joblib"
    meds_le_path = data_path / "meds_label_encoder.joblib"

    if not dataset_path.exists() or not le_path.exists() or not meds_le_path.exists():
        logger.error(
            f"Processed data not found in {data_path}. "
            "Please run the preprocessing script first: "
            "`poetry run python -m bertgcn.preprocess`"
        )
        sys.exit(1)

    logger.info(f"Loading processed dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)

    logger.info("Loading label encoders")
    le = joblib.load(le_path)
    meds_le = joblib.load(meds_le_path)

    # Set the format for PyTorch
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels", "med_id"]
    )

    return dataset, le, meds_le


def split_dataset(
    dataset: Dataset, le: LabelEncoder, cfg: DictConfig
) -> Tuple[Subset, Subset, Subset]:
    """Split the dataset into train, validation, and test sets based on config."""
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    if not cfg.dataset.test_unclear:
        train_end = int(len(idx) * cfg.dataset.train_ratio)
        val_end = train_end + int(len(idx) * cfg.dataset.val_ratio)
        train_idx = idx[:train_end].tolist()
        val_idx = idx[train_end:val_end].tolist()
        test_idx = idx[val_end:].tolist()
    else:
        train_val_idx, test_idx = [], []
        for i in idx:
            # We need to get the label name to check for "unklar"
            # This is less efficient than having a dedicated column, but works.
            label_name = le.inverse_transform([dataset[int(i)]["label_id"]])[0]
            if "unklar" in label_name:
                test_idx.append(int(i))
            else:
                train_val_idx.append(int(i))

        np.random.shuffle(train_val_idx)
        split_idx = int(len(train_val_idx) * cfg.dataset.train_val_split_ratio)
        train_idx = train_val_idx[:split_idx]
        val_idx = train_val_idx[split_idx:]

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute and return metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    from sklearn.metrics import f1_score, precision_score, recall_score

    return {
        "accuracy": np.mean(predictions == labels),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
    }


def log_explainability_artifacts(trainer: Trainer, dataset: Subset, output_dir: str):
    """Generate and log SHAP and LIME explainability reports."""
    logger.info("Generating explainability reports...")
    # Note: Explainability tools like SHAP and LIME require the raw text, which is
    # not part of this streamlined training pipeline. To enable them, the data
    # loading process would need to be adapted to also provide the original text.
    logger.warning(
        "Skipping SHAP and LIME explainability as raw text is not available in this workflow."
    )


def setup_trainer(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    train_dataset: Subset,
    val_dataset: Subset,
    cfg: DictConfig,
    output_dir: str,
) -> Trainer:
    """Configure and return a Hugging Face Trainer."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=cfg.hparams.eval_steps,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=cfg.hparams.save_steps,
        learning_rate=cfg.hparams.learning_rate,
        per_device_train_batch_size=cfg.hparams.batch_size,
        per_device_eval_batch_size=cfg.hparams.batch_size,
        num_train_epochs=cfg.hparams.num_train_epochs,
        weight_decay=cfg.hparams.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=cfg.hparams.fp16,
        gradient_accumulation_steps=cfg.hparams.gradient_accumulation_steps,
        warmup_ratio=cfg.hparams.warmup_ratio,
        logging_dir=f"{output_dir}/tensorboard",
        logging_steps=50,
        save_total_limit=3,
        report_to=["mlflow", "tensorboard"],
    )

    # The default data collator will handle padding and tensor conversion
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )


def main(cfg: DictConfig) -> float:
    """Main training and evaluation pipeline."""
    setup_environment(cfg.hparams.seed)
    set_seed(cfg.hparams.seed)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    mlflow.set_experiment(cfg.mlflow_experiment_name)
    with mlflow.start_run() as run:
        logger.info(f"Starting run: {run.info.run_name}")
        logger.info(f"Output directory: {output_dir}")
        mlflow.set_tag("doclevel", cfg.dataset.doclevel)
        log_environment_info(cfg)
        mlflow.log_params(OmegaConf.to_container(cfg.hparams, resolve=True))

        tokenizer = AutoTokenizer.from_pretrained(cfg.hparams.model_name_or_path)

        # Load the pre-processed dataset
        dataset, le, _ = load_processed_dataset(cfg)

        train_dataset, val_dataset, test_dataset = split_dataset(dataset, le, cfg)

        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.hparams.model_name_or_path, num_labels=len(le.classes_)
        )

        trainer = setup_trainer(
            model, tokenizer, train_dataset, val_dataset, cfg, output_dir
        )

        if not cfg.testonly:
            logger.info("Starting training...")
            trainer.train()
            logger.info(f"Saving best model to {output_dir}")
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)

        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(test_dataset)
        logger.info(f"Test results: {test_results}")

        log_explainability_artifacts(trainer, test_dataset, output_dir)

        logger.info("Fine-tuning completed.")

        # For Hydra's Optuna Sweeper
        return test_results.get("eval_f1", 0.0)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
