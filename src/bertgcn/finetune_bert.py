"""
Fine-tune a Hugging Face Transformer model for sequence classification.

This script is designed for MLOps workflows, integrating:
- Hydra for configuration management.
- MLflow for experiment tracking.
- Hugging Face Trainer for efficient training.
- SHAP and LIME for model explainability.
- Optuna (via Hydra) for hyperparameter sweeping.
"""

import os
import pickle
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple, Union

import hydra
import mlflow
import numpy as np
import pandas as pd
import shap
from lime.lime_text import LimeTextExplainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.training_args import IntervalStrategy

from bertgcn.clinic_datasets import CleanClinicDataset
from bertgcn.core import get_logger, setup_environment

logger = get_logger(__name__)


import importlib.metadata


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
        "packages": {
            dist.metadata["name"]: dist.version
            for dist in importlib.metadata.distributions()
        },
    }
    # Filter to include only packages listed in pyproject.toml under dependencies
    # This avoids logging every single package in the environment.
    try:
        import toml

        with open("pyproject.toml", "r") as f:
            pyproject = toml.load(f)
        dependencies = (
            pyproject.get("tool", {}).get("poetry", {}).get("dependencies", {})
        )
        relevant_packages = {
            name: env_info["packages"].get(name)
            for name in dependencies
            if name in env_info["packages"]
        }
        env_info["packages"] = relevant_packages
    except (FileNotFoundError, ImportError):
        # If pyproject.toml or toml is not available, log all packages
        pass

    mlflow.log_dict(env_info, "environment.json")


def load_or_create_dataset(tokenizer, cfg: DictConfig) -> CleanClinicDataset:
    """Loads or creates the dataset based on the provided config."""
    data_path = Path(cfg.dataset.get("path", None) or Path.cwd() / "data")
    preprocessed_pickle_path = data_path / "medindcls_medbert_letter_clean.pkl"

    if preprocessed_pickle_path.exists():
        logger.info(f"Loading pre-processed dataset from {preprocessed_pickle_path}")

        # Workaround for pickle loading error due to module path change
        # The pickle was created when CleanClinicDataset was in a 'clinic_datasets' module
        # We need to map 'clinic_datasets' to the new module path 'bertgcn.clinic_datasets'
        from bertgcn import clinic_datasets

        sys.modules["clinic_datasets"] = clinic_datasets

        with open(preprocessed_pickle_path, "rb") as f:
            return pickle.load(f)

    # Fallback to original creation logic if the main pickle file is not found
    logger.warning(
        f"Pre-processed pickle not found at {preprocessed_pickle_path}. "
        "Attempting to create dataset from source CSV. This will fail if the CSV is missing."
    )
    doclevel = cfg.dataset.doclevel
    dev_limit = cfg.dataset.get("dev_limit", None)
    clean = cfg.dataset.get("clean", True)

    # The cache path will be unique for each combination of dataset parameters
    cache_path = (
        data_path / f"cached_dataset_{doclevel}_{dev_limit if dev_limit else 'all'}.pkl"
    )

    if cfg.dataset.get("use_cache", True) and cache_path.exists():
        logger.info(f"Loading cached dataset from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info("Creating dataset from CSV...")
    dataset = CleanClinicDataset(
        tokenizer=tokenizer,
        doclevel=doclevel,
        dev_limit=dev_limit,
        clean=clean,
        file_path=data_path / "med_indication_all_RF_diag.csv",
    )

    if cfg.dataset.get("use_cache", True):
        logger.info(f"Saving dataset to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(dataset, f)

    return dataset


def split_dataset(
    dataset: CleanClinicDataset, cfg: DictConfig
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
        for i, x in enumerate(dataset):
            if "unklar" in dataset.LE.classes_[x["labels"]]:
                test_idx.append(i)
            else:
                train_val_idx.append(i)
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
    model = trainer.model
    tokenizer = trainer.tokenizer
    # SHAP
    try:
        explainer = shap.Explainer(model, tokenizer)
        shap_values = explainer(dataset[:10])  # Explain first 10 samples
        shap.save_html(os.path.join(output_dir, "shap_explanation.html"), shap_values)
        mlflow.log_artifact(os.path.join(output_dir, "shap_explanation.html"))
    except Exception as e:
        logger.warning(f"SHAP explainability failed: {e}")
    # LIME
    try:
        class_names = getattr(dataset.dataset, "LE", None).classes_
        lime_explainer = LimeTextExplainer(class_names=class_names)
        text_sample = dataset.dataset.texts[dataset.indices[0]]

        def predictor(texts):
            inputs = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)
            return model(**inputs).logits.detach().cpu().numpy()

        lime_exp = lime_explainer.explain_instance(text_sample, predictor)
        lime_exp.save_to_file(os.path.join(output_dir, "lime_explanation.html"))
        mlflow.log_artifact(os.path.join(output_dir, "lime_explanation.html"))
    except Exception as e:
        logger.warning(f"LIME explainability failed: {e}")


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

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.hparams.patience)],
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
        dataset = load_or_create_dataset(tokenizer, cfg)
        train_dataset, val_dataset, test_dataset = split_dataset(dataset, cfg)

        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.hparams.model_name_or_path, num_labels=len(dataset.LE.classes_)
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


import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
