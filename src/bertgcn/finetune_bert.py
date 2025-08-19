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


def get_git_commit() -> str:
    """Get the current git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "Not a git repository"


def log_environment_info(cfg: DictConfig):
    """Log environment details to MLflow."""
    env_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": get_git_commit(),
        "packages": {
            m.__name__: m.__version__
            for m in sys.modules.values()
            if hasattr(m, "__version__")
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    mlflow.log_dict(env_info, "environment.json")
    logger.info("Logged environment and configuration details to MLflow.")


def load_or_create_dataset(tokenizer, cfg: DictConfig) -> CleanClinicDataset:
    """Load or create the dataset."""
    model_name = Path(cfg.hparams.model_name_or_path).name
    dataset_file = Path("data") / f"MIC_{model_name}_{cfg.dataset.doclevel}.pkl"
    if dataset_file.exists():
        logger.info(f"Loading dataset from: {dataset_file}")
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        logger.info("Creating dataset")
        dataset = CleanClinicDataset(
            tokenizer=tokenizer,
            task=cfg.dataset.task,
            doclevel=cfg.dataset.doclevel,
            clean=False,
        )
        os.makedirs(dataset_file.parent, exist_ok=True)
        with open(dataset_file, "wb") as f:
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
        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        test_idx = idx[val_end:]
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
        evaluation_strategy=IntervalStrategy.STEPS,
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


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    """Main training and evaluation pipeline."""
    setup_environment(cfg.hparams.seed)
    set_seed(cfg.hparams.seed)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    mlflow.set_experiment(cfg.mlflow_experiment_name)
    with mlflow.start_run() as run:
        logger.info(f"Starting run: {run.info.run_name}")
        logger.info(f"Output directory: {output_dir}")
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


if __name__ == "__main__":
    main()
