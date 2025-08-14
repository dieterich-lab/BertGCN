"""
Fine-tune a Hugging Face Transformer model for sequence classification.

This script provides a clean, modern approach to fine-tuning transformer models
on sequence classification tasks with:
- Proper logging and checkpointing
- Early stopping and learning rate scheduling
- Comprehensive evaluation metrics
- Support for various model architectures
"""

import datetime
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import IntervalStrategy

from bertgcn.clinic_datasets import CleanClinicDataset
from bertgcn.config import MODEL_PATHS
from bertgcn.core import get_logger, setup_environment
from bertgcn.params import parse_args

# Initialize logger
logger = get_logger(__name__)


def load_or_create_dataset(
    tokenizer, model_name: str, doc_level: str, task: str = "MIC", clean: bool = False
) -> CleanClinicDataset:
    """
    Load or create the dataset for fine-tuning.

    Args:
        tokenizer: Tokenizer to use for processing text
        model_name: Name of the model (used in filename)
        doc_level: Document level (letter, sentence, etc.)
        task: Task name
        clean: Whether to clean the text

    Returns:
        The dataset
    """
    dataset_file = Path("data") / f"{task.lower()}_{model_name}_{doc_level}.json"

    if dataset_file.exists():
        logger.info(f"Loading dataset from: {dataset_file}")
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        logger.info("Creating dataset")
        dataset = CleanClinicDataset(
            tokenizer=tokenizer,
            task=task,
            doclevel=doc_level,
            clean=clean,
        )
        logger.info(f"Saving dataset to: {dataset_file}")
        os.makedirs(dataset_file.parent, exist_ok=True)
        with open(dataset_file, "wb") as f:
            pickle.dump(dataset, f)

    return dataset


def split_dataset(
    dataset: Dataset, test_unclear: bool = False
) -> Tuple[Subset, Subset, Subset]:
    """
    Split dataset into train, validation and test sets.

    Args:
        dataset: Dataset to split
        test_unclear: If True, use all "unclear" samples as test set

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """

    if not test_unclear:
        # Random split
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        train_idx = idx[: int(len(idx) * 0.7)]
        val_idx = idx[int(len(idx) * 0.7) : int(len(idx) * 0.8)]
        test_idx = idx[int(len(idx) * 0.8) :]
    else:
        # Separate unclear samples
        train_val_idx, test_idx = [], []
        for i, x in enumerate(dataset):
            if "unklar" in dataset.LE.classes_[x["labels"]]:
                test_idx.append(i)
            else:
                train_val_idx.append(i)

        # Split the non-unclear samples into train/val
        np.random.shuffle(train_val_idx)
        split_idx = int(len(train_val_idx) * 0.9)
        train_idx = train_val_idx[:split_idx]
        val_idx = train_val_idx[split_idx:]

        logger.info(
            f"Dataset split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test "
            f"(total: {len(train_idx) + len(val_idx) + len(test_idx)})"
        )

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute evaluation metrics from model predictions.

    Args:
        eval_pred: Model predictions and labels

    Returns:
        Dictionary of metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    # Basic metrics
    accuracy = np.mean(predictions == labels)

    # Per class metrics
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    precision_macro = precision_score(
        labels, predictions, average="macro", zero_division=0
    )
    recall_macro = recall_score(labels, predictions, average="macro", zero_division=0)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)

    # Return metrics
    metrics = {
        "accuracy": accuracy,
        "f1": f1_macro,
        "precision": precision_macro,
        "recall": recall_macro,
    }

    return metrics


def save_detailed_metrics(
    trainer, dataset, output_dir: Union[str, Path], split: str = "test"
):
    """
    Save detailed evaluation metrics like confusion matrix and classification report.

    Args:
        trainer: The Hugging Face Trainer object
        dataset: Dataset to evaluate on
        output_dir: Directory to save metrics to
        split: Dataset split name (test, val)
    """
    from sklearn.metrics import classification_report, confusion_matrix

    # Get predictions
    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    # Get class names
    if hasattr(dataset.dataset, "LE"):
        class_names = dataset.dataset.LE.classes_
    else:
        # If LE is not available, use numeric class names
        class_names = [str(i) for i in range(len(np.unique(labels)))]

    # Create classification report
    cr = classification_report(
        labels, preds, target_names=class_names, output_dict=True
    )
    cr_df = pd.DataFrame(cr).transpose()

    # Create confusion matrix
    cm = confusion_matrix(labels, preds)

    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cr_df.to_csv(
        os.path.join(output_dir, f"{split}_classification_report_{timestamp}.csv")
    )
    np.save(os.path.join(output_dir, f"{split}_confusion_matrix_{timestamp}.npy"), cm)

    # Log metrics
    logger.info(f"\n{cr_df.to_string()}")
    logger.info(f"\nConfusion Matrix:\n{cm}")


def main():
    """Main entry point for fine-tuning."""
    # Parse arguments
    args = parse_args()

    # Set up environment
    setup_environment(args.seed)
    set_seed(args.seed)

    # Set up logging
    log_dir = Path("logs") / "finetune" / args.data
    os.makedirs(log_dir, exist_ok=True)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    file_handler = logging.FileHandler(log_dir / f"{now_str}.log", mode="w")
    logger.addHandler(file_handler)

    # Log parameters
    logger.info(f"Starting fine-tuning with parameters:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")

    # Prepare model output directory
    model_name = Path(args.model_name_or_path).name
    output_dir = Path(args.output_dir) / args.doclevel / f"{model_name}_{now_str}"
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Load or create dataset
    dataset = load_or_create_dataset(
        tokenizer=tokenizer,
        model_name=model_name,
        doc_level=args.doclevel,
    )

    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset=dataset, test_unclear=args.testunklar
    )

    # Load model
    logger.info(f"Loading model from {args.model_name_or_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(dataset.LE.classes_),
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=args.eval_steps,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs
        or args.nepochs,  # Use nepochs as fallback
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_dir=log_dir,
        logging_steps=50,
        save_total_limit=3,
        report_to=["tensorboard"],
    )

    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # Train model
    if not args.testonly:
        logger.info("Starting training")
        trainer.train()

        # Save best model
        logger.info(f"Saving best model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Test results: {test_results}")

    # Save detailed metrics
    save_detailed_metrics(trainer, test_dataset, output_dir, "test")

    # Save model configuration and args
    with open(os.path.join(output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("Fine-tuning completed")


if __name__ == "__main__":
    main()
