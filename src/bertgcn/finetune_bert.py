"""Minimal BERT fine-tuning script with MLflow logging and performance tweaks.

MLflow storage:
    Tracking URI: <project_root>/mlruns
    Launch UI: mlflow ui --backend-store-uri mlruns

Features:
    - Stratified train/val/test split
    - Optional class weighting
    - Macro F1 / precision / recall metrics
    - Confusion matrix artifact

Edit CONFIG SECTION to adjust hyperparameters.
"""

from __future__ import annotations

import inspect
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import joblib
import mlflow
import numpy as np
import torch
from datasets import Dataset, load_from_disk
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.training_args import IntervalStrategy


def _get_logger():
    import logging

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )
    return logging.getLogger("finetune_min")


logger = _get_logger()


######################## CONFIG SECTION ########################
MODEL_NAME_OR_PATH = "/prj/doctoral_letters/PETGUI/med_bert_local"
PROCESSED_DIR = Path("data/processed")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1  # remainder becomes test
SEED = 42
LEARNING_RATE = 1e-5  # lowered for stability
BATCH_SIZE = 16
NUM_EPOCHS = 6  # more epochs
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
LOGGING_STEPS = 50
MLFLOW_EXPERIMENT_NAME = "bertgcn_finetuning_minimal"
USE_STRATIFIED_SPLIT = True
USE_CLASS_WEIGHTS = True
################################################################


def load_processed_dataset() -> Tuple[Dataset, LabelEncoder]:
    dataset_path = PROCESSED_DIR / "tokenized_dataset"
    le_path = PROCESSED_DIR / "label_encoder.joblib"
    if not dataset_path.exists() or not le_path.exists():
        logger.error(
            f"Missing processed data in {PROCESSED_DIR}. Run preprocessing first."
        )
        sys.exit(1)
    ds = load_from_disk(str(dataset_path))
    le: LabelEncoder = joblib.load(le_path)
    trainer_ds = ds.remove_columns(
        [c for c in ["text", "medication_name"] if c in ds.column_names]
    )
    trainer_ds.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels", "med_id"]
    )
    return trainer_ds, le


def split_dataset(dataset: Dataset):
    labels = np.array(dataset["labels"])  # assumes labels column exists
    n = len(labels)
    indices = np.arange(n)

    def random_split():
        rng = np.random.default_rng(SEED)
        rng.shuffle(indices)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)
        return indices[:train_end], indices[train_end:val_end], indices[val_end:]

    if USE_STRATIFIED_SPLIT:
        min_class = min(Counter(labels).values())
        if min_class < 2:
            logger.warning(
                f"Skipping stratified split: smallest class count={min_class} < 2."
            )
            train_idx, val_idx, test_idx = random_split()
        else:
            try:
                from sklearn.model_selection import train_test_split

                train_idx, temp_idx = train_test_split(
                    indices,
                    test_size=1 - TRAIN_RATIO,
                    stratify=labels,
                    random_state=SEED,
                )
                val_ratio_adj = VAL_RATIO / (1 - TRAIN_RATIO)
                val_idx, test_idx = train_test_split(
                    temp_idx,
                    test_size=1 - val_ratio_adj,
                    stratify=labels[temp_idx],
                    random_state=SEED,
                )
            except Exception as e:
                logger.warning(f"Stratified split failed ({e}); using random split.")
                train_idx, val_idx, test_idx = random_split()
    else:
        train_idx, val_idx, test_idx = random_split()

    # Log distribution per split
    for name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        dist = Counter(labels[idxs])
        logger.info(
            f"Split {name}: n={len(idxs)} distinct_classes={len(dist)} sample_dist={dict(list(dist.items())[:10])}"
        )

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def compute_class_weights(labels: np.ndarray):
    if not USE_CLASS_WEIGHTS:
        return None
    try:
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(labels)
        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=labels
        )
        full = np.zeros(len(classes), dtype=np.float32)
        for i, c in enumerate(classes):
            full[c] = weights[i]
        tensor = torch.tensor(full, dtype=torch.float32)
        # Clip extreme weights to stabilize training if there's a very rare class
        if tensor.numel() > 0:
            median = tensor.median()
            if median > 0:
                max_allowed = median * 10.0  # allow up to 10x the median
                if (tensor > max_allowed).any():
                    tensor = torch.clamp(tensor, max=max_allowed)
                    logger.info(
                        f"Clipped class weights to max {max_allowed.item():.4f} to avoid instability"
                    )
        logger.info(f"Class weights (possibly clipped): {tensor.tolist()}")
        return tensor
    except Exception as e:
        logger.warning(f"Class weights failed: {e}")
        return None


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self, model, inputs, return_outputs=False, **kwargs
    ):  # type: ignore[override]
        """Override to apply optional class weights.

        Accepts **kwargs to maintain compatibility with newer Trainer versions
        that may pass additional arguments (e.g., num_items_in_batch).
        """
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            labels=None,
        )
        logits = outputs.logits
        if self.class_weights is not None:
            class_w = self.class_weights.to(logits.device)
            # Safety clamp (should already be clipped earlier, but double safety)
            median = class_w.median()
            if median > 0:
                class_w = torch.clamp(class_w, max=median * 10.0)
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_w)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def setup_trainer(model, tokenizer, train_ds, val_ds, out_dir: str, class_weights=None):
    # Build TrainingArguments with backward compatibility for older Transformers versions
    base_kwargs = dict(
        output_dir=out_dir,
        save_strategy=IntervalStrategy.EPOCH,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=LOGGING_STEPS,
        save_total_limit=2,
        report_to=["mlflow"],
    )
    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        base_kwargs["evaluation_strategy"] = IntervalStrategy.EPOCH
    elif "eval_strategy" in sig.parameters:  # legacy name
        base_kwargs["eval_strategy"] = IntervalStrategy.EPOCH
    else:
        logger.warning(
            "No evaluation/eval_strategy parameter detected; evaluations will rely on default settings."
        )
    args = TrainingArguments(**base_kwargs)
    trainer_cls = WeightedTrainer if class_weights is not None else Trainer
    return trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights if class_weights is not None else None,
    )


def main():
    set_seed(SEED)
    project_root = Path(__file__).resolve().parents[2]
    tracking_uri = project_root / "mlruns"
    mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    out_dir = project_root / "outputs" / "finetuned" / Path(MODEL_NAME_OR_PATH).name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    dataset, le = load_processed_dataset()
    train_ds, val_ds, test_ds = split_dataset(dataset)
    # Efficient label extraction for class weights
    train_indices_int = [int(i) for i in train_ds.indices]
    train_labels = np.array(dataset.select(train_indices_int)["labels"])
    class_weights = compute_class_weights(train_labels)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME_OR_PATH, num_labels=len(le.classes_)
    )
    model.config.id2label = {i: lab for i, lab in enumerate(le.classes_)}
    model.config.label2id = {lab: i for i, lab in enumerate(le.classes_)}

    trainer = setup_trainer(
        model, tokenizer, train_ds, val_ds, str(out_dir), class_weights=class_weights
    )

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_name_or_path": MODEL_NAME_OR_PATH,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "epochs": NUM_EPOCHS,
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio": WARMUP_RATIO,
                "seed": SEED,
                "stratified_split": USE_STRATIFIED_SPLIT,
                "class_weights": USE_CLASS_WEIGHTS,
            }
        )
        logger.info("Training...")
        trainer.train()
        logger.info("Validation metrics (best model):")
        val_metrics = trainer.evaluate()
        mlflow.log_metrics(
            {
                f"val_{k}": float(v)
                for k, v in val_metrics.items()
                if isinstance(v, (int, float))
            }
        )
        logger.info(str(val_metrics))

        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_ds)
        mlflow.log_metrics(
            {
                f"test_{k}": float(v)
                for k, v in test_metrics.items()
                if isinstance(v, (int, float))
            }
        )
        logger.info(str(test_metrics))

        # Confusion matrix
        try:
            from sklearn.metrics import confusion_matrix

            test_output = trainer.predict(test_ds)
            preds = np.argmax(test_output.predictions, axis=1)
            labels = test_output.label_ids
            cm = confusion_matrix(labels, preds)
            import json

            cm_path = out_dir / "confusion_matrix.json"
            with open(cm_path, "w") as f:
                json.dump(
                    {"matrix": cm.tolist(), "labels": le.classes_.tolist()}, f, indent=2
                )
            mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Confusion matrix logging skipped: {e}")

        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        mlflow.log_artifacts(str(out_dir), artifact_path="model")
        logger.info("Done. Launch MLflow UI with: mlflow ui --backend-store-uri mlruns")

    return 0


if __name__ == "__main__":
    main()
