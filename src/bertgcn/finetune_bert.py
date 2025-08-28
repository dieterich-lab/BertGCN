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

import hydra
import joblib
import mlflow
import numpy as np
import torch
from datasets import Dataset, load_from_disk
from hydra.utils import get_original_cwd
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
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.training_args import IntervalStrategy

OmegaConf.register_new_resolver("basename", lambda p: Path(p).name)


def _get_logger():
    import logging

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )
    return logging.getLogger("finetune_bert")


logger = _get_logger()


def load_processed_dataset(cfg: DictConfig) -> Tuple[Dataset, LabelEncoder]:
    project_root = Path(get_original_cwd())
    processed_dir = project_root / "data" / "processed"
    dataset_path = processed_dir / "tokenized_dataset"
    le_path = processed_dir / "label_encoder.joblib"
    if not dataset_path.exists() or not le_path.exists():
        logger.error(
            f"Missing processed data in {processed_dir}. Run preprocessing first."
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


def split_dataset(dataset: Dataset, cfg: DictConfig):
    labels = np.array(dataset["labels"])  # assumes labels column exists
    n = len(labels)
    indices = np.arange(n)
    train_ratio = cfg.dataset.train_ratio
    val_ratio = cfg.dataset.val_ratio
    seed = cfg.hparams.seed
    use_stratified_split = cfg.hparams.use_stratified_split

    def random_split():
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        return indices[:train_end], indices[train_end:val_end], indices[val_end:]

    if use_stratified_split:
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
                    test_size=1 - train_ratio,
                    stratify=labels,
                    random_state=seed,
                )
                val_ratio_adj = val_ratio / (1 - train_ratio)
                val_idx, test_idx = train_test_split(
                    temp_idx,
                    test_size=1 - val_ratio_adj,
                    stratify=labels[temp_idx],
                    random_state=seed,
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


def compute_class_weights(labels: np.ndarray, le: LabelEncoder, cfg: DictConfig):
    if not cfg.hparams.use_class_weights:
        return None
    try:
        from sklearn.utils.class_weight import compute_class_weight

        # Use all classes from the label encoder to determine the shape of the weights tensor
        all_classes = np.arange(len(le.classes_))

        # Calculate weights only for classes present in the provided labels
        present_classes = np.unique(labels)
        weights = compute_class_weight(
            class_weight="balanced", classes=present_classes, y=labels
        )

        # Create a full weights tensor and populate it
        full_weights = np.ones(
            len(all_classes), dtype=np.float32
        )  # Default to 1 for missing classes

        # Map weights to their corresponding class indices
        for cls_idx, weight in zip(present_classes, weights):
            full_weights[cls_idx] = weight

        tensor = torch.tensor(full_weights, dtype=torch.float32)
        # Clip extreme weights to stabilize training if there's a very rare class
        if tensor.numel() > 0:
            # Only consider weights of present classes for median calculation
            present_weights = tensor[present_classes.astype(int)]
            if present_weights.numel() > 0:
                median = present_weights.median()
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
        # Ensure labels are on the same device as logits and have integer dtype
        if labels is not None:
            device = (
                next(model.parameters()).device
                if any(True for _ in model.parameters())
                else torch.device("cpu")
            )
            labels = labels.to(device).long()
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


def setup_trainer(
    model,
    tokenizer,
    train_ds,
    val_ds,
    out_dir: str,
    cfg: DictConfig,
    class_weights=None,
):
    # Build TrainingArguments with backward compatibility for older Transformers versions
    base_kwargs = dict(
        output_dir=out_dir,
        save_strategy=IntervalStrategy.EPOCH,
        learning_rate=cfg.hparams.learning_rate,
        per_device_train_batch_size=cfg.hparams.batch_size,
        per_device_eval_batch_size=cfg.hparams.batch_size,
        num_train_epochs=cfg.hparams.num_train_epochs,
        weight_decay=cfg.hparams.weight_decay,
        warmup_ratio=cfg.hparams.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=cfg.hparams.eval_steps,  # assuming eval_steps is used for logging
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
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer_args = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    if class_weights is not None:
        trainer_args["class_weights"] = class_weights

    # Attach early stopping callback if configured
    patience = getattr(cfg.hparams, "early_stopping_patience", None)
    if patience is not None and int(patience) > 0:
        trainer_args.setdefault("callbacks", [])
        trainer_args["callbacks"].append(
            EarlyStoppingCallback(early_stopping_patience=int(patience))
        )
        logger.info(f"Early stopping enabled with patience={patience}.")

    return trainer_cls(**trainer_args)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.hparams.seed)
    project_root = Path(get_original_cwd())
    tracking_uri = project_root / "mlruns"
    mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_experiment(cfg.mlflow_experiment_name)
    # Enable MLflow autologging for transformers (fallback to pytorch). This
    # helps capture parameters, metrics and artifacts automatically when
    # available.
    try:
        import mlflow.transformers as _mlt

        _mlt.autolog()
        logger.info("Enabled mlflow.transformers.autolog()")
    except Exception:
        try:
            import mlflow.pytorch as _mlp

            _mlp.autolog()
            logger.info("Enabled mlflow.pytorch.autolog()")
        except Exception:
            logger.info("mlflow autolog not available; using manual logging.")
    # Use the original working directory as project root and the Hydra run
    # directory to construct an absolute output path. This avoids relying on
    # nested `cfg.config` attributes which may not exist.
    out_dir = project_root / Path(cfg.hydra.run.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    dataset, le = load_processed_dataset(cfg)
    train_ds, val_ds, test_ds = split_dataset(dataset, cfg)
    # Efficient label extraction for class weights
    train_indices_int = [int(i) for i in train_ds.indices]
    train_labels = np.array(dataset.select(train_indices_int)["labels"])
    class_weights = compute_class_weights(train_labels, le, cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.hparams.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.hparams.model_name_or_path, num_labels=len(le.classes_)
    )
    model.config.id2label = {i: lab for i, lab in enumerate(le.classes_)}
    model.config.label2id = {lab: i for i, lab in enumerate(le.classes_)}

    with mlflow.start_run():
        # Create the Trainer inside the active MLflow run so autolog captures
        # the training lifecycle and artifacts more reliably.
        trainer = setup_trainer(
            model,
            tokenizer,
            train_ds,
            val_ds,
            str(out_dir),
            cfg,
            class_weights=class_weights,
        )

        mlflow.log_params(
            {
                "model_name_or_path": cfg.hparams.model_name_or_path,
                "learning_rate": cfg.hparams.learning_rate,
                "batch_size": cfg.hparams.batch_size,
                "epochs": cfg.hparams.num_train_epochs,
                "weight_decay": cfg.hparams.weight_decay,
                "warmup_ratio": cfg.hparams.warmup_ratio,
                "seed": cfg.hparams.seed,
                "stratified_split": cfg.hparams.use_stratified_split,
                "class_weights": cfg.hparams.use_class_weights,
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
        # Diagnostic: log the files we are about to upload so we can debug
        # empty-artifact cases. This prints relative paths and sizes.
        try:
            logger.info(f"Saved model files to: {out_dir.resolve()}")
            for p in sorted(out_dir.rglob("**/*")):
                try:
                    size = p.stat().st_size
                except Exception:
                    size = -1
                logger.info(f"  {p.relative_to(out_dir)} ({size} bytes)")
        except Exception as e:  # pragma: no cover - best-effort logging
            logger.warning(f"Could not list out_dir contents: {e}")

        # Prefer MLflow-native model logging when available. Try
        # mlflow.transformers.log_model first (captures tokenizer, model and
        # flavor metadata). Fallback to mlflow.pytorch.log_model, and if
        # neither is available fall back to uploading artifacts.
        registered_name = None
        try:
            registered_name = cfg.mlflow_model_name
        except Exception:
            try:
                registered_name = cfg.hparams.get("mlflow_model_name", None)
            except Exception:
                registered_name = None

        logged = False
        try:
            import mlflow.transformers as mlt

            mlt.log_model(
                transformers_model=model,
                artifact_path="model",
                tokenizer=tokenizer,
                registered_model_name=registered_name if registered_name else None,
            )
            logger.info("Logged model with mlflow.transformers.log_model()")
            logged = True
        except Exception:
            try:
                import mlflow.pytorch as mlp

                # mlflow.pytorch.log_model accepts a PyTorch model; HF models
                # subclass torch.nn.Module and the tokenizer should be
                # separately saved as artifacts.
                mlp.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=registered_name if registered_name else None,
                )
                # also log tokenizer files
                mlflow.log_artifacts(str(out_dir), artifact_path="model/tokenizer")
                logger.info(
                    "Logged model with mlflow.pytorch.log_model() and tokenizer artifacts"
                )
                logged = True
            except Exception:
                logger.info(
                    "mlflow transformers/pytorch logging not available; falling back to artifact upload"
                )

        if not logged:
            # Last-resort: upload the directory contents as run artifacts.
            mlflow.log_artifacts(str(out_dir), artifact_path="model")
        logger.info("Done. Launch MLflow UI with: mlflow ui --backend-store-uri mlruns")

    return 0


if __name__ == "__main__":
    main()
