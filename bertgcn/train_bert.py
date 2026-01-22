#!/usr/bin/env python3
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

import joblib
import mlflow
import numpy as np
import torch
from datasets import Dataset, load_from_disk
from hydra.core.hydra_config import HydraConfig
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
from transformers.training_args import IntervalStrategy, SaveStrategy

import hydra

OmegaConf.register_new_resolver("basename", lambda p: Path(p).name)


# Suppress BERT pooler warnings that are not relevant for fine-tuning
warnings.filterwarnings(
    "ignore",
    message="Some weights of.*were not initialized from the model checkpoint",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="You should probably TRAIN this model on a down-stream task",
    category=UserWarning,
)


def _get_logger():
    import logging
    import sys

    # Create a custom formatter for better readability
    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno >= logging.ERROR:
                color = "\033[91m"  # Red
            elif record.levelno >= logging.WARNING:
                color = "\033[93m"  # Yellow
            elif record.levelno >= logging.INFO:
                color = "\033[92m"  # Green
            else:
                color = "\033[0m"  # Reset

            # Add special formatting for key metrics
            if hasattr(record, "highlight") and record.highlight:
                return f"\033[1;94m{'='*60}\n{record.getMessage()}\n{'='*60}\033[0m"
            elif hasattr(record, "section") and record.section:
                return f"\033[1;96m{'─'*50}\n{record.getMessage()}\n{'─'*50}\033[0m"
            else:
                return f"{color}{super().format(record)}\033[0m"

    formatter = ColoredFormatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    logger = logging.getLogger("finetune_bert")
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    # Add console handler with colored formatting
    # Emit logs to stdout so Slurm captures them in *.log files, not *.err.
    class FlushingStreamHandler(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            try:
                self.flush()
                if self.stream and hasattr(self.stream, "flush"):
                    self.stream.flush()
            except Exception:
                pass

    console_handler = FlushingStreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def _format_metrics(metrics_dict, title="Metrics"):
    """Format metrics dictionary into a clean, readable table."""
    lines = [f"{title}:"]
    lines.append("-" * 40)

    for key, value in metrics_dict.items():
        if isinstance(value, float):
            # Format floats nicely
            if key in ["accuracy", "f1", "precision", "recall"]:
                lines.append(f"  {key.capitalize():<12}: {value:.1%}")
            else:
                lines.append(f"  {key.capitalize():<12}: {value:.4f}")
        else:
            lines.append(f"  {key.capitalize():<12}: {value}")

    return "\n".join(lines)


def _format_confusion_matrix(cm, labels, title="Confusion Matrix"):
    """Format confusion matrix into a readable table."""
    lines = [f"{title}:"]
    lines.append("-" * 50)

    # Header
    header_parts = ["True\\Pred".ljust(15)] + [label.ljust(8) for label in labels]
    header = "".join(header_parts)
    lines.append(header)
    lines.append("-" * len(header))

    # Matrix rows
    for i, (label, row) in enumerate(zip(labels, cm)):
        row_parts = [label.ljust(15)] + [str(val).ljust(8) for val in row]
        row_str = "".join(row_parts)
        lines.append(row_str)

    return "\n".join(lines)


def _log_training_summary(cfg, val_metrics, test_metrics, final_dir, mlruns_path):
    """Log a comprehensive training summary."""
    summary_lines = []
    summary_lines.append("🎯 TRAINING COMPLETED SUCCESSFULLY")
    summary_lines.append("")
    summary_lines.append("📊 BEST VALIDATION METRICS:")
    acc = val_metrics.get("eval_accuracy", "N/A")
    summary_lines.append(
        f"   • Accuracy:  {acc:.1%}"
        if isinstance(acc, (int, float))
        else f"   • Accuracy:  {acc}"
    )
    f1 = val_metrics.get("eval_f1", "N/A")
    summary_lines.append(
        f"   • F1 Score:  {f1:.3f}"
        if isinstance(f1, (int, float))
        else f"   • F1 Score:  {f1}"
    )
    prec = val_metrics.get("eval_precision", "N/A")
    summary_lines.append(
        f"   • Precision: {prec:.3f}"
        if isinstance(prec, (int, float))
        else f"   • Precision: {prec}"
    )
    recall = val_metrics.get("eval_recall", "N/A")
    summary_lines.append(
        f"   • Recall:    {recall:.3f}"
        if isinstance(recall, (int, float))
        else f"   • Recall:    {recall}"
    )
    summary_lines.append("")
    summary_lines.append("🧪 TEST SET PERFORMANCE:")
    test_acc = test_metrics.get("eval_accuracy", "N/A")
    summary_lines.append(
        f"   • Accuracy:  {test_acc:.1%}"
        if isinstance(test_acc, (int, float))
        else f"   • Accuracy:  {test_acc}"
    )
    test_f1 = test_metrics.get("eval_f1", "N/A")
    summary_lines.append(
        f"   • F1 Score:  {test_f1:.3f}"
        if isinstance(test_f1, (int, float))
        else f"   • F1 Score:  {test_f1}"
    )
    test_prec = test_metrics.get("eval_precision", "N/A")
    summary_lines.append(
        f"   • Precision: {test_prec:.3f}"
        if isinstance(test_prec, (int, float))
        else f"   • Precision: {test_prec}"
    )
    test_recall = test_metrics.get("eval_recall", "N/A")
    summary_lines.append(
        f"   • Recall:    {test_recall:.3f}"
        if isinstance(test_recall, (int, float))
        else f"   • Recall:    {test_recall}"
    )
    summary_lines.append("")
    summary_lines.append("💾 MODEL ARTIFACTS:")
    summary_lines.append(f"   • Final model logged to MLflow (artifact: final_model)")
    summary_lines.append(f"   • MLflow experiments:   {mlruns_path}")
    summary_lines.append("")
    summary_lines.append("🚀 NEXT STEPS:")
    summary_lines.append(
        f"   • View MLflow UI: mlflow ui --backend-store-uri {mlruns_path}"
    )
    summary_lines.append("   • Load model: Download from MLflow artifact 'final_model'")

    # Create a special log record for highlighting
    import logging

    record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="\n".join(summary_lines),
        args=(),
        exc_info=None,
    )
    record.highlight = True
    logger.handle(record)


logger = _get_logger()


def load_processed_dataset(cfg: DictConfig) -> Tuple[Dataset, LabelEncoder]:
    # Hydra's get_original_cwd() requires HydraConfig to be initialized when
    # the function is called (e.g. when running tests that bypass the
    # hydra.main wrapper). Use a safe helper that falls back to the current
    # working directory when Hydra isn't initialized.
    try:
        project_root = Path(get_original_cwd())
    except Exception:
        project_root = Path.cwd()
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
    # Allow tests or callers to omit `cfg.dataset` by providing sensible
    # defaults (train/val/test = 0.8/0.1/0.1). This keeps the function robust
    # when invoked with minimal configs in unit tests.
    try:
        train_ratio = cfg.dataset.train_ratio
        val_ratio = cfg.dataset.val_ratio
    except Exception:
        train_ratio = 0.8
        val_ratio = 0.1
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
    # Minimal: compute balanced class weights when requested, otherwise None.
    if not cfg.hparams.use_class_weights:
        return None
    if labels is None or len(labels) == 0:
        logger.warning("No labels found for class weight computation; skipping.")
        return None
    from sklearn.utils.class_weight import compute_class_weight

    present_classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced", classes=present_classes, y=labels
    )
    full = np.ones(len(le.classes_), dtype=np.float32)
    for cls, w in zip(present_classes, weights):
        full[int(cls)] = float(w)
    return torch.tensor(full, dtype=torch.float32)


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
        # Minimal loss override: compute logits and apply CrossEntropyLoss with
        # optional class weights.
        labels = inputs.get("labels")
        device = next(model.parameters()).device
        if labels is not None:
            labels = labels.to(device).long()
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
        )
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(device))
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
    # Build TrainingArguments from a dict and filter to supported params so
    # older/newer transformers versions won't raise on unknown kwargs.

    ta_kwargs = {
        "output_dir": out_dir,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_dir": out_dir,  # keep HF logs (trainer_state, events) inside hydra run dir
        "learning_rate": cfg.hparams.learning_rate,
        "per_device_train_batch_size": cfg.hparams.batch_size,
        "per_device_eval_batch_size": cfg.hparams.batch_size,
        "num_train_epochs": cfg.hparams.num_train_epochs,
        "weight_decay": cfg.hparams.weight_decay,
        "logging_steps": cfg.hparams.eval_steps,
        "save_total_limit": 2,
        "report_to": ["mlflow"],
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "fp16": cfg.hparams.fp16,
        "disable_tqdm": True,
        "lr_scheduler_type": getattr(cfg.hparams, "lr_scheduler_type", "linear"),
        "warmup_ratio": getattr(cfg.hparams, "warmup_ratio", 0.1),
        "max_grad_norm": getattr(cfg.hparams, "max_grad_norm", 1.0),
    }

    # Always enforce 'epoch' for both strategies if best model or early stopping is used
    ta_kwargs["eval_strategy"] = "epoch"
    ta_kwargs["save_strategy"] = "epoch"

    # Filter to parameters actually accepted by TrainingArguments.__init__
    try:
        sig = inspect.signature(TrainingArguments.__init__)
        accepted = set(sig.parameters.keys()) - {"self"}
        filtered = {k: v for k, v in ta_kwargs.items() if k in accepted}
    except Exception:
        filtered = ta_kwargs

    # Always ensure both strategies are present before constructing TrainingArguments
    if "eval_strategy" not in filtered:
        filtered["eval_strategy"] = "epoch"
    if "save_strategy" not in filtered:
        filtered["save_strategy"] = "epoch"

    # Now, align them if needed
    if filtered.get("load_best_model_at_end", False):
        eval_val = filtered.get("eval_strategy")
        save_val = filtered.get("save_strategy")
        if str(eval_val) != str(save_val):
            filtered["eval_strategy"] = "epoch"
            filtered["save_strategy"] = "epoch"

    args = TrainingArguments(**filtered)

    # Compatibility guard: some transformers versions require that
    # `save_strategy` and `evaluation_strategy` match when
    # `load_best_model_at_end=True`. If the user requested loading the
    # best model but the strategies are missing or mismatched (for
    # example because certain kwargs were filtered out), align them to
    # EPOCH to avoid a hard ValueError at runtime.
    try:
        load_best = getattr(args, "load_best_model_at_end", False)
        if load_best:
            eval_strat = getattr(args, "eval_strategy", None)
            save_strat = getattr(args, "save_strategy", None)
            if str(eval_strat).lower() in ("no", "none", None) or (
                str(eval_strat) != str(save_strat)
            ):
                logger.info(
                    "Adjusting TrainingArguments: setting eval_strategy and save_strategy to 'epoch' for compatibility with load_best_model_at_end=True"
                )
                try:
                    args.eval_strategy = "epoch"
                except Exception:
                    setattr(args, "eval_strategy", "epoch")
                try:
                    args.save_strategy = "epoch"
                except Exception:
                    setattr(args, "save_strategy", "epoch")
    except Exception:
        # Best-effort; don't fail if introspection fails.
        pass
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
        # Ensure TrainingArguments has a valid evaluation strategy; the
        # EarlyStoppingCallback asserts that args.eval_strategy != NO. Some
        # transformers versions may have filtered out the original
        # `eval_strategy` kwarg, so set it explicitly here.
        try:
            current_eval = getattr(args, "eval_strategy", None)
            if current_eval is None or str(current_eval).lower() == "no":
                logger.info(
                    "Setting TrainingArguments.eval_strategy='epoch' for EarlyStoppingCallback compatibility"
                )
                try:
                    args.eval_strategy = "epoch"
                except Exception:
                    setattr(args, "eval_strategy", "epoch")
            # Ensure save strategy also aligns when load_best_model_at_end may be used
            try:
                current_save = getattr(args, "save_strategy", None)
                if current_save is None:
                    try:
                        args.save_strategy = "epoch"
                    except Exception:
                        setattr(args, "save_strategy", "epoch")
            except Exception:
                pass
        except Exception:
            # Don't fail setup if introspection or assignments fail
            logger.warning(
                "Could not enforce eval/save strategy compatibility for EarlyStoppingCallback"
            )

        trainer_args.setdefault("callbacks", [])
        trainer_args["callbacks"].append(
            EarlyStoppingCallback(early_stopping_patience=int(patience))
        )
        logger.info(f"Early stopping enabled with patience={patience}.")

    return trainer_cls(**trainer_args)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Allow legacy overrides that address keys not present in the base config
    OmegaConf.set_struct(cfg, False)

    set_seed(cfg.hparams.seed)
    try:
        project_root = Path(get_original_cwd())
    except Exception:
        project_root = Path.cwd()

    # Use Hydra's run directory instead of manual creation
    from hydra.core.hydra_config import HydraConfig

    out_dir = Path(HydraConfig.get().runtime.output_dir)

    # Persist resolved config inside the run dir for reproducibility
    try:
        OmegaConf.save(cfg, out_dir / "cfg.yaml")
    except Exception:
        pass

    # Prefer environment var, then explicit config URI, otherwise use a
    # canonical project-local mlruns path (hardcoded here to avoid making it
    # a hyperparameter). This ensures a single tracking store for all jobs.
    job = locals().get("job", getattr(cfg, "mode", None) or "bert")
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    canonical_dir = project_root / "mlruns"  # Use same directory as GCN training
    canonical_uri = f"file:{canonical_dir}"
    if env_uri:
        mlflow.set_tracking_uri(env_uri)
        logger.info(
            f"Using MLflow tracking URI from MLFLOW_TRACKING_URI env: {env_uri}"
        )
    elif cfg.get("mlflow_tracking_uri"):
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        logger.info(f"Using configured MLflow tracking URI: {cfg.mlflow_tracking_uri}")
    else:
        # Enforce the canonical mlruns location inside the project root.
        canonical_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(canonical_uri)
        logger.info(f"Using canonical MLflow tracking URI: {canonical_uri}")
    mlflow.set_experiment("train_bert")
    # Enable MLflow autologging for transformers (fallback to pytorch). This
    # helps capture parameters, metrics and artifacts automatically when
    # available.
    autolog_enabled = False
    autolog_flavor = None
    try:
        import mlflow.transformers as _mlt

        _mlt.autolog()
        autolog_enabled = True
        autolog_flavor = "transformers"
        logger.info("✓ MLflow transformers autolog enabled")
    except Exception:
        try:
            import mlflow.pytorch as _mlp

            _mlp.autolog()
            autolog_enabled = True
            autolog_flavor = "pytorch"
            logger.info("✓ MLflow pytorch autolog enabled")
        except Exception:
            logger.info("ℹ️  MLflow autolog not available; using manual logging")

    logger.info(f"📁 Hydra run directory: {out_dir}")
    logger.info(f"🔗 MLflow tracking URI: {mlflow.get_tracking_uri()}")

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

    # Tag run and start MLflow run with deterministic run_name
    try:
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=str(project_root)
            )
            .decode()
            .strip()
        )
    except Exception:
        git_sha = "unknown"

    with mlflow.start_run(run_name=f"{cfg.mode}_{out_dir.name}"):
        try:
            mlflow.set_tags(
                {
                    "run_dir": str(out_dir),
                    "entry_script": "finetune_bert",
                    "git_sha": git_sha,
                }
            )
        except Exception:
            pass
        # Log resolved config for reproducibility
        try:
            mlflow.log_artifact(str(out_dir / "cfg.yaml"), artifact_path="config")
        except Exception:
            pass
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

        # Avoid duplicate logging if MLflow autolog captured params/metrics.
        if not autolog_enabled:
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
        else:
            logger.info("✓ Parameters logged to MLflow automatically")
        logger.info("🚀 Starting training...")
        trainer.train()

        # Create a section break for evaluation results
        import logging

        record = logging.LogRecord(
            name=logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="📊 EVALUATION RESULTS",
            args=(),
            exc_info=None,
        )
        record.section = True
        logger.handle(record)

        logger.info("🏆 Validation metrics (best model):")
        val_metrics = trainer.evaluate()
        logger.info(
            _format_metrics(
                {
                    k.replace("eval_", ""): v
                    for k, v in val_metrics.items()
                    if k.startswith("eval_")
                },
                "Validation Performance",
            )
        )

        if not autolog_enabled:
            mlflow.log_metrics(
                {
                    f"val_{k}": float(v)
                    for k, v in val_metrics.items()
                    if isinstance(v, (int, float))
                }
            )
        else:
            logger.info("✓ Metrics logged to MLflow automatically")

        logger.info("🧪 Evaluating on test set...")
        test_metrics = trainer.evaluate(test_ds)
        logger.info(
            _format_metrics(
                {
                    k.replace("eval_", ""): v
                    for k, v in test_metrics.items()
                    if k.startswith("eval_")
                },
                "Test Set Performance",
            )
        )

        if not autolog_enabled:
            mlflow.log_metrics(
                {
                    f"test_{k}": float(v)
                    for k, v in test_metrics.items()
                    if isinstance(v, (int, float))
                }
            )
        else:
            logger.info("✓ Test metrics logged to MLflow automatically")

        # Confusion matrix analysis
        logger.info("📈 Confusion Matrix Analysis:")
        from sklearn.metrics import confusion_matrix

        # Trainer.predict in tests may receive a torch.utils.data.Subset which
        # doesn't expose HF Dataset attributes like `column_names`. Convert
        # back to a datasets.Dataset when possible so monkeypatched trainers
        # that inspect `column_names` or indexing work correctly.
        if isinstance(test_ds, Subset) and hasattr(dataset, "select"):
            predict_indices = [int(i) for i in test_ds.indices]
            predict_ds = dataset.select(predict_indices)
        else:
            predict_ds = test_ds

        test_output = trainer.predict(predict_ds)
        # Be defensive: some dummy trainers or unexpected trainer impls may
        # return objects without `predictions` or `label_ids` attributes.
        preds = getattr(test_output, "predictions", None)
        label_ids = getattr(test_output, "label_ids", None)
        if preds is None:
            logger.warning("⚠️  Trainer.predict returned no predictions; using zeros.")
            preds = np.zeros((len(predict_ds), 1))
        if label_ids is None:
            logger.warning("⚠️  Trainer.predict returned no label_ids; using zeros.")
            label_ids = np.zeros(len(predict_ds), dtype=int)
        preds = np.argmax(preds, axis=1)
        labels = label_ids
        # Provide the full set of label indices so the confusion matrix has a
        # consistent shape even when a particular split contains only a subset
        # of classes (avoids sklearn warning about single-label inputs).
        all_label_indices = list(range(len(le.classes_)))
        cm = confusion_matrix(labels, preds, labels=all_label_indices)

        # Display confusion matrix in console
        logger.info(_format_confusion_matrix(cm, le.classes_))

        # Create evaluation subfolder (cosmetic / organizational); we do not
        # persist the confusion matrix JSON locally to avoid redundant files.
        (out_dir / "evaluation").mkdir(exist_ok=True)
        try:
            mlflow.log_dict(
                {"matrix": cm.tolist(), "labels": le.classes_.tolist()},
                artifact_file="evaluation/confusion_matrix.json",
            )
            logger.info("✓ Confusion matrix saved to MLflow")
        except Exception:
            # Fallback: if log_dict unavailable, write temp file and log then remove
            import json
            import tempfile

            tmp_cm = tempfile.NamedTemporaryFile("w", suffix="_cm.json", delete=False)
            with tmp_cm as f:
                json.dump(
                    {"matrix": cm.tolist(), "labels": le.classes_.tolist()},
                    f,
                    indent=2,
                )
            mlflow.log_artifact(tmp_cm.name, artifact_path="evaluation")
            try:
                Path(tmp_cm.name).unlink()
            except Exception:
                pass

        # Prefer MLflow-native model logging. Assume mlflow.transformers is available.
        registered_name = (
            cfg.mlflow_model_name
            if hasattr(cfg, "mlflow_model_name")
            else (
                cfg.hparams.mlflow_model_name
                if hasattr(cfg.hparams, "mlflow_model_name")
                else None
            )
        )

        logged = False
        # Prefer mlflow.transformers but fall back to mlflow.pytorch or a
        # local save if necessary. Wrap in try/except so environments with
        # partial MLflow support don't crash the run.
        if not autolog_enabled:
            try:
                import mlflow.transformers as mlt

                mlt.log_model(
                    transformers_model=model,
                    artifact_path="model",
                    tokenizer=tokenizer,
                    registered_model_name=registered_name,
                )
                logger.info("Logged model with mlflow.transformers.log_model()")
                logged = True
            except Exception as e1:
                logger.warning(f"mlflow.transformers.log_model failed: {e1}")
                try:
                    import mlflow.pytorch as mltp

                    mltp.log_model(pytorch_model=model, artifact_path="model")
                    logger.info("Logged model with mlflow.pytorch.log_model()")
                    logged = True
                except Exception as e2:
                    logger.warning(f"mlflow.pytorch.log_model failed: {e2}")
                    # As a final fallback, save locally (if allowed) so the
                    # run still produces an artifact we can inspect or upload
                    # manually later.
                    if keep_local:
                        try:
                            trainer.save_model(str(out_dir))
                            tokenizer.save_pretrained(str(out_dir))
                            logger.info("Saved local model/tokenizer as fallback")
                            logged = True
                        except Exception as e3:
                            logger.error(f"Local save fallback failed: {e3}")
        else:
            logger.info("Autolog enabled; skipping explicit model log")

        # Respect cfg.hparams.keep_local_copy if present
        keep_local = getattr(cfg.hparams, "keep_local_copy", False)

        # Save final model to temp dir and log to MLflow (MLflow as source of truth)
        param_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_dir = None
        with TemporaryDirectory() as temp_dir:
            final_dir = Path(temp_dir) / f"final_model_{param_str}"
            final_dir.mkdir()
            try:
                trainer.save_model(str(final_dir))
                tokenizer.save_pretrained(str(final_dir))
                # Patch config.json to ensure model_type is 'bert' for compatibility
                import json

                config_path = final_dir / "config.json"
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config_data = json.load(f)
                    config_data["model_type"] = "bert"
                    with open(config_path, "w") as f:
                        json.dump(config_data, f, indent=2)
                # Save label encoder
                le_path = final_dir / "label_encoder.joblib"
                joblib.dump(le, le_path)
                # Log to MLflow
                mlflow.log_artifacts(str(final_dir), artifact_path="final_model")
                logger.info(f"💾 Final model logged to MLflow (artifact: final_model)")
            except Exception as e:
                logger.warning(f"⚠️  Could not log final model to MLflow: {e}")

        # Log comprehensive training summary with the actual MLflow tracking URI
        mlruns_path = "mlruns"
        _log_training_summary(cfg, val_metrics, test_metrics, final_dir, mlruns_path)

    return 0


if __name__ == "__main__":
    main()
