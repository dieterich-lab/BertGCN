"""Configuration management utilities for BertGCN."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class BertConfig:
    """Configuration for BERT fine-tuning."""

    # Model settings
    pretrained_model: str = "/prj/doctoral_letters/PETGUI/med_bert_local"
    max_length: int = 512
    dropout: float = 0.1

    # Training settings
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_clip_val: float = 1.0

    # Data settings
    doclevel: str = "letter"
    clean: bool = True
    nomeds: bool = False
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2

    # Trainer settings
    precision: int = 16
    enable_progress_bar: bool = True
    log_every_n_steps: int = 10
    val_check_interval: float = 0.25

    # Early stopping
    early_stopping_monitor: str = "val_f1"
    early_stopping_patience: int = 5
    early_stopping_mode: str = "max"
    early_stopping_min_delta: float = 0.001

    # Checkpointing
    checkpoint_monitor: str = "val_f1"
    checkpoint_mode: str = "max"
    checkpoint_save_top_k: int = 3
    checkpoint_save_last: bool = True

    # Logging
    experiment_name: str = "bert_finetune"
    log_dir: str = "bert_logs"


@dataclass
class BertGCNConfig:
    """Configuration for BertGCN training."""

    # Model settings
    pretrained_model: str = "/prj/doctoral_letters/PETGUI/med_bert_local"
    mix_factor: float = 0.7
    gcn_layers: int = 2
    hidden_dim: int = 200
    dropout: float = 0.5

    # Training settings
    epochs: int = 50
    batch_size: int = 1
    bert_lr: float = 1e-5
    gcn_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_clip_val: float = 1.0

    # Data settings
    doclevel: str = "letter"
    clean: bool = True
    testunklar: bool = False

    # Graph settings
    window_size: int = 20
    min_word_freq: int = 1

    # Trainer settings
    precision: int = 16
    enable_progress_bar: bool = True
    log_every_n_steps: int = 5
    val_check_interval: float = 0.5

    # Early stopping
    early_stopping_monitor: str = "val_f1"
    early_stopping_patience: int = 5
    early_stopping_mode: str = "max"
    early_stopping_min_delta: float = 0.001

    # Checkpointing
    checkpoint_monitor: str = "val_f1"
    checkpoint_mode: str = "max"
    checkpoint_save_top_k: int = 3
    checkpoint_save_last: bool = True

    # Logging
    experiment_name: str = "bertgcn_train"
    log_dir: str = "bertgcn_logs"


def load_config(config_path: str, config_type: str = "bert") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    return config_data


def create_bert_config_from_file(config_path: str) -> BertConfig:
    """Create BertConfig from YAML file."""
    config_data = load_config(config_path, "bert")

    # Flatten nested structure
    flattened = {}

    # Model settings
    if "model" in config_data:
        model_config = config_data["model"]
        flattened.update(
            {
                "pretrained_model": model_config.get(
                    "pretrained_model", BertConfig.pretrained_model
                ),
                "max_length": model_config.get("max_length", BertConfig.max_length),
                "dropout": model_config.get("dropout", BertConfig.dropout),
            }
        )

    # Training settings
    if "training" in config_data:
        training_config = config_data["training"]
        flattened.update(
            {
                "epochs": training_config.get("epochs", BertConfig.epochs),
                "batch_size": training_config.get("batch_size", BertConfig.batch_size),
                "learning_rate": training_config.get(
                    "learning_rate", BertConfig.learning_rate
                ),
                "weight_decay": training_config.get(
                    "weight_decay", BertConfig.weight_decay
                ),
                "warmup_steps": training_config.get(
                    "warmup_steps", BertConfig.warmup_steps
                ),
                "gradient_clip_val": training_config.get(
                    "gradient_clip_val", BertConfig.gradient_clip_val
                ),
            }
        )

    # Data settings
    if "data" in config_data:
        data_config = config_data["data"]
        flattened.update(
            {
                "doclevel": data_config.get("doclevel", BertConfig.doclevel),
                "clean": data_config.get("clean", BertConfig.clean),
                "nomeds": data_config.get("nomeds", BertConfig.nomeds),
                "train_ratio": data_config.get("train_ratio", BertConfig.train_ratio),
                "val_ratio": data_config.get("val_ratio", BertConfig.val_ratio),
                "test_ratio": data_config.get("test_ratio", BertConfig.test_ratio),
            }
        )

    # Trainer settings
    if "trainer" in config_data:
        trainer_config = config_data["trainer"]
        flattened.update(
            {
                "precision": trainer_config.get("precision", BertConfig.precision),
                "enable_progress_bar": trainer_config.get(
                    "enable_progress_bar", BertConfig.enable_progress_bar
                ),
                "log_every_n_steps": trainer_config.get(
                    "log_every_n_steps", BertConfig.log_every_n_steps
                ),
                "val_check_interval": trainer_config.get(
                    "val_check_interval", BertConfig.val_check_interval
                ),
            }
        )

    # Early stopping
    if "early_stopping" in config_data:
        es_config = config_data["early_stopping"]
        flattened.update(
            {
                "early_stopping_monitor": es_config.get(
                    "monitor", BertConfig.early_stopping_monitor
                ),
                "early_stopping_patience": es_config.get(
                    "patience", BertConfig.early_stopping_patience
                ),
                "early_stopping_mode": es_config.get(
                    "mode", BertConfig.early_stopping_mode
                ),
                "early_stopping_min_delta": es_config.get(
                    "min_delta", BertConfig.early_stopping_min_delta
                ),
            }
        )

    # Checkpointing
    if "checkpointing" in config_data:
        cp_config = config_data["checkpointing"]
        flattened.update(
            {
                "checkpoint_monitor": cp_config.get(
                    "monitor", BertConfig.checkpoint_monitor
                ),
                "checkpoint_mode": cp_config.get("mode", BertConfig.checkpoint_mode),
                "checkpoint_save_top_k": cp_config.get(
                    "save_top_k", BertConfig.checkpoint_save_top_k
                ),
                "checkpoint_save_last": cp_config.get(
                    "save_last", BertConfig.checkpoint_save_last
                ),
            }
        )

    # Logging
    if "logging" in config_data:
        log_config = config_data["logging"]
        flattened.update(
            {
                "experiment_name": log_config.get(
                    "experiment_name", BertConfig.experiment_name
                ),
                "log_dir": log_config.get("log_dir", BertConfig.log_dir),
            }
        )

    return BertConfig(**flattened)


def create_bertgcn_config_from_file(config_path: str) -> BertGCNConfig:
    """Create BertGCNConfig from YAML file."""
    config_data = load_config(config_path, "bertgcn")

    # Flatten nested structure
    flattened = {}

    # Model settings
    if "model" in config_data:
        model_config = config_data["model"]
        flattened.update(
            {
                "pretrained_model": model_config.get(
                    "pretrained_model", BertGCNConfig.pretrained_model
                ),
                "mix_factor": model_config.get("mix_factor", BertGCNConfig.mix_factor),
                "gcn_layers": model_config.get("gcn_layers", BertGCNConfig.gcn_layers),
                "hidden_dim": model_config.get("hidden_dim", BertGCNConfig.hidden_dim),
                "dropout": model_config.get("dropout", BertGCNConfig.dropout),
            }
        )

    # Training settings
    if "training" in config_data:
        training_config = config_data["training"]
        flattened.update(
            {
                "epochs": training_config.get("epochs", BertGCNConfig.epochs),
                "batch_size": training_config.get(
                    "batch_size", BertGCNConfig.batch_size
                ),
                "bert_lr": training_config.get("bert_lr", BertGCNConfig.bert_lr),
                "gcn_lr": training_config.get("gcn_lr", BertGCNConfig.gcn_lr),
                "weight_decay": training_config.get(
                    "weight_decay", BertGCNConfig.weight_decay
                ),
                "warmup_steps": training_config.get(
                    "warmup_steps", BertGCNConfig.warmup_steps
                ),
                "gradient_clip_val": training_config.get(
                    "gradient_clip_val", BertGCNConfig.gradient_clip_val
                ),
            }
        )

    # Data settings
    if "data" in config_data:
        data_config = config_data["data"]
        flattened.update(
            {
                "doclevel": data_config.get("doclevel", BertGCNConfig.doclevel),
                "clean": data_config.get("clean", BertGCNConfig.clean),
                "testunklar": data_config.get("testunklar", BertGCNConfig.testunklar),
            }
        )

    # Graph settings
    if "graph" in config_data:
        graph_config = config_data["graph"]
        flattened.update(
            {
                "window_size": graph_config.get(
                    "window_size", BertGCNConfig.window_size
                ),
                "min_word_freq": graph_config.get(
                    "min_word_freq", BertGCNConfig.min_word_freq
                ),
            }
        )

    # Trainer settings
    if "trainer" in config_data:
        trainer_config = config_data["trainer"]
        flattened.update(
            {
                "precision": trainer_config.get("precision", BertGCNConfig.precision),
                "enable_progress_bar": trainer_config.get(
                    "enable_progress_bar", BertGCNConfig.enable_progress_bar
                ),
                "log_every_n_steps": trainer_config.get(
                    "log_every_n_steps", BertGCNConfig.log_every_n_steps
                ),
                "val_check_interval": trainer_config.get(
                    "val_check_interval", BertGCNConfig.val_check_interval
                ),
            }
        )

    # Early stopping
    if "early_stopping" in config_data:
        es_config = config_data["early_stopping"]
        flattened.update(
            {
                "early_stopping_monitor": es_config.get(
                    "monitor", BertGCNConfig.early_stopping_monitor
                ),
                "early_stopping_patience": es_config.get(
                    "patience", BertGCNConfig.early_stopping_patience
                ),
                "early_stopping_mode": es_config.get(
                    "mode", BertGCNConfig.early_stopping_mode
                ),
                "early_stopping_min_delta": es_config.get(
                    "min_delta", BertGCNConfig.early_stopping_min_delta
                ),
            }
        )

    # Checkpointing
    if "checkpointing" in config_data:
        cp_config = config_data["checkpointing"]
        flattened.update(
            {
                "checkpoint_monitor": cp_config.get(
                    "monitor", BertGCNConfig.checkpoint_monitor
                ),
                "checkpoint_mode": cp_config.get("mode", BertGCNConfig.checkpoint_mode),
                "checkpoint_save_top_k": cp_config.get(
                    "save_top_k", BertGCNConfig.checkpoint_save_top_k
                ),
                "checkpoint_save_last": cp_config.get(
                    "save_last", BertGCNConfig.checkpoint_save_last
                ),
            }
        )

    # Logging
    if "logging" in config_data:
        log_config = config_data["logging"]
        flattened.update(
            {
                "experiment_name": log_config.get(
                    "experiment_name", BertGCNConfig.experiment_name
                ),
                "log_dir": log_config.get("log_dir", BertGCNConfig.log_dir),
            }
        )

    return BertGCNConfig(**flattened)


def get_default_config_path(config_name: str) -> str:
    """Get default config path for a given config name."""
    project_root = Path(__file__).parent.parent.parent
    return str(project_root / "configs" / f"{config_name}.yaml")
