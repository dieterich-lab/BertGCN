"""Minimal BERT-GCN Training with Hydra and MLflow.

Educational example demonstrating:
- Hydra configuration management
- MLflow experiment tracking
- PyTorch GCN training
- Real dataset integration

This version uses the same processed dataset as finetune_bert.py
"""

# Suppress all warnings and logging messages
import os
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

warnings.simplefilter("ignore")
import json
import logging
import sys

logging.getLogger().setLevel(logging.ERROR)
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

import logging
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import joblib
import mlflow
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric import nn as pyg_nn
from torch_geometric.utils import dense_to_sparse
from transformers import AutoModel, AutoTokenizer

# Reuse the plain BERT finetuning pipeline when mix_factor=0 for parity.
from bertgcn import train_bert as tb
from bertgcn.utils import load_corpus


def _get_logger():
    """Get a beautifully formatted logger for GCN training."""

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

    logger = logging.getLogger("train_gcn")
    logger.setLevel(logging.INFO)

    # Avoid duplicate entries from parent/root handlers (Hydra config can attach its own).
    logger.propagate = False

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add console handler with colored formatting
    # Send logs to stdout so Slurm captures them in *.log (not *.err).
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def _format_metrics_gcn(epoch, loss, train_acc, val_acc, max_epochs):
    """Format GCN training metrics into a clean display."""
    progress = f"Epoch {epoch+1}/{max_epochs}"
    metrics = f"Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
    return f"{progress} | {metrics}"


def _log_gcn_training_summary(test_acc, final_dir, mlruns_path, logger):
    """Log a comprehensive GCN training summary."""
    summary_lines = []
    summary_lines.append("🎯 GCN TRAINING COMPLETED SUCCESSFULLY")
    summary_lines.append("")
    summary_lines.append("📊 FINAL TEST PERFORMANCE:")
    summary_lines.append(f"   • Test Accuracy: {test_acc:.1%}")
    summary_lines.append("")
    summary_lines.append("💾 MODEL ARTIFACTS:")
    summary_lines.append(f"   • Final model saved to: {final_dir}")
    summary_lines.append(f"   • MLflow experiments:   {mlruns_path}")
    summary_lines.append("")
    summary_lines.append("🚀 NEXT STEPS:")
    summary_lines.append(
        f"   • View MLflow UI: mlflow ui --backend-store-uri {mlruns_path}"
    )
    summary_lines.append(
        "   • Load model: BertGCN.from_pretrained() with saved state_dict"
    )

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


class SimpleGCN(nn.Module):
    """Simple 2-layer GCN for node classification using PyTorch Geometric."""

    def __init__(
        self, n_features: int, n_hidden: int, n_classes: int, dropout: float = 0.5
    ):
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(n_features, n_hidden)
        self.conv2 = pyg_nn.GCNConv(n_hidden, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass through GCN."""
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class BertGCN(nn.Module):
    """BertGCN model combining BERT and GCN predictions."""

    def __init__(
        self,
        pretrained_model: str,
        nb_class: int,
        m: float = 0.7,
        gcn_layers: int = 2,
        n_hidden: int = 32,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = self.bert_model.config.hidden_size
        self.classifier = nn.Linear(self.feat_dim, nb_class)
        self.gcn = SimpleGCN(
            n_features=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        idx: torch.Tensor = None,
    ):
        # BERT prediction
        if input_ids is not None and attention_mask is not None:
            cls_feats = self.bert_model(
                input_ids=input_ids, attention_mask=attention_mask
            )[0][:, 0]
            bert_logit = self.classifier(cls_feats)
            # When mixing is disabled (m=0), short-circuit to pure BERT for docs
            if self.m == 0:
                return F.log_softmax(bert_logit, dim=1)
            bert_pred = F.softmax(bert_logit, dim=1)
        else:
            bert_pred = None

        # GCN prediction
        gcn_logit = self.gcn(x, edge_index, edge_weight)
        gcn_pred = torch.exp(gcn_logit)  # since log_softmax, exp to get softmax

        if bert_pred is not None and idx is not None:
            # Ensemble for the batch
            pred = gcn_pred * self.m
            pred[idx] = pred[idx] + bert_pred * (1 - self.m)
            return torch.log(pred[idx])
        else:
            # Full prediction
            if bert_pred is not None:
                pred = gcn_pred * self.m + bert_pred * (1 - self.m)
            else:
                pred = gcn_pred
            return torch.log(pred)


def load_processed_dataset(cfg: DictConfig) -> Tuple[Dataset, LabelEncoder]:
    """Load the processed dataset (same as finetune_bert.py)."""
    try:
        project_root = Path(get_original_cwd())
    except Exception:
        project_root = Path.cwd()
    processed_dir = project_root / "data" / "processed"
    dataset_path = processed_dir / "tokenized_dataset"
    le_path = processed_dir / "label_encoder.joblib"

    if not dataset_path.exists() or not le_path.exists():
        raise FileNotFoundError(
            f"Missing processed data in {processed_dir}. Run preprocessing first."
        )

    ds = load_from_disk(str(dataset_path))
    le: LabelEncoder = joblib.load(le_path)

    # Keep only the columns we need for GCN
    gcn_ds = ds.remove_columns(
        [c for c in ["text", "medication_name"] if c in ds.column_names]
    )
    gcn_ds.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels", "med_id"]
    )

    return gcn_ds, le


def _check_graph_alignment(
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    labels_onehot: np.ndarray,
    splits: Dict[str, Any] | None,
):
    """Assert graph ordering and label coverage match expectations.

    The graph is expected to order nodes as [train docs][words][val docs][test docs].
    This validates mask contiguity, split metadata counts, and label presence for docs.
    """

    logger = logging.getLogger("train_gcn")

    train_idx = torch.where(torch.tensor(train_mask))[0]
    val_idx = torch.where(torch.tensor(val_mask))[0]
    test_idx = torch.where(torch.tensor(test_mask))[0]

    train_count = int(train_mask.sum())
    val_count = int(val_mask.sum())
    test_count = int(test_mask.sum())
    word_count = int((~(train_mask | val_mask | test_mask)).sum())

    val_offset = train_count + word_count
    test_offset = val_offset + val_count

    expected_train = torch.arange(train_count)
    expected_val = torch.arange(val_offset, val_offset + val_count)
    expected_test = torch.arange(test_offset, test_offset + test_count)

    if not torch.equal(train_idx, expected_train):
        raise ValueError(
            "Train mask indices are not contiguous from 0; graph/doc ordering likely misaligned."
        )
    if not torch.equal(val_idx, expected_val):
        raise ValueError(
            "Val mask indices do not match expected offset after word nodes; rebuild graph."
        )
    if not torch.equal(test_idx, expected_test):
        raise ValueError(
            "Test mask indices do not match expected offset after val nodes; rebuild graph."
        )

    # Validate split metadata counts and uniqueness if provided
    if splits:
        for name, arr, expected_len in (
            ("train_idx", splits.get("train_idx", []), train_count),
            ("val_idx", splits.get("val_idx", []), val_count),
            ("test_idx", splits.get("test_idx", []), test_count),
        ):
            if len(arr) != expected_len:
                raise ValueError(
                    f"Split metadata {name} length {len(arr)} != expected {expected_len}; rebuild graph."
                )
            if len(set(arr)) != len(arr):
                raise ValueError(
                    f"Split metadata {name} contains duplicates; rebuild graph."
                )

    # Ensure every doc node has a label (one-hot rows must sum to 1)
    labels_sum = labels_onehot.sum(axis=1)
    for name, idx in ("train", train_idx), ("val", val_idx), ("test", test_idx):
        if np.any(labels_sum[idx] == 0):
            raise ValueError(
                f"{name} split contains unlabeled docs; inspect label encoder or graph build."
            )

    logger.info(
        "Graph alignment check passed: train=%d val=%d test=%d word=%d (val_offset=%d, test_offset=%d)",
        train_count,
        val_count,
        test_count,
        word_count,
        val_offset,
        test_offset,
    )


def _create_random_graph_data(
    dataset: Dataset, cfg: DictConfig
) -> Dict[str, torch.Tensor]:
    """Create a toy graph dataset used for testing and quick experiments."""
    n_nodes = len(dataset)

    # Create random adjacency matrix (normalized)
    adj = torch.randn(n_nodes, n_nodes)
    adj = (adj + adj.T) / 2  # Make symmetric
    adj = F.normalize(adj, p=1, dim=1)  # Row-normalize

    # Use random features for now (in practice, you'd use BERT embeddings)
    features = torch.randn(n_nodes, cfg.model.n_features)

    # Get labels
    labels = dataset["labels"]
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        labels = labels.clone().detach().long()

    # Create train/val/test splits (simple random split)
    indices = torch.randperm(n_nodes)
    n_train = int(cfg.data.train_ratio * n_nodes)
    n_val = int(cfg.data.val_ratio * n_nodes)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[indices[:n_train]] = True
    val_mask[indices[n_train : n_train + n_val]] = True
    test_mask[indices[n_train + n_val :]] = True

    return {
        "adj": adj,
        "features": features,
        "labels": labels,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }


def load_graph_data_from_disk(cfg: DictConfig) -> Dict[str, torch.Tensor]:
    """Load the document-word graph produced by `build_graph`."""
    dataset_str = cfg.data.get("graph_dataset")
    if not dataset_str:
        raise ValueError(
            "`cfg.data.graph_dataset` must point to the saved graph files."
        )

    check_graph = bool(getattr(cfg, "check_graph", False))

    try:
        project_root = Path(get_original_cwd())
    except Exception:
        project_root = Path.cwd()
    dataset_path = project_root / dataset_str
    meta_path = Path(f"{dataset_path}.splits.json")
    splits = None
    graph_indicator = Path(f"{dataset_path}.x")
    if not graph_indicator.exists():
        raise FileNotFoundError(
            f"Graph dataset prefix not found at {dataset_path}. Run `poetry run build-graph` first."
        )

    logging.getLogger(__name__).info(f"Loading graph data from {dataset_path}")
    print("Starting load_corpus", flush=True)
    (
        adj,
        features,
        y_train,
        y_val,
        y_test,
        train_mask,
        val_mask,
        test_mask,
        train_size,
        test_size,
    ) = load_corpus(str(dataset_path))
    print("load_corpus done", flush=True)

    print("Starting toarray", flush=True)
    features_array = (
        features.toarray() if hasattr(features, "toarray") else np.array(features)
    )
    print("toarray done", flush=True)
    labels_onehot = y_train + y_val + y_test

    # Set features for word nodes (default: random; optional: zeros for parity with original).
    # Masks are ordered as: train docs -> word nodes -> val docs -> test docs.
    doc_mask = train_mask | val_mask | test_mask
    word_mask = ~doc_mask
    zero_word_features = bool(
        getattr(cfg.training, "zero_word_features", False)
        or getattr(cfg, "zero_word_features", False)
    )
    if zero_word_features:
        print("leaving word features as zeros (parity mode)", flush=True)
        features_array[word_mask] = 0.0
    else:
        print("setting random features", flush=True)
        random_feats = np.random.randn(word_mask.sum(), features_array.shape[1]).astype(
            np.float32
        )
        features_array[word_mask] = random_feats
        print("random done", flush=True)

    # Symmetrize and add self-loops to match original
    adj = adj + adj.T
    adj = adj + sp.eye(adj.shape[0])

    # Normalize adjacency matrix sparsely for GCN: D^{-1/2} A D^{-1/2}
    print("computing degrees", flush=True)
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees = np.clip(degrees, 1e-10, None)  # avoid division by zero
    d_inv_sqrt = np.power(degrees, -0.5)
    print("degrees done", flush=True)

    # Get sparse indices and data
    row, col = adj.nonzero()
    data = adj.data
    # Normalize edge weights
    new_data = data * d_inv_sqrt[row] * d_inv_sqrt[col]
    print("normalization done", flush=True)

    # Convert to PyTorch tensors
    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
    edge_weight = torch.tensor(new_data, dtype=torch.float32)
    print("sparse conversion done", flush=True)

    # Load tokenized dataset for input_ids and attention_mask and scatter into graph order
    processed_dir = project_root / "data" / "processed"
    dataset_path_hf = processed_dir / "tokenized_dataset"
    if dataset_path_hf.exists():
        ds = load_from_disk(str(dataset_path_hf))
        tokens_input_ids = torch.stack([torch.tensor(ids) for ids in ds["input_ids"]])
        tokens_attention = torch.stack(
            [torch.tensor(mask) for mask in ds["attention_mask"]]
        )

        nb_node = adj.shape[0]
        seq_len = tokens_input_ids.shape[1]

        # Expect total docs to match train+val+test counts
        total_docs = train_mask.sum() + val_mask.sum() + test_mask.sum()
        if tokens_input_ids.shape[0] != total_docs:
            raise ValueError(
                f"Tokenized dataset docs ({tokens_input_ids.shape[0]}) != graph docs ({total_docs})"
            )

        # Default contiguous ordering: [train docs][words][val docs][test docs]
        doc_mask = train_mask | val_mask | test_mask
        word_mask = ~doc_mask
        train_count = int(train_mask.sum())
        val_count = int(val_mask.sum())
        test_count = int(test_mask.sum())
        word_count = int(word_mask.sum())
        val_offset = train_count + word_count
        test_offset = val_offset + val_count

        # Reorder token sequences using saved split metadata (produced by build_graph)
        if meta_path.exists():
            splits = json.loads(meta_path.read_text())
            train_order = torch.tensor(splits.get("train_idx", []), dtype=torch.long)
            val_order = torch.tensor(splits.get("val_idx", []), dtype=torch.long)
            test_order = torch.tensor(splits.get("test_idx", []), dtype=torch.long)

            if (
                len(train_order) != train_count
                or len(val_order) != val_count
                or len(test_order) != test_count
            ):
                raise ValueError(
                    "Split metadata counts do not match graph masks; rebuild graph and processed data."
                )

            # Gather tokens in the exact order used to build the graph
            train_tokens = tokens_input_ids[train_order]
            val_tokens = tokens_input_ids[val_order]
            test_tokens = tokens_input_ids[test_order]

            train_attention = tokens_attention[train_order]
            val_attention = tokens_attention[val_order]
            test_attention = tokens_attention[test_order]
        else:
            # Fallback to legacy contiguous ordering (may misalign if graph was shuffled)
            train_tokens = tokens_input_ids[:train_count]
            val_tokens = tokens_input_ids[train_count : train_count + val_count]
            test_tokens = tokens_input_ids[train_count + val_count :]

            train_attention = tokens_attention[:train_count]
            val_attention = tokens_attention[train_count : train_count + val_count]
            test_attention = tokens_attention[train_count + val_count :]

        if check_graph:
            _check_graph_alignment(
                train_mask, val_mask, test_mask, labels_onehot, splits
            )

        input_ids = torch.zeros((nb_node, seq_len), dtype=torch.long)
        attention_mask = torch.zeros((nb_node, seq_len), dtype=torch.long)

        # Scatter tokens into the positions indicated by masks (graph order: train -> words -> val -> test)
        train_idx = torch.where(torch.tensor(train_mask))[0]
        val_idx = torch.where(torch.tensor(val_mask))[0]
        test_idx = torch.where(torch.tensor(test_mask))[0]

        input_ids[train_idx] = train_tokens
        attention_mask[train_idx] = train_attention

        input_ids[val_idx] = val_tokens
        attention_mask[val_idx] = val_attention

        input_ids[test_idx] = test_tokens
        attention_mask[test_idx] = test_attention
    else:
        if check_graph:
            raise FileNotFoundError(
                "Tokenized dataset missing while check_graph is enabled; run preprocess first."
            )
        input_ids = torch.zeros((adj.shape[0], 512), dtype=torch.long)  # dummy
        attention_mask = torch.zeros((adj.shape[0], 512), dtype=torch.long)

    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "features": torch.tensor(features_array, dtype=torch.float32),
        "labels": torch.tensor(np.argmax(labels_onehot, axis=1), dtype=torch.long),
        "train_mask": torch.tensor(train_mask, dtype=torch.bool),
        "val_mask": torch.tensor(val_mask, dtype=torch.bool),
        "test_mask": torch.tensor(test_mask, dtype=torch.bool),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def update_feature(model, data, device):
    """Update BERT features for documents."""
    model.eval()
    doc_mask = data["train_mask"] | data["val_mask"] | data["test_mask"]
    input_ids = data["input_ids"][doc_mask].to(device)
    attention_mask = data["attention_mask"][doc_mask].to(device)

    with torch.no_grad():
        bert_output = model.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_feats = bert_output.last_hidden_state[:, 0]

    # Update features
    data["features"][doc_mask] = cls_feats.cpu()
    return data


def update_features(
    model: BertGCN, data: Dict[str, torch.Tensor], device: torch.device
):
    """Update document features using the current BERT encoder (no gradients)."""
    print("Starting update_features", flush=True)

    model.eval()
    doc_mask = data["train_mask"] | data["val_mask"] | data["test_mask"]
    doc_indices = torch.where(doc_mask)[0]

    batch_size = 128  # Larger batch for speed
    cls_list = []
    with torch.no_grad():
        for i in range(0, len(doc_indices), batch_size):
            batch_indices = doc_indices[i : i + batch_size]
            batch_input_ids = data["input_ids"][batch_indices]
            batch_attention_mask = data["attention_mask"][batch_indices]
            batch_cls = model.bert_model(
                input_ids=batch_input_ids, attention_mask=batch_attention_mask
            )[0][:, 0]
            cls_list.append(batch_cls)

    cls_feats = torch.cat(cls_list, dim=0)
    data["features"][doc_mask] = cls_feats
    print("update_features done", flush=True)
    return data


def train_epoch(
    model: nn.Module,
    data: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()

    # For training, only optimize on train documents
    train_indices = torch.where(data["train_mask"])[0]
    input_ids = data["input_ids"][train_indices].to(device)
    attention_mask = data["attention_mask"][train_indices].to(device)

    out = model(
        data["features"].to(device),
        data["edge_index"].to(device),
        data["edge_weight"].to(device),
        input_ids,
        attention_mask,
        train_indices,
    )
    loss = criterion(out, data["labels"][train_indices].to(device))

    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(
    model: nn.Module,
    data: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    device: torch.device,
) -> float:
    """Evaluate model accuracy by batching to match original implementation."""
    model.eval()
    eval_indices = torch.where(mask)[0]
    batch_size = 64
    all_pred = []
    all_true = []
    with torch.no_grad():
        for i in range(0, len(eval_indices), batch_size):
            batch_indices = eval_indices[i : i + batch_size]
            input_ids_batch = data["input_ids"][batch_indices]
            attention_mask_batch = data["attention_mask"][batch_indices]
            out = model(
                data["features"],
                data["edge_index"],
                data["edge_weight"],
                input_ids_batch,
                attention_mask_batch,
                batch_indices,
            )
            pred = out.argmax(dim=1)
            all_pred.append(pred)
            all_true.append(data["labels"][batch_indices])
    all_pred = torch.cat(all_pred)
    all_true = torch.cat(all_true)
    acc = (all_pred == all_true).float().mean().item()
    return acc


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """Main training function."""
    # Allow legacy overrides that address keys not present in the base config
    OmegaConf.set_struct(cfg, False)

    # Setup improved logging
    logger = _get_logger()

    project_root = Path(get_original_cwd())

    logger.info("🚀 Starting BertGCN Training")
    logger.info(f"📋 Configuration loaded from: {cfg.__class__.__module__}")

    # Handle dev mode: limit to 1 epoch for quick testing
    if cfg.get("dev", False):
        cfg.training.epochs = 1
        logger.info("⚡ Dev mode enabled: limiting to 1 epoch")

    # Set random seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Using device: {device}")

    freeze_features_after_init = bool(
        getattr(cfg, "freeze_features_after_init", False)
        or getattr(cfg.get("training", {}), "freeze_features_after_init", False)
    )

    # If mix_factor==0, run the plain BERT finetuning pipeline directly to
    # ensure parity with train_bert results (and skip the GCN path entirely).
    if cfg.gcn.mix_factor == 0:
        logger.info(
            "mix_factor=0 detected; delegating to plain BERT finetune for parity."
        )
        return tb.main.__wrapped__(cfg)

    # Setup MLflow to write inside outputs/train_gcn (default if unset)
    default_mlflow_uri = f"file:{project_root / 'outputs' / 'train_gcn' / 'mlruns'}"
    tracking_uri = cfg.get("mlflow_tracking_uri") or default_mlflow_uri
    Path(tracking_uri.replace("file:", "")).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)
    if hasattr(mlflow, "log_system_metrics"):
        mlflow.log_system_metrics(True)
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        # Load real processed dataset
        logger.info("Loading processed dataset...")
        dataset, label_encoder = load_processed_dataset(cfg)
        n_classes = len(label_encoder.classes_)

        # Create graph data from real dataset
        data = load_graph_data_from_disk(cfg)
        logger.info("Graph data loaded, updating features...")

        # Optional overfit/debug mode: shrink train set and reuse for val/test
        overfit_debug = bool(getattr(cfg.training, "overfit_debug", False))
        if overfit_debug:
            overfit_n = int(getattr(cfg.training, "overfit_num_samples", 32))
            train_indices_full = torch.where(data["train_mask"])[0]
            if len(train_indices_full) == 0:
                raise ValueError("Overfit mode requested but no training nodes found.")

            overfit_n = min(overfit_n, len(train_indices_full))
            subset = train_indices_full[:overfit_n]

            new_train = torch.zeros_like(data["train_mask"], dtype=torch.bool)
            new_val = torch.zeros_like(data["val_mask"], dtype=torch.bool)
            new_test = torch.zeros_like(data["test_mask"], dtype=torch.bool)
            new_train[subset] = True
            new_val[subset] = True
            new_test[subset] = True

            data["train_mask"] = new_train
            data["val_mask"] = new_val
            data["test_mask"] = new_test

            logger.info(
                "🎯 Overfit debug enabled: using %d samples and reusing them for val/test.",
                overfit_n,
            )

        # Move data to device
        data = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }

        # Create model
        n_features = data["features"].shape[1]
        model = BertGCN(
            pretrained_model=cfg.hparams.model_name_or_path,
            nb_class=n_classes,
            m=cfg.gcn.mix_factor,
            gcn_layers=cfg.gcn.gcn_layers,
            n_hidden=cfg.gcn.n_hidden,
            dropout=cfg.gcn.dropout,
        )

        model = model.to(device)

        # Setup optimizer, weight decay, and loss. When mix_factor==0 the GCN
        # path is bypassed, so only optimize BERT + classifier to mirror the
        # plain BERT training path.
        bert_lr = cfg.training.bert_lr
        gcn_lr = cfg.training.lr
        weight_decay = getattr(cfg.training, "weight_decay", 1e-4)
        param_groups = [
            {"params": model.bert_model.parameters(), "lr": bert_lr},
            {"params": model.classifier.parameters(), "lr": bert_lr},
        ]
        if cfg.gcn.mix_factor != 0:
            param_groups.append({"params": model.gcn.parameters(), "lr": gcn_lr})

        optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)

        # Scheduler is initialized after data loaders are built
        scheduler = None
        criterion = torch.nn.NLLLoss()

        # Update features once with BERT (uses the current model)
        data = update_features(model, data, device)
        if freeze_features_after_init:
            logger.info(
                "📌 Freezing graph features after initial CLS extraction (no per-epoch refresh)"
            )

        # Create data loaders
        from torch.utils.data import DataLoader, TensorDataset

        train_indices = torch.where(data["train_mask"])[0]
        val_indices = torch.where(data["val_mask"])[0]
        test_indices = torch.where(data["test_mask"])[0]
        batch_size = getattr(cfg.training, "batch_size", 64)
        train_loader = DataLoader(
            TensorDataset(train_indices), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(TensorDataset(val_indices), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(test_indices), batch_size=batch_size)

        # Scheduler selection: default warmup+linear; optional MultiStep to mirror original impl
        scheduler_type = getattr(cfg.training, "scheduler_type", "linear_warmup")
        if scheduler_type == "multistep":
            milestones = getattr(cfg.training, "multistep_milestones", [30])
            gamma = float(getattr(cfg.training, "multistep_gamma", 0.1))
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=gamma
            )
        else:
            total_steps = cfg.training.epochs * max(1, len(train_loader))
            warmup_steps = max(1, int(0.1 * total_steps))

            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(
                    0.1,
                    float(total_steps - current_step)
                    / float(max(1, total_steps - warmup_steps)),
                )

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        def _safe_load_scheduler_state(
            scheduler_obj: torch.optim.lr_scheduler._LRScheduler,
            state_dict: Dict[str, Any],
            logger_obj: logging.Logger,
        ) -> None:
            """Load scheduler state without failing when keys differ across scheduler types."""

            try:
                scheduler_obj.load_state_dict(state_dict)
            except Exception as exc:  # noqa: BLE001
                logger_obj.warning(
                    "Skipping scheduler state restore due to mismatch: %s", exc
                )

        # Setup checkpoint directory (per run to avoid clashes across jobs)
        checkpoint_dir = Path(cfg.hydra.run.dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_acc = 0.0
        start_epoch = 0
        best_checkpoint = checkpoint_dir / "best_checkpoint.pt"

        # Controlled resume behavior (default: fresh start)
        resume_requested = bool(
            getattr(cfg.training, "resume", False)
            or getattr(cfg.training, "resume_from_checkpoint", False)
        )

        if best_checkpoint.exists():
            if resume_requested:
                checkpoint = torch.load(best_checkpoint, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                _safe_load_scheduler_state(
                    scheduler, checkpoint["scheduler_state_dict"], logger
                )
                start_epoch = checkpoint.get("epoch", 0)
                best_val_acc = checkpoint.get("best_val_acc", 0.0)
                logger.info(
                    f"📂 Resumed from checkpoint: {best_checkpoint} at epoch {start_epoch}"
                )
            else:
                best_checkpoint.unlink()
                logger.info(
                    "🧹 Removed existing checkpoint to start a fresh run (resume disabled)."
                )

        # Training loop
        logger.info("🏃 Starting training loop...")
        for epoch in range(start_epoch, cfg.training.epochs):
            model.train()
            epoch_loss = 0.0
            nan_flag = False
            for batch in train_loader:
                indices = batch[0]
                input_ids_batch = data["input_ids"][indices]
                attention_mask_batch = data["attention_mask"][indices]
                out = model(
                    data["features"],
                    data["edge_index"],
                    data["edge_weight"],
                    input_ids_batch,
                    attention_mask_batch,
                    indices,
                )
                loss = criterion(out, data["labels"][indices])

                if torch.isnan(loss):
                    logger.error("NaN loss encountered; stopping training early.")
                    nan_flag = True
                    break

                optimizer.zero_grad()
                loss.backward()
                if getattr(cfg.training, "grad_clip_enabled", True):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            if nan_flag:
                break

            epoch_loss /= len(train_loader)

            # Update features before evaluation unless explicitly frozen
            if not freeze_features_after_init:
                data = update_features(model, data, device)
                torch.cuda.empty_cache()

            # Evaluate on train/val and track validation loss
            train_acc = evaluate(model, data, data["train_mask"], device)
            val_acc = evaluate(model, data, data["val_mask"], device)

            # Compute validation loss without touching the test split
            val_indices = torch.where(data["val_mask"])[0]
            val_loss = 0.0
            batch_size = 64
            model.eval()
            with torch.no_grad():
                for i in range(0, len(val_indices), batch_size):
                    batch_indices = val_indices[i : i + batch_size]
                    input_ids_batch = data["input_ids"][batch_indices]
                    attention_mask_batch = data["attention_mask"][batch_indices]
                    logits = model(
                        data["features"],
                        data["edge_index"],
                        data["edge_weight"],
                        input_ids_batch,
                        attention_mask_batch,
                        batch_indices,
                    )
                    val_loss += criterion(
                        logits, data["labels"][batch_indices]
                    ).item() * len(batch_indices)
            val_loss /= max(1, len(val_indices))

            mlflow.log_metrics(
                {
                    "train_loss": epoch_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                step=epoch,
            )

            logger.info(
                _format_metrics_gcn(
                    epoch, epoch_loss, train_acc, val_acc, cfg.training.epochs
                )
            )

            # Best-checkpoint on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": epoch_loss,
                        "best_val_acc": best_val_acc,
                    },
                    best_checkpoint,
                )
                logger.info(
                    f"💾 New best checkpoint (val_acc={best_val_acc:.4f}) at epoch {epoch+1}: {best_checkpoint}"
                )

        # Load best checkpoint (val accuracy) before final test
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            _safe_load_scheduler_state(
                scheduler, checkpoint["scheduler_state_dict"], logger
            )
            logger.info(
                f"📂 Loaded best checkpoint for final eval (val_acc={checkpoint.get('best_val_acc', 0):.4f})"
            )

        # Recompute features once more with the best model before test
        data = update_features(model, data, device)
        torch.cuda.empty_cache()

        # Create a section break for final evaluation
        import logging as log_module

        record = log_module.LogRecord(
            name=logger.name,
            level=log_module.INFO,
            pathname="",
            lineno=0,
            msg="📊 FINAL EVALUATION RESULTS",
            args=(),
            exc_info=None,
        )
        record.section = True
        logger.handle(record)

        # Final evaluation
        test_acc = evaluate(model, data, data["test_mask"], device)
        mlflow.log_metric("test_acc", test_acc)

        logger.info(f"🎯 Final Test Accuracy: {test_acc:.1%}")
        logger.info("✅ Training completed successfully!")

        # Save final model hierarchically
        final_model_dir = Path(cfg.hydra.run.dir) / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)

        # Save model state dict
        torch.save(model.state_dict(), final_model_dir / "pytorch_model.bin")

        # Save tokenizer (same as BERT tokenizer used)
        model.tokenizer.save_pretrained(final_model_dir)

        # Save model config in transformers format
        config_dict = {
            "model_type": "bertgcn",
            "pretrained_model": cfg.hparams.model_name_or_path,
            "n_classes": n_classes,
            "n_features": cfg.model.n_features,
            "hidden_dim": cfg.gcn.n_hidden,
            "mix_factor": cfg.gcn.mix_factor,
            "gcn_layers": cfg.gcn.gcn_layers,
            "dropout": cfg.model.dropout,
            "architectures": ["BertGCN"],
        }
        with open(final_model_dir / "config.json", "w") as f:
            import json

            json.dump(config_dict, f, indent=2)

        # Save training args for compatibility
        training_args = {
            "output_dir": str(final_model_dir),
            "num_train_epochs": cfg.training.epochs,
            "per_device_train_batch_size": 64,  # From dataloader
            "per_device_eval_batch_size": 64,
            "learning_rate": cfg.training.lr,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
        }
        with open(final_model_dir / "training_args.bin", "wb") as f:
            import pickle

            pickle.dump(training_args, f)

        logger.info(f"💾 Final BertGCN model saved to: {final_model_dir}")
        mlflow.log_artifact(str(final_model_dir), artifact_path="final_model")

        # Log comprehensive training summary
        mlruns_path = str(project_root / "outputs" / "train_gcn" / "mlruns")
        _log_gcn_training_summary(test_acc, final_model_dir, mlruns_path, logger)

        return test_acc


if __name__ == "__main__":
    main()
