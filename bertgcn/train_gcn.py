"""Minimal BERT-GCN Training with Hydra and MLflow.

Educational example demonstrating:
- Hydra configuration management
- MLflow experiment tracking
- PyTorch GCN training
- Real dataset integration

This version uses the same processed dataset as finetune_bert.py
"""

import logging
import random
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

from bertgcn.utils import load_corpus


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

    try:
        project_root = Path(get_original_cwd())
    except Exception:
        project_root = Path.cwd()
    dataset_path = project_root / dataset_str
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

    # Set random features for word nodes (they are zero in the graph)
    n_docs = train_mask.sum() + val_mask.sum() + test_mask.sum()
    vocab_size = adj.shape[0] - n_docs
    print("setting random features", flush=True)
    features_array[n_docs:] = np.random.randn(
        vocab_size, features_array.shape[1]
    ).astype(np.float32)
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

    # Load tokenized dataset for input_ids and attention_mask
    processed_dir = project_root / "data" / "processed"
    dataset_path_hf = processed_dir / "tokenized_dataset"
    if dataset_path_hf.exists():
        ds = load_from_disk(str(dataset_path_hf))
        input_ids = torch.stack([torch.tensor(ids) for ids in ds["input_ids"]])
        attention_mask = torch.stack(
            [torch.tensor(mask) for mask in ds["attention_mask"]]
        )
        # Pad to match graph size: docs + words
        nb_node = adj.shape[0]
        if input_ids.shape[0] < nb_node:
            padding = torch.zeros(
                (nb_node - input_ids.shape[0], input_ids.shape[1]), dtype=torch.long
            )
            input_ids = torch.cat([input_ids, padding], dim=0)
            attention_mask = torch.cat([attention_mask, padding], dim=0)
    else:
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
    model_name: str, data: Dict[str, torch.Tensor], device: torch.device
):
    """Update document features using BERT."""
    print("Starting update_features", flush=True)
    from transformers import AutoModel

    bert_model = AutoModel.from_pretrained(model_name).to(device)
    bert_model.eval()
    doc_mask = data["train_mask"] | data["val_mask"] | data["test_mask"]
    doc_indices = torch.where(doc_mask)[0]
    batch_size = 64  # Larger batch for speed
    cls_list = []
    for i in range(0, len(doc_indices), batch_size):
        batch_indices = doc_indices[i : i + batch_size]
        batch_input_ids = data["input_ids"][batch_indices]
        batch_attention_mask = data["attention_mask"][batch_indices]
        with torch.no_grad():
            batch_cls = bert_model(
                input_ids=batch_input_ids, attention_mask=batch_attention_mask
            )[0][:, 0]
        cls_list.append(batch_cls)
    cls_feats = torch.cat(cls_list, dim=0)
    data["features"][doc_mask] = cls_feats
    print("update_features done", flush=True)
    return data
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

    # For training, pass inputs for BERT prediction on documents
    doc_mask = data["train_mask"] | data["val_mask"] | data["test_mask"]
    input_ids = data["input_ids"][doc_mask].to(device) if "input_ids" in data else None
    attention_mask = (
        data["attention_mask"][doc_mask].to(device)
        if "attention_mask" in data
        else None
    )

    # Call model with appropriate arguments
    out = model(
        data["features"].to(device),
        data["adj"].to(device),
        input_ids,
        attention_mask,
    )
    loss = criterion(
        out[data["train_mask"]], data["labels"][data["train_mask"]].to(device)
    )

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


@hydra.main(version_base=None, config_path="../conf", config_name="mode/train_gcn")
def main(cfg: DictConfig) -> float:
    """Main training function."""
    # Setup logging
    from .logging_config import setup_logging

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting Minimal GCN Training")
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")

    # Handle dev mode: limit to 1 epoch for quick testing
    if cfg.get("dev", False):
        cfg.training.epochs = 1
        logger.info("Dev mode enabled: limiting to 1 epoch")

    # Set random seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup MLflow
    mlflow.set_experiment(cfg.mlflow_experiment_name)
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

        # Setup optimizer and loss
        bert_lr = cfg.training.bert_lr
        gcn_lr = cfg.training.lr
        optimizer = torch.optim.Adam(
            [
                {"params": model.bert_model.parameters(), "lr": bert_lr},
                {"params": model.classifier.parameters(), "lr": bert_lr},
                {"params": model.gcn.parameters(), "lr": gcn_lr},
            ]
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30], gamma=0.1
        )
        criterion = torch.nn.NLLLoss()

        # Update features once with BERT
        data = update_features(cfg.hparams.model_name_or_path, data, device)

        # Create data loaders
        from torch.utils.data import DataLoader, TensorDataset

        train_indices = torch.where(data["train_mask"])[0]
        val_indices = torch.where(data["val_mask"])[0]
        test_indices = torch.where(data["test_mask"])[0]
        train_loader = DataLoader(
            TensorDataset(train_indices), batch_size=64, shuffle=True
        )
        val_loader = DataLoader(TensorDataset(val_indices), batch_size=64)
        test_loader = DataLoader(TensorDataset(test_indices), batch_size=64)

        # Training loop
        for epoch in range(cfg.training.epochs):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                indices = batch[0]
                # Forward
                out = model(
                    data["features"],
                    data["edge_index"],
                    data["edge_weight"],
                )
                loss = criterion(out[indices], data["labels"][indices])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_loader)

            # Evaluate
            train_acc = evaluate(model, data, data["train_mask"], device)
            val_acc = evaluate(model, data, data["val_mask"], device)

            # Log metrics
            mlflow.log_metrics(
                {"train_loss": epoch_loss, "train_acc": train_acc, "val_acc": val_acc},
                step=epoch,
            )

            logger.info(
                f"Epoch {epoch+1}/{cfg.training.epochs} - "
                f"Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
            )

            scheduler.step()

            # Update features for next epoch
            data = update_features(cfg.hparams.model_name_or_path, data, device)

            # Save best model (removed, save only final)

        # Final evaluation
        test_acc = evaluate(model, data, data["test_mask"], device)
        mlflow.log_metric("test_acc", test_acc)

        logger.info(f"Final Test Accuracy: {test_acc:.4f}")
        logger.info("Training completed!")

        # Save final model hierarchically
        import os
        from pathlib import Path

        final_model_dir = Path("models/final_model")
        final_model_dir.mkdir(parents=True, exist_ok=True)

        # Save model state dict
        torch.save(model.state_dict(), final_model_dir / "pytorch_model.bin")

        # Save tokenizer (same as BERT tokenizer used)
        model.tokenizer.save_pretrained(final_model_dir)

        # Save model config in transformers format
        config_dict = {
            "model_type": "bertgcn",
            "pretrained_model": cfg.hparams.model_name_or_path,
            "n_classes": cfg.model.n_classes,
            "n_features": cfg.model.n_features,
            "hidden_dim": cfg.model.hidden_dim,
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

        logger.info(f"Saved final model to {final_model_dir}")
        mlflow.log_artifact(str(final_model_dir), artifact_path="final_model")

        return test_acc


if __name__ == "__main__":
    main()
