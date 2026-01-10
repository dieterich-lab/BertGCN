import argparse
import json
import logging
import os
import random
import sys
import types
from pathlib import Path
from typing import Tuple

# DGL 2.x tries to load GraphBolt; torch 2.8 wheels lack the compiled lib. Stub
# the module to bypass the native load without touching the installed package.
if "dgl.graphbolt" not in sys.modules:
    sys.modules["dgl.graphbolt"] = types.ModuleType("dgl.graphbolt")

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from bertgcn.utils import load_corpus, normalize_adj


def get_logger() -> logging.Logger:
    logger = logging.getLogger("train_gcn_dgl")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BertGCNDGL(nn.Module):
    """DGL-based BertGCN mirroring the original reference implementation."""

    def __init__(
        self,
        pretrained_model: str,
        nb_class: int,
        m: float = 1.0,
        gcn_layers: int = 2,
        n_hidden: int = 200,
        dropout: float = 0.5,
        train_with_bert: bool = True,
    ):
        super().__init__()
        self.m = m
        self.nb_class = nb_class
        self.train_with_bert = train_with_bert
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = self.bert_model.config.hidden_size
        self.classifier = nn.Linear(self.feat_dim, nb_class)
        self.gcn = dgl.nn.pytorch.GraphConv(
            in_feats=self.feat_dim,
            out_feats=n_hidden,
            activation=F.elu,
            norm="none",
        )
        # Additional layers beyond first
        self.gcn_hidden = nn.ModuleList()
        for _ in range(max(0, gcn_layers - 2)):
            self.gcn_hidden.append(
                dgl.nn.pytorch.GraphConv(
                    in_feats=n_hidden,
                    out_feats=n_hidden,
                    activation=F.elu,
                    norm="none",
                )
            )
        self.gcn_out = dgl.nn.pytorch.GraphConv(
            in_feats=n_hidden,
            out_feats=nb_class,
            norm="none",
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, g: dgl.DGLGraph, idx: torch.Tensor) -> torch.Tensor:
        input_ids = g.ndata["input_ids"][idx]
        attention_mask = g.ndata["attention_mask"][idx]

        if self.train_with_bert and self.training:
            cls_feats = self.bert_model(
                input_ids=input_ids, attention_mask=attention_mask
            )[0][:, 0]
            g.ndata["cls_feats"][idx] = cls_feats
        else:
            cls_feats = g.ndata["cls_feats"][idx]

        cls_logit = self.classifier(cls_feats)
        cls_pred = torch.softmax(cls_logit, dim=1)

        h = g.ndata["cls_feats"]
        h = self.gcn(g, h, edge_weight=g.edata["edge_weight"])
        for layer in self.gcn_hidden:
            h = self.dropout(h)
            h = layer(g, h, edge_weight=g.edata["edge_weight"])
        h = self.dropout(h)
        gcn_logit = self.gcn_out(g, h, edge_weight=g.edata["edge_weight"])
        gcn_pred = torch.softmax(gcn_logit[idx], dim=1)

        pred = (gcn_pred + 1e-10) * self.m + cls_pred * (1 - self.m)
        return torch.log(pred)


def load_tokens(
    dataset_prefix: Path, processed_dir: Path, splits_path: Path, logger: logging.Logger
) -> Tuple[torch.Tensor, torch.Tensor]:
    hf_path = processed_dir / "tokenized_dataset"
    if not hf_path.exists():
        raise FileNotFoundError(
            f"Missing tokenized dataset at {hf_path}; run preprocess first."
        )

    import datasets

    ds = datasets.load_from_disk(str(hf_path))
    tokens_input_ids = torch.stack([torch.tensor(ids) for ids in ds["input_ids"]])
    tokens_attention = torch.stack(
        [torch.tensor(mask) for mask in ds["attention_mask"]]
    )

    if not splits_path.exists():
        raise FileNotFoundError(
            f"Missing split metadata {splits_path}; rebuild graph to generate it."
        )
    splits = json.loads(splits_path.read_text())
    train_idx = torch.tensor(splits.get("train_idx", []), dtype=torch.long)
    val_idx = torch.tensor(splits.get("val_idx", []), dtype=torch.long)
    test_idx = torch.tensor(splits.get("test_idx", []), dtype=torch.long)

    logger.info(
        "Token order: train=%d, val=%d, test=%d",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    train_tokens = tokens_input_ids[train_idx]
    val_tokens = tokens_input_ids[val_idx]
    test_tokens = tokens_input_ids[test_idx]

    train_attention = tokens_attention[train_idx]
    val_attention = tokens_attention[val_idx]
    test_attention = tokens_attention[test_idx]

    return (
        torch.cat([train_tokens, val_tokens, test_tokens], dim=0),
        torch.cat([train_attention, val_attention, test_attention], dim=0),
    )


def build_graph(
    dataset_prefix: Path, tokens_input_ids: torch.Tensor, tokens_attention: torch.Tensor
):
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
    ) = load_corpus(str(dataset_prefix))

    val_size = int(val_mask.sum())
    total_docs = train_size + val_size + test_size
    vocab_size = features.shape[0] - total_docs
    nb_node = features.shape[0]

    # Normalize adjacency and build DGL graph
    adj_norm = normalize_adj(
        adj
        + adj.T.multiply(adj.T > adj)
        - adj.multiply(adj.T > adj)
        + np.eye(adj.shape[0])
    )
    g = dgl.from_scipy(adj_norm.astype("float32"), eweight_name="edge_weight")

    labels = y_train + y_val + y_test
    labels = np.argmax(labels, axis=1)
    label_train = np.argmax(y_train, axis=1)

    doc_positions_train = torch.arange(train_size)
    doc_positions_val = torch.arange(
        train_size + vocab_size, train_size + vocab_size + val_size
    )
    doc_positions_test = torch.arange(
        train_size + vocab_size + val_size,
        train_size + vocab_size + val_size + test_size,
    )

    input_ids = torch.zeros((nb_node, tokens_input_ids.shape[1]), dtype=torch.long)
    attention_mask = torch.zeros((nb_node, tokens_attention.shape[1]), dtype=torch.long)

    input_ids[doc_positions_train] = tokens_input_ids[:train_size]
    attention_mask[doc_positions_train] = tokens_attention[:train_size]

    input_ids[doc_positions_val] = tokens_input_ids[train_size : train_size + val_size]
    attention_mask[doc_positions_val] = tokens_attention[
        train_size : train_size + val_size
    ]

    input_ids[doc_positions_test] = tokens_input_ids[train_size + val_size :]
    attention_mask[doc_positions_test] = tokens_attention[train_size + val_size :]

    g.ndata["input_ids"] = input_ids
    g.ndata["attention_mask"] = attention_mask
    g.ndata["label"] = torch.tensor(labels, dtype=torch.long)
    g.ndata["train"] = torch.tensor(train_mask, dtype=torch.float32)
    g.ndata["val"] = torch.tensor(val_mask, dtype=torch.float32)
    g.ndata["test"] = torch.tensor(test_mask, dtype=torch.float32)
    g.ndata["label_train"] = torch.tensor(label_train, dtype=torch.long)
    g.ndata["cls_feats"] = torch.zeros((nb_node, features.shape[1]))

    return g, train_size, val_size, test_size, vocab_size


def update_feature(
    model: BertGCNDGL,
    g: dgl.DGLGraph,
    doc_mask: torch.Tensor,
    device: torch.device,
    bert_device: torch.device | None = None,
    batch_size: int = 64,
):
    dataset = TensorDataset(
        g.ndata["input_ids"][doc_mask], g.ndata["attention_mask"][doc_mask]
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    cls_list = []
    model.eval()
    orig_device = next(model.bert_model.parameters()).device
    bert_device = bert_device or device
    model.bert_model.to(bert_device)
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(bert_device)
            attention_mask = attention_mask.to(bert_device)
            output = model.bert_model(
                input_ids=input_ids, attention_mask=attention_mask
            )[0][:, 0]
            cls_list.append(output.cpu())
    model.bert_model.to(orig_device)
    cls_feat = torch.cat(cls_list, dim=0)
    g = g.to("cpu")
    g.ndata["cls_feats"][doc_mask] = cls_feat
    return g


def evaluate(
    model: BertGCNDGL,
    g: dgl.DGLGraph,
    loader: DataLoader,
    split_mask: torch.Tensor,
    device: torch.device,
):
    model.eval()
    total_correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.NLLLoss()
    with torch.no_grad():
        for (idx,) in loader:
            idx = idx.to(device)
            mask = split_mask[idx].bool()
            if mask.sum() == 0:
                continue
            out = model(g, idx)[mask]
            target = g.ndata["label"][idx][mask]
            loss = criterion(out, target)
            pred = out.argmax(dim=1)
            total_correct += (pred == target).sum().item()
            total += target.numel()
            total_loss += loss.item() * target.numel()
    acc = total_correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return acc, avg_loss


def main():
    parser = argparse.ArgumentParser(
        description="DGL BertGCN trainer (GCN-only mirror)"
    )
    parser.add_argument(
        "--dataset_prefix",
        default="data/ind.medindcls_letter_letter",
        help="Graph prefix",
    )
    parser.add_argument(
        "--model_name",
        default="/prj/doctoral_letters/PETGUI/med_bert_local",
        help="BERT checkpoint",
    )
    parser.add_argument(
        "--m", type=float, default=1.0, help="Mix factor (1.0 = GCN only)"
    )
    parser.add_argument("--gcn_layers", type=int, default=2)
    parser.add_argument("--n_hidden", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gcn_lr", type=float, default=1e-3)
    parser.add_argument("--bert_lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint_dir",
        default="outputs/train_gcn_dgl",
        help="Where to save checkpoints",
    )
    parser.add_argument(
        "--freeze_bert",
        action="store_true",
        help="Freeze BERT and reuse cached CLS features",
    )
    parser.add_argument(
        "--feat_batch_size",
        type=int,
        default=64,
        help="Batch size for CLS feature extraction",
    )
    args = parser.parse_args()

    # Default to freezing BERT when GCN-only (m >= 1.0) to avoid redundant CPU-bound computation.
    if not args.freeze_bert:
        args.freeze_bert = args.m >= 1.0

    logger = get_logger()
    seed_everything(args.seed)

    use_cuda = (
        torch.cuda.is_available()
        and hasattr(dgl, "cuda")
        and hasattr(dgl.cuda, "is_available")
        and dgl.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    bert_device = torch.device("cuda" if torch.cuda.is_available() else device)
    logger.info("Using device: %s", device)
    logger.info("BERT feature device: %s", bert_device)

    dataset_prefix = Path(args.dataset_prefix)
    processed_dir = Path("data/processed")
    splits_path = dataset_prefix.parent / f"{dataset_prefix.name}.splits.json"

    tokens_input_ids, tokens_attention = load_tokens(
        dataset_prefix, processed_dir, splits_path, logger
    )
    g, train_size, val_size, test_size, vocab_size = build_graph(
        dataset_prefix, tokens_input_ids, tokens_attention
    )

    nb_node = g.num_nodes()
    doc_mask = (g.ndata["train"] + g.ndata["val"] + g.ndata["test"]).bool()

    model = BertGCNDGL(
        pretrained_model=args.model_name,
        nb_class=int(g.ndata["label"].max().item() + 1),
        m=args.m,
        gcn_layers=args.gcn_layers,
        n_hidden=args.n_hidden,
        dropout=args.dropout,
        train_with_bert=not args.freeze_bert,
    ).to(device)

    if args.freeze_bert:
        for p in model.bert_model.parameters():
            p.requires_grad = False
        logger.info("Freezing BERT parameters and reusing cached CLS features")

    param_groups = [
        {
            "params": list(model.gcn.parameters())
            + list(model.gcn_hidden.parameters())
            + list(model.gcn_out.parameters()),
            "lr": args.gcn_lr,
        }
    ]
    if not args.freeze_bert:
        param_groups.append(
            {"params": model.bert_model.parameters(), "lr": args.bert_lr}
        )
        param_groups.append(
            {"params": model.classifier.parameters(), "lr": args.bert_lr}
        )
    else:
        param_groups.append(
            {"params": model.classifier.parameters(), "lr": args.gcn_lr}
        )
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30], gamma=0.1
    )
    criterion = nn.NLLLoss()

    val_start = train_size + vocab_size
    test_start = val_start + val_size
    idx_train = TensorDataset(torch.arange(0, train_size, dtype=torch.long))
    idx_val = TensorDataset(
        torch.arange(val_start, val_start + val_size, dtype=torch.long)
    )
    idx_test = TensorDataset(
        torch.arange(test_start, test_start + test_size, dtype=torch.long)
    )

    loader_train = DataLoader(idx_train, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(idx_val, batch_size=args.batch_size)
    loader_test = DataLoader(idx_test, batch_size=args.batch_size)

    best_val = 0.0
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "checkpoint.pth"

    # Initial feature extraction
    g = update_feature(
        model,
        g,
        doc_mask,
        device,
        bert_device=bert_device,
        batch_size=args.feat_batch_size,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        g = g.to(device)
        total_loss = 0.0
        total_samples = 0
        for (idx,) in loader_train:
            idx = idx.to(device)
            optimizer.zero_grad()
            out = model(g, idx)
            target = g.ndata["label_train"][idx]
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * target.numel()
            total_samples += target.numel()
        scheduler.step()
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        # Refresh CLS features
        g = g.to("cpu")
        if not args.freeze_bert:
            g = update_feature(
                model,
                g,
                doc_mask,
                device,
                bert_device=bert_device,
                batch_size=args.feat_batch_size,
            )

        # Evaluation
        g = g.to(device)
        train_acc, train_nll = evaluate(
            model, g, loader_train, g.ndata["train"], device
        )
        val_acc, val_nll = evaluate(model, g, loader_val, g.ndata["val"], device)
        test_acc, test_nll = evaluate(model, g, loader_test, g.ndata["test"], device)

        logger.info(
            "Epoch %d | Loss %.4f | Train acc %.4f nll %.4f | Val acc %.4f nll %.4f | Test acc %.4f nll %.4f",
            epoch,
            avg_loss,
            train_acc,
            train_nll,
            val_acc,
            val_nll,
            test_acc,
            test_nll,
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
            logger.info(
                "New best checkpoint saved (val_acc=%.4f) to %s", val_acc, ckpt_path
            )

    logger.info("Training completed. Best val acc=%.4f", best_val)


if __name__ == "__main__":
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    main()
