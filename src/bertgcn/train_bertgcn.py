#!/usr/bin/env python3
"""Minimal BertGCN Training for Clinical Text Classification"""

import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer

from bertgcn.config import PRETRAINEDMODEL, get_paths, set_random_seeds
from bertgcn.data import CleanClinicDataset
from bertgcn.data_manager import load_graph_files
from bertgcn.models import BertGCN


class BertGCNTrainer(pl.LightningModule):
    """PyTorch Lightning module for BertGCN training."""

    def __init__(self, model, num_classes, lr_bert=1e-5, lr_gcn=1e-4):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr_bert = lr_bert
        self.lr_gcn = lr_gcn
        self.val_preds, self.val_labels = [], []

    def forward(self, graph, idx):
        return self.model(graph, idx)

    def training_step(self, batch, batch_idx):
        graph, train_idx, y_train = batch
        y_pred = self.model(graph, train_idx)
        loss = F.cross_entropy(y_pred, y_train)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        graph, val_idx, y_val = batch
        y_pred = self.model(graph, val_idx)
        loss = F.cross_entropy(y_pred, y_val)
        preds = torch.argmax(y_pred, dim=1)
        self.val_preds.append(preds)
        self.val_labels.append(y_val)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if self.val_preds:
            all_preds = torch.cat(self.val_preds).cpu().numpy()
            all_labels = torch.cat(self.val_labels).cpu().numpy()
            f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            self.log("val_f1", f1, prog_bar=True)
            self.val_preds.clear()
            self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        graph, test_idx, y_test = batch
        y_pred = self.model(graph, test_idx)
        preds = torch.argmax(y_pred, dim=1)
        return {"preds": preds, "labels": y_test}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).cpu().numpy()
        print("\nTest Results:")
        print(classification_report(labels, preds, zero_division=0))

    def configure_optimizers(self):
        return torch.optim.Adam(
            [
                {"params": self.model.bert_model.parameters(), "lr": self.lr_bert},
                {"params": self.model.classifier.parameters(), "lr": self.lr_bert},
                {"params": self.model.gcn.parameters(), "lr": self.lr_gcn},
            ]
        )


def create_simple_graph_data(
    adj_matrix: csr_matrix,
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> Dict:
    """Create a simple graph data structure for DGL-style processing."""
    # Convert to PyTorch tensors
    features_tensor = torch.FloatTensor(features)
    labels_tensor = torch.LongTensor(labels)

    # Create edge information from adjacency matrix
    edges = adj_matrix.nonzero()
    edge_index = torch.LongTensor(np.vstack([edges[0], edges[1]]))
    edge_weights = torch.FloatTensor(adj_matrix.data)

    return {
        "x": features_tensor,
        "edge_index": edge_index,
        "edge_attr": edge_weights,
        "y": labels_tensor,
        "train_mask": torch.BoolTensor(train_mask),
        "val_mask": torch.BoolTensor(val_mask),
        "test_mask": torch.BoolTensor(test_mask),
    }


def create_dgl_graph(graph_data: Dict):
    """Create a DGL graph from the graph data (simplified version)."""
    import dgl

    # Create graph from edge indices
    num_nodes = graph_data["x"].shape[0]
    src, dst = graph_data["edge_index"]
    g = dgl.graph((src, dst), num_nodes=num_nodes)

    # Add node and edge features
    g.ndata["cls_feats"] = graph_data["x"]
    g.edata["edge_weight"] = graph_data["edge_attr"]
    g.ndata["labels"] = graph_data["y"]

    return g


def main(args=None):
    """Main training function for BertGCN."""
    if args is None:
        # Fallback for direct execution
        class Args:
            doclevel = "letter"
            nepochs = 50
            mixfactor = 0.7
            clean = True
            testunklar = False
            testonly = False

        args = Args()

    set_random_seeds(42)
    pl.seed_everything(42)
    paths = get_paths()

    # Load dataset
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINEDMODEL)
    dataset_file = paths.get_dataset_path(args.doclevel, "medbert", clean=args.clean)

    if dataset_file.exists():
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = CleanClinicDataset(tokenizer, args.doclevel, clean=args.clean)
        with open(dataset_file, "wb") as f:
            pickle.dump(dataset, f)

    # Load graph data
    dataset_name = f"medindcls_{args.doclevel}"
    try:
        adj_matrix, data_matrices, metadata = load_graph_files(
            dataset_name, args.doclevel, args.testunklar
        )

        # Create graph data structure
        graph_data = create_simple_graph_data(
            adj_matrix,
            data_matrices["features"],
            data_matrices["labels"],
            data_matrices["train_mask"],
            data_matrices["val_mask"],
            data_matrices["test_mask"],
        )

        # Create DGL graph
        graph = create_dgl_graph(graph_data)

        # Initialize model
        num_classes = len(dataset.LE.classes_)
        model = BertGCN(
            nb_class=num_classes,
            pretrained_model=PRETRAINEDMODEL,
            mix_factor=args.mixfactor,
            gcn_layers=2,
            n_hidden=200,
            dropout=0.5,
        )

        # Load fine-tuned BERT weights if available
        bert_model_path = paths.get_model_path("bert", args.doclevel)
        if bert_model_path.exists():
            try:
                bert_ckpt_files = list(bert_model_path.glob("*.ckpt"))
                if bert_ckpt_files:
                    bert_ckpt = torch.load(bert_ckpt_files[0], map_location="cpu")
                    if "state_dict" in bert_ckpt:
                        # Extract BERT and classifier weights
                        bert_state = {}
                        classifier_state = {}
                        for key, value in bert_ckpt["state_dict"].items():
                            if "model.bert_model" in key:
                                new_key = key.replace("model.bert_model.", "")
                                bert_state[new_key] = value
                            elif "model.classifier" in key:
                                new_key = key.replace("model.classifier.", "")
                                classifier_state[new_key] = value

                        if bert_state:
                            model.bert_model.load_state_dict(bert_state)
                            print("✅ Loaded fine-tuned BERT weights")
                        if classifier_state:
                            model.classifier.load_state_dict(classifier_state)
                            print("✅ Loaded fine-tuned classifier weights")
            except Exception as e:
                print(f"⚠️  Could not load BERT weights: {e}")

        # Create trainer
        lightning_model = BertGCNTrainer(model, num_classes)

        save_dir = paths.get_model_path(
            "gcn", args.doclevel, f"mixfactor_{args.mixfactor}"
        )

        # Generate meaningful log directory name
        experiment_name = f"gcn_epochs_{args.nepochs}_mix_{args.mixfactor}"
        if args.clean:
            experiment_name += "_clean"
        if args.testunklar:
            experiment_name += "_testunklar"

        log_dir = paths.get_log_path("bertgcn", args.doclevel, experiment_name)

        trainer = pl.Trainer(
            max_epochs=args.nepochs,
            callbacks=[
                pl.callbacks.ModelCheckpoint(save_dir, monitor="val_f1", mode="max"),
                pl.callbacks.EarlyStopping(monitor="val_f1", patience=5, mode="max"),
                pl.callbacks.RichProgressBar(),
            ],
            enable_progress_bar=True,
            log_every_n_steps=10,
            default_root_dir=log_dir,
        )

        print(f"🔥 Starting BertGCN training...")
        print(f"   • Training logs will be saved to: {log_dir}")
        print(f"   • Experiment name: {experiment_name}")
        print(f"   • Model checkpoints will be saved to: {save_dir}")

        # Create data loaders (simplified - single graph approach)
        train_indices = graph_data["train_mask"].nonzero().squeeze()
        val_indices = graph_data["val_mask"].nonzero().squeeze()
        test_indices = graph_data["test_mask"].nonzero().squeeze()

        train_data = [(graph, train_indices, graph_data["y"][train_indices])]
        val_data = [(graph, val_indices, graph_data["y"][val_indices])]
        test_data = [(graph, test_indices, graph_data["y"][test_indices])]

        # Train and test
        if not args.testonly:
            trainer.fit(lightning_model, train_data, val_data)
        trainer.test(lightning_model, test_data)

        print(f"✅ BertGCN training completed for {args.doclevel}")

    except Exception as e:
        print(f"❌ Error during BertGCN training: {e}")
        print("Make sure you have built the graph first using: bertgcn build-graph-cmd")
        raise


if __name__ == "__main__":
    main()
