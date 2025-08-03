#!/usr/bin/env python3
"""Minimal BertGCN Training for Clinical Text Classification"""

import pickle

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from transformers import AutoTokenizer

from clinic_datasets import CleanClinicDataset
from config import PRETRAINEDMODEL, get_paths, set_random_seeds
from data_manager import load_graph_files
from model import BertGCN


class BertGCNTrainer(pl.LightningModule):
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


def create_graph_data(adj_matrix, features, labels, train_mask, val_mask, test_mask):
    """Create PyTorch Geometric data object."""
    edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)

    return Data(
        x=features,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def main(args=None):
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
    graph_dir = paths.get_graph_path(dataset_name, args.doclevel, args.testunklar)
    adj_matrix, features, labels, train_mask, val_mask, test_mask = load_graph_files(
        graph_dir, dataset_name, args.doclevel, args.testunklar
    )

    # Create PyTorch Geometric graph
    graph_data = create_graph_data(
        adj_matrix, features, labels, train_mask, val_mask, test_mask
    )

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
        bert_ckpt = torch.load(list(bert_model_path.glob("*.ckpt"))[0])
        model.bert_model.load_state_dict(bert_ckpt["state_dict"]["model.bert_model"])
        model.classifier.load_state_dict(bert_ckpt["state_dict"]["model.classifier"])

    # Create trainer
    lightning_model = BertGCNTrainer(model, num_classes)

    save_dir = paths.get_model_path("gcn", args.doclevel, f"mixfactor_{args.mixfactor}")

    trainer = pl.Trainer(
        max_epochs=args.nepochs,
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_dir, monitor="val_f1", mode="max"),
            pl.callbacks.EarlyStopping(monitor="val_f1", patience=5, mode="max"),
        ],
        enable_progress_bar=False,
    )

    # Create data loaders (simplified - single graph approach)
    train_data = [
        (
            graph_data,
            graph_data.train_mask.nonzero().squeeze(),
            graph_data.y[graph_data.train_mask],
        )
    ]
    val_data = [
        (
            graph_data,
            graph_data.val_mask.nonzero().squeeze(),
            graph_data.y[graph_data.val_mask],
        )
    ]
    test_data = [
        (
            graph_data,
            graph_data.test_mask.nonzero().squeeze(),
            graph_data.y[graph_data.test_mask],
        )
    ]

    # Train and test
    if not args.testonly:
        trainer.fit(lightning_model, train_data, val_data)
    trainer.test(lightning_model, test_data)


if __name__ == "__main__":
    main()
