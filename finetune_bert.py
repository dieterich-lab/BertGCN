#!/usr/bin/env python3
"""Minimal BERT Fine-tuning for Clinical Text Classification"""

import pickle

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from clinic_datasets import CleanClinicDataset
from config import get_paths
from entry import PRETRAINEDMODEL
from params import parse_args


class BertClassifier(pl.LightningModule):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )
        self.preds, self.labels = [], []

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        return self(**batch).loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.preds.append(torch.argmax(outputs.logits, dim=1))
        self.labels.append(batch["labels"])
        return outputs.loss

    def on_validation_epoch_end(self):
        if self.preds:
            preds = torch.cat(self.preds).cpu()
            labels = torch.cat(self.labels).cpu()
            self.log(
                "val_f1", f1_score(labels, preds, average="macro", zero_division=0)
            )
            self.preds.clear()
            self.labels.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)


def main():
    args = parse_args()
    pl.seed_everything(42)
    paths = get_paths()

    # Load dataset
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINEDMODEL)
    dataset_file = paths.get_dataset_path(
        args.doclevel,
        "medbert",
        suffix="_nomeds" if args.noarznei else "",
        clean=args.clean,
    )

    if dataset_file.exists():
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = CleanClinicDataset(
            tokenizer, args.doclevel, clean=args.clean, nomeds=args.noarznei
        )
        with open(dataset_file, "wb") as f:
            pickle.dump(dataset, f)

    # Create splits and loaders
    indices = np.random.permutation(len(dataset))
    splits = [int(len(indices) * r) for r in [0.7, 0.8]]
    loaders = [
        DataLoader(
            Subset(dataset, indices[s:e]), batch_size=args.batchsize, shuffle=i == 0
        )
        for i, (s, e) in enumerate(
            [(0, splits[0]), (splits[0], splits[1]), (splits[1], len(indices))]
        )
    ]

    # Train
    model = BertClassifier(PRETRAINEDMODEL, len(dataset.LE.classes_))
    trainer = pl.Trainer(
        max_epochs=args.nepochs,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                paths.get_model_path("bert", args.doclevel),
                monitor="val_f1",
                mode="max",
            )
        ],
        enable_progress_bar=False,
    )

    if not args.testonly:
        trainer.fit(model, loaders[0], loaders[1])
    trainer.test(model, loaders[2])


if __name__ == "__main__":
    main()
