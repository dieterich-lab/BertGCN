#!/usr/bin/env python3
"""Minimal BERT Fine-tuning for Clinical Text Classification"""

import pickle

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bertgcn.config import PRETRAINEDMODEL, get_paths, set_random_seeds
from bertgcn.data import CleanClinicDataset


class BertClassifier(pl.LightningModule):
    def __init__(
        self, model_name, num_classes, lr=5e-5, weight_decay=0.01, warmup_steps=0
    ):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.preds, self.labels = [], []

    def forward(self, **batch):
        # Filter out non-BERT arguments (like 'meds')
        bert_inputs = {
            k: v
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask", "labels"]
        }
        return self.model(**bert_inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.preds.append(torch.argmax(outputs.logits, dim=1))
        self.labels.append(batch["labels"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if self.preds:
            preds = torch.cat(self.preds).cpu()
            labels = torch.cat(self.labels).cpu()
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            self.log("val_f1", f1, prog_bar=True)
            self.log("val_accuracy", (preds == labels).float().mean(), prog_bar=True)
            self.preds.clear()
            self.labels.clear()

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)
        labels = batch["labels"]

        # Log test metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        return {"test_preds": preds, "test_labels": labels}

    def test_epoch_end(self, outputs):
        # Aggregate test predictions
        all_preds = torch.cat([x["test_preds"] for x in outputs]).cpu()
        all_labels = torch.cat([x["test_labels"] for x in outputs]).cpu()

        # Calculate metrics
        test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        test_acc = (all_preds == all_labels).float().mean()

        # Log metrics
        self.log("test_f1", test_f1)
        self.log("test_accuracy", test_acc)

        return {"test_f1": test_f1, "test_accuracy": test_acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        if self.warmup_steps > 0:
            from transformers import get_linear_schedule_with_warmup

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        return optimizer


def main(args=None):
    if args is None:
        # Fallback for direct execution
        class Args:
            doclevel = "letter"
            nepochs = 50
            batchsize = 1
            lr = 5e-5
            weight_decay = 0.01
            warmup_steps = 0
            max_length = 512
            clean = True
            noarznei = False
            testonly = False

        args = Args()

    set_random_seeds(42)
    pl.seed_everything(42)
    paths = get_paths()

    print(f"🚀 Starting BERT fine-tuning for {args.doclevel} documents")
    print(f"📊 Configuration:")
    print(f"   • Document level: {args.doclevel}")
    print(f"   • Epochs: {args.nepochs}")
    print(f"   • Batch size: {args.batchsize}")
    print(f"   • Learning rate: {args.lr}")
    print(f"   • Weight decay: {args.weight_decay}")
    print(f"   • Warmup steps: {args.warmup_steps}")
    print(f"   • Max sequence length: {args.max_length}")
    print(f"   • Text cleaning: {'Yes' if args.clean else 'No'}")
    print(f"   • Exclude medications: {'Yes' if args.noarznei else 'No'}")
    print(f"   • Test only: {'Yes' if args.testonly else 'No'}")
    print(f"   • Model: {PRETRAINEDMODEL}")
    print()

    # Load dataset
    print("📚 Loading and preparing dataset...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINEDMODEL)
    dataset_file = paths.get_dataset_path(
        args.doclevel,
        "medbert",
        suffix="_nomeds" if args.noarznei else "",
        clean=args.clean,
    )

    if dataset_file.exists():
        print(f"   • Loading cached dataset from {dataset_file}")
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        print(f"   • Creating new dataset and caching to {dataset_file}")
        dataset = CleanClinicDataset(
            tokenizer,
            args.doclevel,
            clean=args.clean,
            nomeds=args.noarznei,
            max_length=args.max_length,
        )
        with open(dataset_file, "wb") as f:
            pickle.dump(dataset, f)

    print(f"   • Dataset size: {len(dataset)} examples")
    print(
        f"   • Classes: {len(dataset.LE.classes_)} ({', '.join(dataset.LE.classes_[:5])}{'...' if len(dataset.LE.classes_) > 5 else ''})"
    )
    print()

    # Create splits and loaders
    print("🔄 Creating data splits...")
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

    print(f"   • Training samples: {len(loaders[0].dataset)}")
    print(f"   • Validation samples: {len(loaders[1].dataset)}")
    print(f"   • Test samples: {len(loaders[2].dataset)}")
    print()

    # Train
    print("🧠 Initializing BERT model...")
    model = BertClassifier(
        PRETRAINEDMODEL,
        len(dataset.LE.classes_),
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
    )
    print(f"   • Model: {model.model.__class__.__name__}")
    print(f"   • Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"   • Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print()

    # Create model checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=paths.get_model_path("bert", args.doclevel),
        filename="best-model-{epoch:02d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        verbose=True,
    )

    # Create progress callback
    progress_callback = pl.callbacks.RichProgressBar()

    # Generate meaningful log directory name
    experiment_name = f"epochs_{args.nepochs}_batch_{args.batchsize}_lr_{args.lr:.0e}_wd_{args.weight_decay:.0e}"
    if args.warmup_steps > 0:
        experiment_name += f"_warmup_{args.warmup_steps}"
    if args.max_length != 512:
        experiment_name += f"_maxlen_{args.max_length}"
    if args.clean:
        experiment_name += "_clean"
    if args.noarznei:
        experiment_name += "_nomeds"

    log_dir = paths.get_log_path("bert", args.doclevel, experiment_name)

    # Create trainer with progress bars and better logging
    trainer = pl.Trainer(
        max_epochs=args.nepochs,
        callbacks=[checkpoint_callback, progress_callback],
        enable_progress_bar=True,
        log_every_n_steps=10,
        enable_model_summary=True,
        default_root_dir=log_dir,
    )

    if not args.testonly:
        print("🏋️ Starting training...")
        print(f"   • Training logs will be saved to: {log_dir}")
        print(f"   • Experiment name: {experiment_name}")
        print(f"   • Training for {args.nepochs} epochs")
        print(
            f"   • Model checkpoints will be saved to: {paths.get_model_path('bert', args.doclevel)}"
        )
        print()
        trainer.fit(model, loaders[0], loaders[1])
        print("\n✅ Training completed!")
    else:
        print("⏭️ Skipping training (test-only mode)")

    print("\n🧪 Running final evaluation...")
    test_results = trainer.test(model, loaders[2])

    print("\n🎯 Final Results:")
    if test_results:
        for key, value in test_results[0].items():
            if "test" in key:
                print(f"   • {key}: {value:.4f}")

    print(f"\n💾 Model saved to: {paths.get_model_path('bert', args.doclevel)}")
    print("🎉 BERT fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
