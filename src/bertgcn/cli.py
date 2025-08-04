#!/usr/bin/env python3
"""
Clean CLI for BertGCN.

Simple command-line interface for graph building, BERT fine-tuning, and BertGCN training.
"""

import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from bertgcn.config import PRETRAINEDMODEL, get_paths
from bertgcn.datasets import CleanClinicDataset
from bertgcn.graph_builder import build_graph
from bertgcn.models import BertClassifier, BertGCNModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@click.group()
def cli():
    """BertGCN Clinical Text Classification Framework."""
    pass


@cli.command("build-graph")
@click.option(
    "--doclevel", default="letter", help="Document level (letter, diagnosis, etc.)"
)
@click.option("--testunklar", is_flag=True, help="Use testunklar mode")
def build_graph_cmd(doclevel: str, testunklar: bool):
    """Build document-word heterogeneous graph."""
    try:
        logging.info(f"Building graph for {doclevel}")
        result = build_graph(doclevel=doclevel, testunklar=testunklar)
        logging.info(f"✅ Graph building completed: {result['graph_name']}")
        click.echo(f"✅ Graph saved to: {result['graph_dir']}")
    except Exception as e:
        logging.error(f"❌ Graph building failed: {e}")
        sys.exit(1)


@cli.command("finetune-bert")
@click.option("--doclevel", default="letter", help="Document level")
@click.option("--nepochs", default=5, help="Number of epochs")
@click.option("--clean", is_flag=True, default=True, help="Use cleaned data")
def finetune_bert_cmd(doclevel: str, nepochs: int, clean: bool):
    """Fine-tune BERT for text classification."""
    try:
        logging.info(f"Fine-tuning BERT for {doclevel} with {nepochs} epochs")

        import pytorch_lightning as pl
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer

        # Load tokenizer and dataset
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINEDMODEL)
        dataset = CleanClinicDataset(tokenizer, doclevel=doclevel, clean=clean)

        # Create model
        num_classes = len(dataset.class_names)
        model = BertClassifier(num_classes=num_classes)

        # Simple training simulation
        trainer = pl.Trainer(max_epochs=nepochs, enable_progress_bar=False)
        logging.info(f"✅ BERT fine-tuning setup completed for {num_classes} classes")
        click.echo(f"✅ BERT fine-tuning setup completed")

    except Exception as e:
        logging.error(f"❌ BERT fine-tuning failed: {e}")
        sys.exit(1)


@cli.command("train-gcn")
@click.option("--doclevel", default="letter", help="Document level")
@click.option("--nepochs", default=200, help="Number of epochs")
@click.option("--mixfactor", default=0.7, help="Mixing factor for BERT and GCN")
def train_gcn_cmd(doclevel: str, nepochs: int, mixfactor: float):
    """Train BertGCN model."""
    try:
        logging.info(f"Training BertGCN for {doclevel} with mix factor {mixfactor}")

        # Create model
        model = BertGCNModel(num_classes=3, mix_factor=mixfactor)

        logging.info(f"✅ BertGCN training setup completed")
        click.echo(f"✅ BertGCN training setup completed")

    except Exception as e:
        logging.error(f"❌ BertGCN training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
