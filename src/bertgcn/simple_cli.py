#!/usr/bin/env python3
"""
Modern CLI for BertGCN with Click

Clean, minimal CLI interface for the migrated package structure.
"""

import logging
import warnings
from pathlib import Path

import click

from bertgcn.config import set_random_seeds
from bertgcn.data_manager import create_data_matrices, save_graph_files
from bertgcn.finetune_bert import main as finetune_bert_main
from bertgcn.graph_builder import build_graph
from bertgcn.train_bertgcn import main as train_bertgcn_main

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

# Set seeds for reproducibility
set_random_seeds(0)


@click.group()
def cli():
    """BertGCN: Document-Word Graph Networks for Clinical Text Classification"""
    pass


@cli.command()
@click.option(
    "--doclevel",
    type=click.Choice(["letter", "diagnosis", "riskfactor", "anamnesis"]),
    required=True,
    help="Document level to process",
)
@click.option("--testunklar", is_flag=True, help='Use "unklar" labels as test set')
@click.option(
    "--data",
    type=click.Choice(["MIC", "CSC", "Patho"]),
    default="MIC",
    help="Dataset type",
)
def build_graph_cmd(doclevel, testunklar, data):
    """Build document-word heterogeneous graph."""
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

    adj_matrix, metadata, dataset = build_graph(doclevel, testunklar)
    data_matrices = create_data_matrices(dataset, metadata, metadata["embed_dim"])
    dataset_name = f"medindcls_{doclevel}"
    save_graph_files(
        adj_matrix, data_matrices, metadata, dataset_name, doclevel, testunklar
    )

    click.echo(f"✅ Graph built: {metadata['node_size']} nodes, {adj_matrix.nnz} edges")
    click.echo(
        f"📊 Train/Val/Test: {metadata['train_size']}/{metadata['val_size']}/{metadata['test_size']}"
    )


@cli.command()
@click.option(
    "--doclevel",
    type=click.Choice(["letter", "diagnosis", "riskfactor", "anamnesis"]),
    help="Document level to process",
)
@click.option("--nepochs", type=int, help="Number of training epochs")
@click.option("--batchsize", type=int, help="Training batch size")
@click.option(
    "--lr",
    "--learning-rate",
    type=float,
    help="Learning rate for BERT fine-tuning",
)
@click.option("--weight-decay", type=float, help="Weight decay for AdamW optimizer")
@click.option(
    "--warmup-steps",
    type=int,
    help="Number of warmup steps for learning rate scheduling",
)
@click.option(
    "--max-length",
    type=int,
    help="Maximum sequence length for tokenization",
)
@click.option("--clean", is_flag=True, help="Apply text cleaning and stopword removal")
@click.option("--noarznei", is_flag=True, help="Exclude medication names from text")
@click.option("--testonly", is_flag=True, help="Skip training, only run testing")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to YAML config file (overrides other options)",
)
def finetune_bert(
    doclevel,
    nepochs,
    batchsize,
    lr,
    weight_decay,
    warmup_steps,
    max_length,
    clean,
    noarznei,
    testonly,
    config,
):
    """Fine-tune BERT model for clinical text classification."""
    from bertgcn.config_loader import (
        create_bert_config_from_file,
        get_default_config_path,
    )

    # Load configuration
    if config:
        # Use provided config file
        bert_config = create_bert_config_from_file(config)
        click.echo(f"📁 Using config file: {config}")
    else:
        # Use default config file or fallback to CLI args
        default_config_path = get_default_config_path("bert_finetune")
        try:
            bert_config = create_bert_config_from_file(default_config_path)
            click.echo(f"📁 Using default config: {default_config_path}")
        except FileNotFoundError:
            # Fallback to CLI arguments
            from bertgcn.config_loader import BertConfig

            bert_config = BertConfig()
            click.echo("⚙️  Using default configuration values")

    # Override config with CLI arguments if provided
    if doclevel is not None:
        bert_config.doclevel = doclevel
    if nepochs is not None:
        bert_config.epochs = nepochs
    if batchsize is not None:
        bert_config.batch_size = batchsize
    if lr is not None:
        bert_config.learning_rate = lr
    if weight_decay is not None:
        bert_config.weight_decay = weight_decay
    if warmup_steps is not None:
        bert_config.warmup_steps = warmup_steps
    if max_length is not None:
        bert_config.max_length = max_length
    if clean:
        bert_config.clean = clean
    if noarznei:
        bert_config.nomeds = noarznei

    # Print configuration summary
    click.echo(f"🤖 Model: {bert_config.pretrained_model}")
    click.echo(f"📚 Document level: {bert_config.doclevel}")
    click.echo(f"⏱️  Epochs: {bert_config.epochs}")
    click.echo(f"📦 Batch size: {bert_config.batch_size}")
    click.echo(f"📈 Learning rate: {bert_config.learning_rate}")

    # Create a namespace object compatible with the existing main function
    class Args:
        def __init__(self):
            self.doclevel = bert_config.doclevel
            self.nepochs = bert_config.epochs
            self.batchsize = bert_config.batch_size
            self.lr = bert_config.learning_rate
            self.weight_decay = bert_config.weight_decay
            self.warmup_steps = bert_config.warmup_steps
            self.max_length = bert_config.max_length
            self.clean = bert_config.clean
            self.noarznei = bert_config.nomeds
            self.testonly = testonly
            # Add config object for advanced features
            self.config = bert_config

    finetune_bert_main(Args())


@cli.command()
@click.option(
    "--doclevel",
    type=click.Choice(["letter", "diagnosis", "riskfactor", "anamnesis"]),
    help="Document level to process",
)
@click.option("--nepochs", type=int, help="Number of training epochs")
@click.option(
    "--mixfactor",
    type=float,
    help="Mixing factor between BERT and GCN features",
)
@click.option("--clean", is_flag=True, help="Apply text cleaning and stopword removal")
@click.option("--testunklar", is_flag=True, help='Use "unklar" labels as test set')
@click.option("--testonly", is_flag=True, help="Skip training, only run testing")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to YAML config file (overrides other options)",
)
def train_gcn(doclevel, nepochs, mixfactor, clean, testunklar, testonly, config):
    """Train BertGCN hybrid model for clinical text classification."""
    from bertgcn.config_loader import (
        create_bertgcn_config_from_file,
        get_default_config_path,
    )

    # Load configuration
    if config:
        # Use provided config file
        gcn_config = create_bertgcn_config_from_file(config)
        click.echo(f"📁 Using config file: {config}")
    else:
        # Use default config file or fallback to CLI args
        default_config_path = get_default_config_path("bertgcn_train")
        try:
            gcn_config = create_bertgcn_config_from_file(default_config_path)
            click.echo(f"📁 Using default config: {default_config_path}")
        except FileNotFoundError:
            # Fallback to CLI arguments
            from bertgcn.config_loader import BertGCNConfig

            gcn_config = BertGCNConfig()
            click.echo("⚙️  Using default configuration values")

    # Override config with CLI arguments if provided
    if doclevel is not None:
        gcn_config.doclevel = doclevel
    if nepochs is not None:
        gcn_config.epochs = nepochs
    if mixfactor is not None:
        gcn_config.mix_factor = mixfactor
    if clean:
        gcn_config.clean = clean
    if testunklar:
        gcn_config.testunklar = testunklar

    # Print configuration summary
    click.echo(f"🤖 Model: {gcn_config.pretrained_model}")
    click.echo(f"📚 Document level: {gcn_config.doclevel}")
    click.echo(f"⏱️  Epochs: {gcn_config.epochs}")
    click.echo(f"🔀 Mix factor: {gcn_config.mix_factor}")
    click.echo(f"🧠 GCN layers: {gcn_config.gcn_layers}")

    class Args:
        def __init__(self):
            self.doclevel = gcn_config.doclevel
            self.nepochs = gcn_config.epochs
            self.mixfactor = gcn_config.mix_factor
            self.clean = gcn_config.clean
            self.testunklar = gcn_config.testunklar
            self.testonly = testonly
            # Add config object for advanced features
            self.config = gcn_config

    train_bertgcn_main(Args())


if __name__ == "__main__":
    cli()
