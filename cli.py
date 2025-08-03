"""
Modern CLI interface using Click for BertGCN project.

Provides clean, intuitive command-line interfaces for graph building and BERT fine-tuning.
"""

import os
import random
import warnings

import click
import numpy as np
import torch

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Model configuration
PRETRAINEDMODEL = "/prj/doctoral_letters/PETGUI/med_bert_local"


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
def build_graph(doclevel, testunklar, data):
    """Build document-word heterogeneous graph."""
    import logging

    from data_manager import create_data_matrices, save_graph_files
    from graph_builder import build_graph as _build_graph

    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

    adj_matrix, metadata, dataset = _build_graph(doclevel, testunklar)
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
    required=True,
    help="Document level to process",
)
@click.option("--nepochs", default=50, help="Number of training epochs")
@click.option("--batchsize", default=1, help="Training batch size")
@click.option("--clean", is_flag=True, help="Apply text cleaning and stopword removal")
@click.option("--noarznei", is_flag=True, help="Exclude medication names from text")
@click.option("--testonly", is_flag=True, help="Skip training, only run testing")
def finetune_bert(doclevel, nepochs, batchsize, clean, noarznei, testonly):
    """Fine-tune BERT model for clinical text classification."""
    from finetune_bert import main as _finetune_main

    # Create a namespace object to mimic argparse
    class Args:
        def __init__(self):
            self.doclevel = doclevel
            self.nepochs = nepochs
            self.batchsize = batchsize
            self.clean = clean
            self.noarznei = noarznei
            self.testonly = testonly

    _finetune_main(Args())


@cli.command()
@click.option(
    "--doclevel",
    type=click.Choice(["letter", "diagnosis", "riskfactor", "anamnesis"]),
    required=True,
    help="Document level to process",
)
@click.option("--nepochs", default=50, help="Number of training epochs")
@click.option(
    "--mixfactor",
    default=0.7,
    type=float,
    help="Mixing factor between BERT and GCN features",
)
@click.option("--clean", is_flag=True, help="Apply text cleaning and stopword removal")
@click.option("--testunklar", is_flag=True, help='Use "unklar" labels as test set')
@click.option("--testonly", is_flag=True, help="Skip training, only run testing")
def train_gcn(doclevel, nepochs, mixfactor, clean, testunklar, testonly):
    """Train BertGCN hybrid model for clinical text classification."""
    from train_bert_gcn import main as _train_gcn_main

    class Args:
        def __init__(self):
            self.doclevel = doclevel
            self.nepochs = nepochs
            self.mixfactor = mixfactor
            self.clean = clean
            self.testunklar = testunklar
            self.testonly = testonly

    _train_gcn_main(Args())


if __name__ == "__main__":
    cli()
