"""
BertGCN - Graph Convolutional Network with BERT embeddings for document classification.

This package provides tools to build and train BertGCN models for document classification.
"""

__version__ = "0.1.0"

from bertgcn.build_graph import GraphBuilder
from bertgcn.config import DATA_PATHS, DEFAULT_MODEL_PATH, MODEL_PATHS
from bertgcn.core import get_logger, setup_environment

__all__ = [
    "GraphBuilder",
    "DATA_PATHS",
    "MODEL_PATHS",
    "DEFAULT_MODEL_PATH",
    "get_logger",
    "setup_environment",
]
