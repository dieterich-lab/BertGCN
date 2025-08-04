#!/usr/bin/env python3
"""
BertGCN: Clinical Text Classification with Graph Neural Networks

A clean, professional implementation for clinical text classification
using BERT embeddings and Graph Convolutional Networks.
"""

__version__ = "0.1.0"
__author__ = "BertGCN Team"

from .config import get_paths, get_graph_config, get_model_path

# Core imports for graph building
from .datasets import CleanClinicDataset
from .graph_builder import build_graph_enhanced as build_graph

__all__ = [
    "CleanClinicDataset",
    "build_graph",
    "get_paths",
    "get_enhanced_paths",
    "get_model_path",
    "get_graph_config",
]
