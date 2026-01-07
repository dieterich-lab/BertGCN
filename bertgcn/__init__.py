"""
BertGCN - Graph Convolutional Network with BERT embeddings for document classification.

This package provides tools to build and train BertGCN models for document classification.
"""

__version__ = "0.1.0"

# Keep imports minimal: avoid importing heavy optional dependencies (DGL, torchdata,
# transformers) at package import time. Import submodules explicitly where needed
# in tests or runtime code (e.g. `from bertgcn import model`).

from bertgcn.config import DATA_PATHS, DEFAULT_MODEL_PATH, MODEL_PATHS

__all__ = [
    "DATA_PATHS",
    "MODEL_PATHS",
    "DEFAULT_MODEL_PATH",
]
