#!/usr/bin/env python3
"""
BertGCN Core Package

Production-ready BertGCN framework for clinical text classification.
"""

__version__ = "1.0.0"
__author__ = "Clinical AI Team"

# Import only the core working components
from .config import PRETRAINEDMODEL, get_paths, set_random_seeds
from .data import CleanClinicDataset
from .models import BertClassifier, BertGCN

__all__ = [
    "PRETRAINEDMODEL",
    "get_paths",
    "set_random_seeds",
    "CleanClinicDataset",
    "BertClassifier",
    "BertGCN",
]
