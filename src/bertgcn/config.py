"""
Configuration for BertGCN project.

This module contains default configurations, paths and constants used throughout the project.
Configuration values can be overridden by environment variables or command-line arguments.
"""

import os
from pathlib import Path

# Data paths
# These paths should ideally be configurable via environment variables or config files
DATA_PATHS = {
    "train": "/prj/doctoral_letters/MIEdeep/corpus/cardiode/24_final_hq/tsv/CARDIODE400_main",
    "test": "/prj/doctoral_letters/MIEdeep/corpus/cardiode/24_final_hq/tsv/CARDIODE100_heldout",
    "medindcls": "/prj/doctoral_letters/MIEdeep/corpus/annotated_gold500/med_indication_all_RF_diag.csv",
}

# Model paths
MODEL_PATHS = {
    # "gbert": "deepset/gbert-base",
    "medbert": "/prj/doctoral_letters/PETGUI/med_bert_local",
}

# Default model
DEFAULT_MODEL_PATH = MODEL_PATHS["medbert"]

# Default parameters for graph building
DEFAULT_WINDOW_SIZE = 20
DEFAULT_BATCH_SIZE = 1000
DEFAULT_USE_BIDIRECTIONAL_TFIDF = True
DEFAULT_MIN_PMI = 0.0
DEFAULT_SEED = 0
DEFAULT_DOCUMENT_LEVEL = "letter"

# Environment configuration
# Control parallelism in tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
