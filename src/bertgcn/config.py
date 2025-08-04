"""
Enhanced configuration management for BertGCN.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = OUTPUTS_DIR / "data"
MODELS_DIR = OUTPUTS_DIR / "models"
GRAPHS_DIR = DATA_DIR / "graphs"
DATASETS_DIR = DATA_DIR / "datasets"
CACHE_DIR = OUTPUTS_DIR / "cache"

# Model configuration
PRETRAINEDMODEL = os.getenv(
    "BERTGCN_MODEL_PATH", "/prj/doctoral_letters/PETGUI/med_bert_local"
)

# Default model fallback (in case the primary model is not available)
FALLBACK_MODEL = "bert-base-uncased"

# Graph building configuration
DEFAULT_VOCAB_MIN_FREQ = 2
DEFAULT_MAX_VOCAB_SIZE = 10000
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.2

# Create directories
for dir_path in [
    OUTPUTS_DIR,
    DATA_DIR,
    MODELS_DIR,
    GRAPHS_DIR,
    DATASETS_DIR,
    CACHE_DIR,
]:
    dir_path.mkdir(parents=True, exist_ok=True)


def get_paths():
    """Get project paths."""
    return {
        "project_root": PROJECT_ROOT,
        "outputs": OUTPUTS_DIR,
        "data": DATA_DIR,
        "models": MODELS_DIR,
        "graphs": GRAPHS_DIR,
        "datasets": DATASETS_DIR,
        "cache": CACHE_DIR,
    }


def get_model_path():
    """Get the model path with fallback."""
    model_path = Path(PRETRAINEDMODEL)
    if model_path.exists():
        return str(model_path)
    else:
        print(
            f"Warning: Primary model path {PRETRAINEDMODEL} not found. Using fallback: {FALLBACK_MODEL}"
        )
        return FALLBACK_MODEL


def get_graph_config():
    """Get graph building configuration."""
    return {
        "vocab_min_freq": DEFAULT_VOCAB_MIN_FREQ,
        "max_vocab_size": DEFAULT_MAX_VOCAB_SIZE,
        "train_ratio": DEFAULT_TRAIN_RATIO,
        "val_ratio": DEFAULT_VAL_RATIO,
        "test_ratio": DEFAULT_TEST_RATIO,
    }
