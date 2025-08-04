"""
Configuration management for BertGCN.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = OUTPUTS_DIR / "data"
MODELS_DIR = OUTPUTS_DIR / "models"
GRAPHS_DIR = DATA_DIR / "graphs"
DATASETS_DIR = DATA_DIR / "datasets"

# Model path
PRETRAINEDMODEL = "/prj/doctoral_letters/PETGUI/med_bert_local"

# Create directories
for dir_path in [OUTPUTS_DIR, DATA_DIR, MODELS_DIR, GRAPHS_DIR, DATASETS_DIR]:
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
    }
