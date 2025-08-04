"""
Configuration and path management for BertGCN project.

This module provides centralized configuration for data paths, model storage,
and project settings to ensure consistency across all components.
"""

import os
import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

# Environment settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Model configurations
PRETRAINEDMODEL = "/prj/doctoral_letters/PETGUI/med_bert_local"

DATADICT = {
    "train": "/prj/doctoral_letters/MIEdeep/corpus/cardiode/24_final_hq/tsv/CARDIODE400_main",
    "test": "/prj/doctoral_letters/MIEdeep/corpus/cardiode/24_final_hq/tsv/CARDIODE100_heldout",
    "medindcls": "/prj/doctoral_letters/MIEdeep/corpus/annotated_gold500/med_indication_all_RF_diag.csv",
}


def set_random_seeds(seed: int = 0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ProjectPaths:
    """Centralized path management for the BertGCN project."""

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize project paths."""
        if base_dir is None:
            # Go up to the project root from src/bertgcn/
            base_dir = Path(__file__).parent.parent.parent

        self.base_dir = Path(base_dir)

        # Main directories
        self.data_dir = self.base_dir / "outputs" / "data"
        self.models_dir = self.base_dir / "outputs" / "models"
        self.cache_dir = self.base_dir / "outputs" / "cache"
        self.logs_dir = self.base_dir / "outputs" / "logs"

        # Data subdirectories
        self.datasets_dir = self.data_dir / "datasets"
        self.graphs_dir = self.data_dir / "graphs"
        self.features_dir = self.data_dir / "features"

        # Model subdirectories
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.finetuned_dir = self.models_dir / "finetuned"
        self.gcn_dir = self.models_dir / "gcn"

        # Create all directories
        self._create_directories()

    def _create_directories(self):
        """Create all necessary directories."""
        dirs = [
            self.data_dir,
            self.models_dir,
            self.cache_dir,
            self.logs_dir,
            self.datasets_dir,
            self.graphs_dir,
            self.features_dir,
            self.checkpoints_dir,
            self.finetuned_dir,
            self.gcn_dir,
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_dataset_path(
        self,
        doclevel: str,
        model_type: str = "medbert",
        suffix: str = "",
        clean: bool = True,
    ) -> Path:
        """Get path for dataset file."""
        clean_suffix = "_clean" if clean else ""
        filename = f"medindcls_{model_type}_{doclevel}{suffix}{clean_suffix}.pkl"
        return self.datasets_dir / filename

    def get_graph_path(
        self, dataset_name: str, doclevel: str, testunklar: bool = False
    ) -> Path:
        """Get directory path for graph files."""
        suffix = "_testunklar" if testunklar else ""
        graph_name = f"{dataset_name}_{doclevel}{suffix}"
        return self.graphs_dir / graph_name

    def get_model_path(
        self, model_type: str, doclevel: str, experiment_name: Optional[str] = None
    ) -> Path:
        """Get path for model storage."""
        if model_type == "bert":
            base_path = self.finetuned_dir / doclevel
        elif model_type == "gcn":
            base_path = self.gcn_dir / doclevel
        else:
            base_path = self.checkpoints_dir / model_type / doclevel

        if experiment_name:
            base_path = base_path / experiment_name

        return base_path

    def get_cache_path(self, cache_type: str) -> Path:
        """Get path for cache files."""
        return self.cache_dir / cache_type

    def get_log_path(
        self, model_type: str, doclevel: str, experiment_name: Optional[str] = None
    ) -> Path:
        """Get path for PyTorch Lightning logs with meaningful names."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if experiment_name:
            log_name = f"{model_type}_{doclevel}_{experiment_name}_{timestamp}"
        else:
            log_name = f"{model_type}_{doclevel}_{timestamp}"

        log_path = self.logs_dir / "lightning_logs" / log_name
        log_path.mkdir(parents=True, exist_ok=True)
        return log_path


# Global instance for easy access
paths = ProjectPaths()


def get_paths() -> ProjectPaths:
    """Get the global paths instance."""
    return paths
